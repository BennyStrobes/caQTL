import numpy as np
import pdb
import sys
import time
from numba import njit



@njit(cache=True)
def spike_and_slab_variant_sweep(z, se, ld_mat, prior_mean, prior_mean_sq, prior_var, extra_precision, extra_linear, log_prior_odds, alphas, mus, variances, scaled_beta, ld_scaled_beta):
    """
    One sequential sweep of spike-and-slab coordinate updates across variants (compiled with numba).
    RSS likelihood: effects ~ N(S R S^-1 beta, S R S) with S = diag(se), z = effects/se
    Prior: beta_j ~ N(prior_mean_j, prior_var_j) if included. prior_mean_sq_j is the expected squared prior mean (E[m^2], not E[m]^2)
    extra_precision and extra_linear: additional gaussian information on beta_j (from other layers of the model)
    scaled_beta = E[beta]/se and ld_scaled_beta = R*scaled_beta at start of sweep
    Updates alphas, mus, variances (and scaled_beta, ld_scaled_beta) in place
    ld_mat is assumed symmetric (row variant_iter is used in place of column variant_iter, as rows are contiguous in memory)
    """
    n_variants = len(z)
    # Loop through variants (updates are sequential)
    for variant_iter in range(n_variants):
        ld_diag = ld_mat[variant_iter, variant_iter]
        # Residual z-score after removing effects of all other variants
        resid_z = z[variant_iter] - (ld_scaled_beta[variant_iter] - ld_diag*scaled_beta[variant_iter])

        # Update variational posterior of slab
        new_var = 1.0/(ld_diag/(se[variant_iter]*se[variant_iter]) + 1.0/prior_var[variant_iter] + extra_precision[variant_iter])
        new_mu = new_var*(resid_z/se[variant_iter] + prior_mean[variant_iter]/prior_var[variant_iter] + extra_linear[variant_iter])

        # Update inclusion probability
        log_odds = log_prior_odds + 0.5*np.log(new_var/prior_var[variant_iter]) + (new_mu*new_mu)/(2.0*new_var) - prior_mean_sq[variant_iter]/(2.0*prior_var[variant_iter])
        if log_odds > 500.0:
            log_odds = 500.0
        if log_odds < -500.0:
            log_odds = -500.0
        new_alpha = 1.0/(1.0 + np.exp(-log_odds))

        variances[variant_iter] = new_var
        mus[variant_iter] = new_mu
        alphas[variant_iter] = new_alpha

        # Update LD-weighted sum of effects to reflect new effect of this variant
        new_scaled_beta = new_alpha*new_mu/se[variant_iter]
        delta = new_scaled_beta - scaled_beta[variant_iter]
        if delta != 0.0:
            for other_variant_iter in range(n_variants):
                ld_scaled_beta[other_variant_iter] = ld_scaled_beta[other_variant_iter] + ld_mat[variant_iter, other_variant_iter]*delta
        scaled_beta[variant_iter] = new_scaled_beta
    return


class JOINT_EQTL_CAQTL_EXPRESSION_PREDICTION(object):
    """
    Joint model of eQTLs and caQTLs from summary statistics, used for genetic prediction of expression. Fit once across all genes.
    Posterior mean eQTL effects (eqtl_alphas*eqtl_mus) are the expression prediction weights. Inclusion probabilities are NOT fine-mapping PIPs (not valid under tight LD).
    """

    def __init__(self, tol=1e-4, max_iter=300, mediation_indicator=False, mediation_burn_in=5):
        self.tol = tol
        self.max_iter = max_iter
        self.mediation_indicator = mediation_indicator  # If True, each peak-gene pair gets an additional binary variable indicating whether the peak mediates eQTL effects on the gene
        self.mediation_burn_in = mediation_burn_in  # Number of initial iterations where all mediation indicators are held at 1 (only used if mediation_indicator is True)

        # Filled in by fit()
        self.gene_to_data = None     # dict: gene_id -> dict with keys variant_ids_file, ld_file, eqtl_effects, eqtl_se, caqtl_effects, caqtl_se, peak_effects, peak_se
        self.gene_ids = None         # sorted array of gene ids
        self.converged = None
        self.n_iter = None

    def fit(self, gene_to_data):
        self.gene_to_data = gene_to_data
        self.gene_ids = np.sort(np.asarray(list(gene_to_data.keys())))
        self.initialize_model()

        for self.n_iter in range(self.max_iter):
            # Hyperparameters at start of iteration (used to assess convergence)
            prev_hyperparameters = self.get_hyperparameter_vector()

            # Loop through genes
            for gene_id in self.gene_ids:
                # Extract LD for gene
                ld_mat = np.load(self.gene_to_data[gene_id]['ld_file'])
                # Update eQTL effects for this gene
                self.update_eqtl_effects(gene_id, ld_mat)
                # Update caQTL effects for this gene
                self.update_caqtl_effects(gene_id, ld_mat)
                # Update peak-gene effects for this gene (and mediation indicators, if used)
                self.update_peak_gene_effects(gene_id)


            # Now update hyperparameters (pi, sigma2, lambda) across all genes
            self.update_pis()
            self.update_lambda()
            self.update_sigma_sqs()

            # Print summary of hyperparameters at this iteration
            self.print_hyperparameter_summary()

            # Check for convergence: relative change in every hyperparameter is less than tol
            hyperparameters = self.get_hyperparameter_vector()
            relative_changes = np.abs(hyperparameters - prev_hyperparameters)/np.maximum(np.abs(prev_hyperparameters), 1e-300)
            print('max relative change in hyperparameters=' + str(np.max(relative_changes)))
            sys.stdout.flush()
            if np.max(relative_changes) < self.tol and (self.mediation_indicator == False or self.n_iter > self.mediation_burn_in):
                self.converged = True
                print('Converged after ' + str(self.n_iter + 1) + ' iterations')
                break

        return self

    def get_hyperparameter_vector(self):
        """
        Vector of current values of hyperparameters (shared across genes). Used to assess convergence.
        """
        hyperparameters = [self.lambda_mu, self.lambda_var, self.eqtl_pi, self.caqtl_pi, self.peak_pi, self.sigma2_eqtl, self.sigma2_caqtl, self.sigma2_peak_gene]
        if self.mediation_indicator:
            hyperparameters.append(self.mediation_pi)
        return np.asarray(hyperparameters)

    def print_hyperparameter_summary(self):
        """
        Print current values of hyperparameters (shared across genes).
        """
        print('##########################')
        print('Iteration ' + str(self.n_iter))
        print('lambda: mean=' + str(self.lambda_mu) + '  sd=' + str(np.sqrt(self.lambda_var)))
        print('eqtl_pi=' + str(self.eqtl_pi) + '  caqtl_pi=' + str(self.caqtl_pi) + '  peak_pi=' + str(self.peak_pi))
        print('sigma2_eqtl=' + str(self.sigma2_eqtl) + '  sigma2_caqtl=' + str(self.sigma2_caqtl) + '  sigma2_peak_gene=' + str(self.sigma2_peak_gene))
        if self.mediation_indicator:
            print('mediation_pi=' + str(self.mediation_pi))
        sys.stdout.flush()
        return

    def compute_pred_eqtl_moments(self, gene_id):
        """
        Compute first and second moments of chromatin-predicted eQTL effect m_j = sum_k w_k*g_k*b_kj (before scaling by lambda) for single gene.
        w_k is the mediation indicator (mediation_probs are all 1 if mediation_indicator is False)
        """
        peak_gene_effects = self.mediation_probs[gene_id]*self.peak_gene_mus[gene_id]*self.peak_gene_alphas[gene_id]  # E[w*g]
        peak_gene_effects_sq = self.mediation_probs[gene_id]*self.peak_gene_alphas[gene_id]*(np.square(self.peak_gene_mus[gene_id]) + self.peak_gene_vars[gene_id])  # E[(w*g)^2]
        variant_peak_effects = self.caqtl_mus[gene_id]*self.caqtl_alphas[gene_id]  # E[b]
        variant_peak_effects_sq = self.caqtl_alphas[gene_id]*(np.square(self.caqtl_mus[gene_id]) + self.caqtl_vars[gene_id])  # E[b^2]
        pred_eqtl = np.dot(peak_gene_effects, variant_peak_effects)  # E[m]
        pred_eqtl_sq = np.square(pred_eqtl) + np.dot(peak_gene_effects_sq, variant_peak_effects_sq) - np.dot(np.square(peak_gene_effects), np.square(variant_peak_effects))  # E[m^2]
        return pred_eqtl, pred_eqtl_sq

    def update_pis(self):
        """
        Update prior inclusion probabilities (shared across genes). Each is the average posterior inclusion probability.
        """
        eqtl_alpha_sum = 0.0
        eqtl_count = 0.0
        caqtl_alpha_sum = 0.0
        caqtl_count = 0.0
        peak_alpha_sum = 0.0
        peak_count = 0.0
        # Loop through genes
        for gene_id in self.gene_ids:
            eqtl_alpha_sum += np.sum(self.eqtl_alphas[gene_id])
            eqtl_count += self.eqtl_alphas[gene_id].size
            caqtl_alpha_sum += np.sum(self.caqtl_alphas[gene_id])
            caqtl_count += self.caqtl_alphas[gene_id].size
            peak_alpha_sum += np.sum(self.peak_gene_alphas[gene_id])
            peak_count += self.peak_gene_alphas[gene_id].size

        if eqtl_count > 0:
            self.eqtl_pi = eqtl_alpha_sum/eqtl_count
        if caqtl_count > 0:
            self.caqtl_pi = caqtl_alpha_sum/caqtl_count
        if peak_count > 0:
            self.peak_pi = peak_alpha_sum/peak_count

        # Prior probability a peak mediates (only updated once mediation indicators are being updated)
        if self.mediation_indicator and self.n_iter >= self.mediation_burn_in:
            mediation_prob_sum = 0.0
            for gene_id in self.gene_ids:
                mediation_prob_sum += np.sum(self.mediation_probs[gene_id])
            if peak_count > 0:
                self.mediation_pi = mediation_prob_sum/peak_count
        return

    def update_lambda(self):
        """
        Update variational posterior (gaussian) of link scale parameter lambda (shared across genes).
        """
        # Prior on lambda
        lambda_prior_mean = 0.0
        lambda_prior_var = 100.0

        # eQTL prior: beta_j ~ N(lambda*m_j, sigma2_eqtl) at variants with an eQTL effect (hence weighting by eqtl_alphas)
        precision_sum = 0.0
        linear_sum = 0.0
        # Loop through genes
        for gene_id in self.gene_ids:
            pred_eqtl, pred_eqtl_sq = self.compute_pred_eqtl_moments(gene_id)
            precision_sum += np.sum(self.eqtl_alphas[gene_id]*pred_eqtl_sq)
            linear_sum += np.sum(self.eqtl_alphas[gene_id]*self.eqtl_mus[gene_id]*pred_eqtl)

        self.lambda_var = 1.0/(1.0/lambda_prior_var + precision_sum/self.sigma2_eqtl)
        self.lambda_mu = self.lambda_var*(lambda_prior_mean/lambda_prior_var + linear_sum/self.sigma2_eqtl)
        return

    def update_sigma_sqs(self):
        """
        Update slab variances (shared across genes). Each is the average expected squared deviation of an effect from its prior mean, across included effects.
        """
        lambda_sq = np.square(self.lambda_mu) + self.lambda_var  # E[lambda^2]
        eqtl_sq_sum = 0.0
        eqtl_alpha_sum = 0.0
        caqtl_sq_sum = 0.0
        caqtl_alpha_sum = 0.0
        peak_sq_sum = 0.0
        peak_alpha_sum = 0.0
        # Loop through genes
        for gene_id in self.gene_ids:
            # eQTL: E[(beta_j - lambda*m_j)^2]
            pred_eqtl, pred_eqtl_sq = self.compute_pred_eqtl_moments(gene_id)
            eqtl_sq_dev = np.square(self.eqtl_mus[gene_id]) + self.eqtl_vars[gene_id] - 2.0*self.eqtl_mus[gene_id]*self.lambda_mu*pred_eqtl + lambda_sq*pred_eqtl_sq
            eqtl_sq_sum += np.sum(self.eqtl_alphas[gene_id]*eqtl_sq_dev)
            eqtl_alpha_sum += np.sum(self.eqtl_alphas[gene_id])
            # caQTL: E[b_kj^2]
            caqtl_sq_sum += np.sum(self.caqtl_alphas[gene_id]*(np.square(self.caqtl_mus[gene_id]) + self.caqtl_vars[gene_id]))
            caqtl_alpha_sum += np.sum(self.caqtl_alphas[gene_id])
            # Peak-gene: E[g_k^2]
            peak_sq_sum += np.sum(self.peak_gene_alphas[gene_id]*(np.square(self.peak_gene_mus[gene_id]) + self.peak_gene_vars[gene_id]))
            peak_alpha_sum += np.sum(self.peak_gene_alphas[gene_id])

        if eqtl_alpha_sum > 0:
            self.sigma2_eqtl = eqtl_sq_sum/eqtl_alpha_sum
        if caqtl_alpha_sum > 0:
            self.sigma2_caqtl = caqtl_sq_sum/caqtl_alpha_sum
        if peak_alpha_sum > 0:
            self.sigma2_peak_gene = peak_sq_sum/peak_alpha_sum
        return

    def update_peak_gene_effects(self, gene_id):
        """
        Update variational posterior parameters for peak-gene effect sizes for single gene.
        """
        # Peak-gene likelihood info: peak_effects ~ N(g, peak_se^2), independent across peaks
        peak_effects = self.gene_to_data[gene_id]['peak_effects']
        peak_se = self.gene_to_data[gene_id]['peak_se']
        n_peaks_per_gene = len(peak_effects)

        # Prior on peak-gene effects
        peak_gene_prior_mean = 0.0
        peak_gene_prior_var = self.sigma2_peak_gene

        # Expected values of other model variables needed for the update
        eqtl_alphas = self.eqtl_alphas[gene_id]
        eqtl_mus = self.eqtl_mus[gene_id]
        lambda_sq = np.square(self.lambda_mu) + self.lambda_var  # E[lambda^2]
        variant_peak_effects = self.caqtl_mus[gene_id]*self.caqtl_alphas[gene_id]  # E[b]
        variant_peak_effects_sq = self.caqtl_alphas[gene_id]*(np.square(self.caqtl_mus[gene_id]) + self.caqtl_vars[gene_id])  # E[b^2]
        mediation_probs = self.mediation_probs[gene_id]  # E[w] (w is mediation indicator; all 1 if mediation_indicator is False)
        peak_gene_effects = mediation_probs*self.peak_gene_mus[gene_id]*self.peak_gene_alphas[gene_id]  # E[w*g]
        # Chromatin-predicted eQTL effect (before scaling by lambda) summed across all peaks
        pred_eqtl = np.dot(peak_gene_effects, variant_peak_effects)

        log_prior_odds = np.log(self.peak_pi) - np.log(1.0 - self.peak_pi)
        update_mediation_probs = self.mediation_indicator and self.n_iter >= self.mediation_burn_in
        if update_mediation_probs:
            mediation_log_prior_odds = np.log(self.mediation_pi) - np.log(1.0 - self.mediation_pi)
        # Loop through peaks (updates are sequential)
        for peak_iter in range(n_peaks_per_gene):
            # Chromatin-predicted eQTL effect from all other peaks
            pred_eqtl_other_peaks = pred_eqtl - peak_gene_effects[peak_iter]*variant_peak_effects[peak_iter, :]

            # Contribution of eQTL prior to the peak-gene effect (acts as an extra gaussian observation of the peak-gene effect)
            # eQTL prior only applies at variants with an eQTL effect (hence weighting by eqtl_alphas)
            eqtl_prior_precision = lambda_sq*np.sum(eqtl_alphas*variant_peak_effects_sq[peak_iter, :])/self.sigma2_eqtl
            eqtl_prior_linear = np.sum(eqtl_alphas*variant_peak_effects[peak_iter, :]*(self.lambda_mu*eqtl_mus - lambda_sq*pred_eqtl_other_peaks))/self.sigma2_eqtl

            # Update variational posterior of slab
            # eQTL prior is only informative about the peak-gene effect if the peak mediates (hence weighting by mediation_probs)
            new_var = 1.0/(1.0/np.square(peak_se[peak_iter]) + 1.0/peak_gene_prior_var + mediation_probs[peak_iter]*eqtl_prior_precision)
            new_mu = new_var*(peak_effects[peak_iter]/np.square(peak_se[peak_iter]) + peak_gene_prior_mean/peak_gene_prior_var + mediation_probs[peak_iter]*eqtl_prior_linear)

            # Update inclusion probability
            log_odds = log_prior_odds + 0.5*np.log(new_var/peak_gene_prior_var) + np.square(new_mu)/(2.0*new_var) - np.square(peak_gene_prior_mean)/(2.0*peak_gene_prior_var)
            new_alpha = 1.0/(1.0 + np.exp(-np.clip(log_odds, -500, 500)))

            self.peak_gene_vars[gene_id][peak_iter] = new_var
            self.peak_gene_mus[gene_id][peak_iter] = new_mu
            self.peak_gene_alphas[gene_id][peak_iter] = new_alpha

            # Update mediation indicator: difference in expected eQTL prior with peak mediating vs not mediating
            if update_mediation_probs:
                mediation_log_odds = mediation_log_prior_odds + new_alpha*new_mu*eqtl_prior_linear - 0.5*new_alpha*(np.square(new_mu) + new_var)*eqtl_prior_precision
                mediation_probs[peak_iter] = 1.0/(1.0 + np.exp(-np.clip(mediation_log_odds, -500, 500)))

            # Update chromatin-predicted eQTL effect to reflect new peak-gene effect of this peak
            peak_gene_effects[peak_iter] = mediation_probs[peak_iter]*new_alpha*new_mu
            pred_eqtl = pred_eqtl_other_peaks + peak_gene_effects[peak_iter]*variant_peak_effects[peak_iter, :]
        return

    def update_caqtl_effects(self, gene_id, ld_mat):
        """
        Update variational posterior parameters for caQTL effect sizes for single gene.
        """
        # caQTL likelihood info
        caqtl_effects = self.gene_to_data[gene_id]['caqtl_effects']
        caqtl_se = self.gene_to_data[gene_id]['caqtl_se']
        n_peaks_per_gene = caqtl_effects.shape[0]
        n_variants_per_gene = caqtl_effects.shape[1]

        # Prior on caqtls
        caqtl_prior_mean = 0.0
        caqtl_prior_var = self.sigma2_caqtl

        # Expected values of other model variables needed for the update
        eqtl_alphas = self.eqtl_alphas[gene_id]
        eqtl_mus = self.eqtl_mus[gene_id]
        peak_gene_effects = self.mediation_probs[gene_id]*self.peak_gene_mus[gene_id]*self.peak_gene_alphas[gene_id]  # E[w*g] (w is mediation indicator)
        peak_gene_effects_sq = self.mediation_probs[gene_id]*self.peak_gene_alphas[gene_id]*(np.square(self.peak_gene_mus[gene_id]) + self.peak_gene_vars[gene_id])  # E[(w*g)^2]
        lambda_sq = np.square(self.lambda_mu) + self.lambda_var  # E[lambda^2]
        variant_peak_effects = self.caqtl_mus[gene_id]*self.caqtl_alphas[gene_id]  # E[b]
        # Chromatin-predicted eQTL effect (before scaling by lambda) summed across all peaks
        pred_eqtl = np.dot(peak_gene_effects, variant_peak_effects)

        log_prior_odds = np.log(self.caqtl_pi) - np.log(1.0 - self.caqtl_pi)
        # Prior is the same for every variant
        caqtl_prior_mean_vec = np.ones(n_variants_per_gene)*caqtl_prior_mean
        caqtl_prior_mean_sq_vec = np.square(caqtl_prior_mean_vec)
        caqtl_prior_var_vec = np.ones(n_variants_per_gene)*caqtl_prior_var
        # Loop through peaks
        for peak_iter in range(n_peaks_per_gene):
            peak_caqtl_se = caqtl_se[peak_iter, :]
            caqtl_z = caqtl_effects[peak_iter, :]/peak_caqtl_se

            # Chromatin-predicted eQTL effect from all other peaks
            pred_eqtl_other_peaks = pred_eqtl - peak_gene_effects[peak_iter]*variant_peak_effects[peak_iter, :]

            # Contribution of eQTL prior to the caQTL effect at each variant (acts as an extra gaussian observation of the caQTL effect)
            # eQTL prior only applies at variants with an eQTL effect (hence weighting by eqtl_alphas)
            eqtl_prior_precision = eqtl_alphas*lambda_sq*peak_gene_effects_sq[peak_iter]/self.sigma2_eqtl
            eqtl_prior_linear = eqtl_alphas*peak_gene_effects[peak_iter]*(self.lambda_mu*eqtl_mus - lambda_sq*pred_eqtl_other_peaks)/self.sigma2_eqtl

            # RSS likelihood: caqtl_effects ~ N(S R S^-1 b, S R S) with S = diag(caqtl_se)
            scaled_beta = variant_peak_effects[peak_iter, :]/peak_caqtl_se  # E[b]/se
            ld_scaled_beta = np.dot(ld_mat, scaled_beta)
            # Loop through variants (updates are sequential). Updates caqtl_alphas, caqtl_mus, caqtl_vars of this peak in place
            spike_and_slab_variant_sweep(caqtl_z, peak_caqtl_se, ld_mat, caqtl_prior_mean_vec, caqtl_prior_mean_sq_vec, caqtl_prior_var_vec, eqtl_prior_precision, eqtl_prior_linear, log_prior_odds, self.caqtl_alphas[gene_id][peak_iter, :], self.caqtl_mus[gene_id][peak_iter, :], self.caqtl_vars[gene_id][peak_iter, :], scaled_beta, ld_scaled_beta)

            # Update chromatin-predicted eQTL effect to reflect new caQTL effects of this peak
            variant_peak_effects[peak_iter, :] = self.caqtl_mus[gene_id][peak_iter, :]*self.caqtl_alphas[gene_id][peak_iter, :]
            pred_eqtl = pred_eqtl_other_peaks + peak_gene_effects[peak_iter]*variant_peak_effects[peak_iter, :]
        return



    def update_eqtl_effects(self, gene_id, ld_mat):
        """
        Update variational posterior parameters for eQTL effect sizes for single gen.
        """
        # eQTL likelihood info
        eqtl_effects = self.gene_to_data[gene_id]['eqtl_effects']
        eqtl_se = self.gene_to_data[gene_id]['eqtl_se']
        n_variants_per_gene = len(eqtl_effects)
        n_peaks_per_gene = self.gene_to_data[gene_id]['caqtl_effects'].shape[0]

        # Prior on eqtls
        if n_peaks_per_gene == 0:
            eqtl_prior_mean = np.zeros(n_variants_per_gene)
            eqtl_prior_mean_sq = np.zeros(n_variants_per_gene)
            eqtl_prior_var = np.ones(n_variants_per_gene) * self.sigma2_eqtl
        else:
            peak_gene_effects = self.mediation_probs[gene_id]*self.peak_gene_mus[gene_id]*self.peak_gene_alphas[gene_id]  # E[w*g] (w is mediation indicator)
            variant_peak_effects = self.caqtl_mus[gene_id]*self.caqtl_alphas[gene_id]
            eqtl_prior_mean = self.lambda_mu*np.dot(peak_gene_effects, variant_peak_effects)
            eqtl_prior_var = np.ones(n_variants_per_gene) * self.sigma2_eqtl
            # Expected squared prior mean (E[m^2], not E[m]^2). Needed for the inclusion probability update
            peak_gene_effects_sq = self.mediation_probs[gene_id]*self.peak_gene_alphas[gene_id]*(np.square(self.peak_gene_mus[gene_id]) + self.peak_gene_vars[gene_id])  # E[(w*g)^2]
            variant_peak_effects_sq = self.caqtl_alphas[gene_id]*(np.square(self.caqtl_mus[gene_id]) + self.caqtl_vars[gene_id])
            pred_sq = np.square(np.dot(peak_gene_effects, variant_peak_effects)) + np.dot(peak_gene_effects_sq, variant_peak_effects_sq) - np.dot(np.square(peak_gene_effects), np.square(variant_peak_effects))
            eqtl_prior_mean_sq = (np.square(self.lambda_mu) + self.lambda_var)*pred_sq

        # Update variational posterior parameters for eQTL effect sizes
        # RSS likelihood: eqtl_effects ~ N(S R S^-1 beta, S R S) with S = diag(eqtl_se)
        # beta is on scale of standardized genotype and un-standardized expression (same scale as eqtl_effects)
        eqtl_z = eqtl_effects/eqtl_se
        scaled_beta = self.eqtl_alphas[gene_id]*self.eqtl_mus[gene_id]/eqtl_se  # E[beta]/se
        ld_scaled_beta = np.dot(ld_mat, scaled_beta)
        log_prior_odds = np.log(self.eqtl_pi) - np.log(1.0 - self.eqtl_pi)
        # No additional gaussian information on eQTL effects from other layers of the model
        no_extra_info = np.zeros(n_variants_per_gene)
        # Loop through variants (updates are sequential). Updates eqtl_alphas, eqtl_mus, eqtl_vars in place
        spike_and_slab_variant_sweep(eqtl_z, eqtl_se, ld_mat, eqtl_prior_mean, eqtl_prior_mean_sq, eqtl_prior_var, no_extra_info, no_extra_info, log_prior_odds, self.eqtl_alphas[gene_id], self.eqtl_mus[gene_id], self.eqtl_vars[gene_id], scaled_beta, ld_scaled_beta)
        return


    def initialize_model(self):
        """
        Initialize model parameters and hyperparameters.
        """
        self.converged = False
        self.n_iter = 0

        self.n_genes = len(self.gene_ids)
        # Model variables
        # Prior variables
        self.eqtl_pi = 0.1 # Prior inclusion probability eqtl effect size
        self.caqtl_pi = 0.1 # Prior inclusion probability caqtl effect size
        self.peak_pi = 0.1 # Prior inclusion probability peak effect size
        self.lambda_mu = 1.0 # Variational posterior mean of link scale parameter lambda
        self.lambda_var = 10.0 # Variational posterior variance of link scale parameter lambda
        self.sigma2_eqtl = 0.1 # Variance parameter defining the prior on eqtl effect sizes conditional on caqtl effects
        self.sigma2_caqtl = 0.1 # Variance parameter defining the prior on caqtl effect sizes
        self.sigma2_peak_gene = 0.1 # Variance parameter defining the prior on peak-gene effect sizes
        self.mediation_pi = 0.9 # Prior probability that a peak mediates eQTL effects on the gene (only used if mediation_indicator is True)

        self.eqtl_mus = {} # Variational posterior mean eqtl effect size
        self.eqtl_vars = {} # Variational posterior variance eqtl effect size
        self.eqtl_alphas = {} # Variational posterior inclusion probability eqtl effect size
        self.caqtl_mus = {} # Variational posterior mean caqtl effect size
        self.caqtl_vars = {} # Variational posterior variance caqtl effect size
        self.caqtl_alphas = {} # Variational posterior inclusion probability caqtl effect size
        self.peak_gene_mus = {} # Variational posterior mean peak-gene effect size
        self.peak_gene_vars = {} # Variational posterior variance peak-gene effect size
        self.peak_gene_alphas = {} # Variational posterior inclusion probability peak-gene effect size
        self.mediation_probs = {} # Variational posterior probability that peak mediates eQTL effects on the gene (held at 1 if mediation_indicator is False)
        # Loop through genes to initialize gene-specific variables
        for gene_id in self.gene_ids:
            n_variants = len(np.loadtxt(self.gene_to_data[gene_id]['variant_ids_file'], dtype=str))
            n_peaks = self.gene_to_data[gene_id]['caqtl_effects'].shape[0]
            # Quick error check
            if n_variants != len(self.gene_to_data[gene_id]['eqtl_effects']) or n_variants != (self.gene_to_data[gene_id]['caqtl_effects']).shape[1]:
                print("Error: n_variants != len(eqtl_effects) for gene_id %s" % (gene_id))
                sys.exit(1)
            if (self.gene_to_data[gene_id]['caqtl_effects']).shape[0] != len(self.gene_to_data[gene_id]['peak_effects']):
                print("Error: n_peaks != len(peak_effects) for gene_id %s" % (gene_id))
                sys.exit(1)
            self.eqtl_mus[gene_id] = np.zeros(n_variants)
            self.eqtl_vars[gene_id] = np.ones(n_variants)
            self.eqtl_alphas[gene_id] = np.ones(n_variants) * 0.1
            self.caqtl_mus[gene_id] = np.zeros((n_peaks, n_variants))
            self.caqtl_vars[gene_id] = np.ones((n_peaks, n_variants))
            self.caqtl_alphas[gene_id] = np.ones((n_peaks, n_variants)) * 0.1
            self.peak_gene_mus[gene_id] = np.zeros(n_peaks)
            self.peak_gene_vars[gene_id] = np.ones(n_peaks)
            self.peak_gene_alphas[gene_id] = np.ones(n_peaks) * 0.1
            self.mediation_probs[gene_id] = np.ones(n_peaks)
        return




