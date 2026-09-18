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


class EQTL_ONLY_EXPRESSION_PREDICTION(object):
    """
    Spike-and-slab model of eQTL effects from summary statistics (eQTL data only), used for genetic prediction of expression. Fit once across all genes.
    beta_j ~ eqtl_pi*N(0, sigma2_eqtl) + (1 - eqtl_pi)*delta_0, with eqtl_pi and sigma2_eqtl shared across genes.
    Posterior mean eQTL effects (eqtl_alphas*eqtl_mus) are the expression prediction weights. Inclusion probabilities are NOT fine-mapping PIPs (not valid under tight LD).
    """

    def __init__(self, tol=1e-4, max_iter=300):
        self.tol = tol
        self.max_iter = max_iter

        # Filled in by fit()
        self.gene_to_data = None     # dict: gene_id -> dict with keys variant_ids_file, ld_file, eqtl_effects, eqtl_se
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

            # Now update hyperparameters (pi, sigma2) across all genes
            self.update_pis()
            self.update_sigma_sqs()

            # Print summary of hyperparameters at this iteration
            self.print_hyperparameter_summary()

            # Check for convergence: relative change in every hyperparameter is less than tol
            hyperparameters = self.get_hyperparameter_vector()
            relative_changes = np.abs(hyperparameters - prev_hyperparameters)/np.maximum(np.abs(prev_hyperparameters), 1e-300)
            print('max relative change in hyperparameters=' + str(np.max(relative_changes)))
            sys.stdout.flush()
            if np.max(relative_changes) < self.tol:
                self.converged = True
                print('Converged after ' + str(self.n_iter + 1) + ' iterations')
                break

        return self

    def get_hyperparameter_vector(self):
        """
        Vector of current values of hyperparameters (shared across genes). Used to assess convergence.
        """
        return np.asarray([self.eqtl_pi, self.sigma2_eqtl])

    def print_hyperparameter_summary(self):
        """
        Print current values of hyperparameters (shared across genes).
        """
        print('##########################')
        print('Iteration ' + str(self.n_iter))
        print('eqtl_pi=' + str(self.eqtl_pi))
        print('sigma2_eqtl=' + str(self.sigma2_eqtl))
        sys.stdout.flush()
        return

    def update_pis(self):
        """
        Update prior inclusion probability (shared across genes). It is the average posterior inclusion probability.
        """
        eqtl_alpha_sum = 0.0
        eqtl_count = 0.0
        # Loop through genes
        for gene_id in self.gene_ids:
            eqtl_alpha_sum += np.sum(self.eqtl_alphas[gene_id])
            eqtl_count += self.eqtl_alphas[gene_id].size

        if eqtl_count > 0:
            self.eqtl_pi = eqtl_alpha_sum/eqtl_count
        return

    def update_sigma_sqs(self):
        """
        Update slab variance (shared across genes). It is the average expected squared effect, across included effects.
        """
        eqtl_sq_sum = 0.0
        eqtl_alpha_sum = 0.0
        # Loop through genes
        for gene_id in self.gene_ids:
            # eQTL: E[beta_j^2]
            eqtl_sq_sum += np.sum(self.eqtl_alphas[gene_id]*(np.square(self.eqtl_mus[gene_id]) + self.eqtl_vars[gene_id]))
            eqtl_alpha_sum += np.sum(self.eqtl_alphas[gene_id])

        if eqtl_alpha_sum > 0:
            self.sigma2_eqtl = eqtl_sq_sum/eqtl_alpha_sum
        return

    def update_eqtl_effects(self, gene_id, ld_mat):
        """
        Update variational posterior parameters for eQTL effect sizes for single gene.
        """
        # eQTL likelihood info
        eqtl_effects = self.gene_to_data[gene_id]['eqtl_effects']
        eqtl_se = self.gene_to_data[gene_id]['eqtl_se']
        n_variants_per_gene = len(eqtl_effects)

        # Prior on eqtls
        eqtl_prior_mean = np.zeros(n_variants_per_gene)
        eqtl_prior_mean_sq = np.zeros(n_variants_per_gene)
        eqtl_prior_var = np.ones(n_variants_per_gene) * self.sigma2_eqtl

        # Update variational posterior parameters for eQTL effect sizes
        # RSS likelihood: eqtl_effects ~ N(S R S^-1 beta, S R S) with S = diag(eqtl_se)
        # beta is on scale of standardized genotype and un-standardized expression (same scale as eqtl_effects)
        eqtl_z = eqtl_effects/eqtl_se
        scaled_beta = self.eqtl_alphas[gene_id]*self.eqtl_mus[gene_id]/eqtl_se  # E[beta]/se
        ld_scaled_beta = np.dot(ld_mat, scaled_beta)
        log_prior_odds = np.log(self.eqtl_pi) - np.log(1.0 - self.eqtl_pi)
        # No additional gaussian information on eQTL effects
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
        self.sigma2_eqtl = 0.1 # Variance parameter defining the prior on eqtl effect sizes

        self.eqtl_mus = {} # Variational posterior mean eqtl effect size
        self.eqtl_vars = {} # Variational posterior variance eqtl effect size
        self.eqtl_alphas = {} # Variational posterior inclusion probability eqtl effect size

        # Loop through genes to initialize gene-specific variables
        for gene_id in self.gene_ids:
            n_variants = len(np.loadtxt(self.gene_to_data[gene_id]['variant_ids_file'], dtype=str, ndmin=1))
            # Quick error check
            if n_variants != len(self.gene_to_data[gene_id]['eqtl_effects']):
                print("Error: n_variants != len(eqtl_effects) for gene_id %s" % (gene_id))
                sys.exit(1)

            self.eqtl_mus[gene_id] = np.zeros(n_variants)
            self.eqtl_vars[gene_id] = np.ones(n_variants)
            self.eqtl_alphas[gene_id] = np.ones(n_variants) * 0.1

        return
