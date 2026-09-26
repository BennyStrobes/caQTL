import numpy as np
import pdb
import sys
import time
from numba import njit



@njit(cache=True)
def two_slab_variant_sweep(z, se, ld_mat, path_prob, slab0_prior_var, slab1_prior_mean, slab1_prior_var, pi0, pi1, q0, q1, mu0, mu1, var0, var1, scaled_beta, ld_scaled_beta):
    """
    One sequential sweep of coordinate updates across variants (compiled with numba), for a spike and two-slab prior.
    RSS likelihood: effects ~ N(S R S^-1 beta, S R S) with S = diag(se), z = effects/se
    Prior: variant j has a chromatin path with probability path_prob_j
        no path: beta_j nonzero with probability pi0, and then beta_j ~ N(0, slab0_prior_var)
        path: beta_j nonzero with probability pi1, and then beta_j ~ N(slab1_prior_mean_j, slab1_prior_var_j)
    Posterior of each variant is over three outcomes: no effect, effect without path (probability q0, mean mu0, variance var0), effect with path (q1, mu1, var1)
    scaled_beta = E[beta]/se and ld_scaled_beta = R*scaled_beta at start of sweep
    Updates q0, q1, mu0, mu1, var0, var1 (and scaled_beta, ld_scaled_beta) in place
    ld_mat is assumed symmetric (row variant_iter is used in place of column variant_iter, as rows are contiguous in memory)
    """
    n_variants = len(z)
    very_negative = -1e300
    # Loop through variants (updates are sequential)
    for variant_iter in range(n_variants):
        ld_diag = ld_mat[variant_iter, variant_iter]
        # Residual z-score after removing effects of all other variants
        resid_z = z[variant_iter] - (ld_scaled_beta[variant_iter] - ld_diag*scaled_beta[variant_iter])
        lik_precision = ld_diag/(se[variant_iter]*se[variant_iter])
        lik_linear = resid_z/se[variant_iter]
        rho = path_prob[variant_iter]

        # No effect
        log_w_null = np.log((1.0 - rho)*(1.0 - pi0) + rho*(1.0 - pi1))

        # Effect without chromatin path: slab N(0, slab0_prior_var)
        new_var0 = 1.0/(lik_precision + 1.0/slab0_prior_var)
        new_mu0 = new_var0*lik_linear
        if (1.0 - rho)*pi0 > 0.0:
            log_w0 = np.log((1.0 - rho)*pi0) + 0.5*np.log(new_var0/slab0_prior_var) + (new_mu0*new_mu0)/(2.0*new_var0)
        else:
            log_w0 = very_negative

        # Effect with chromatin path: slab N(slab1_prior_mean, slab1_prior_var)
        if rho*pi1 > 0.0:
            new_var1 = 1.0/(lik_precision + 1.0/slab1_prior_var[variant_iter])
            new_mu1 = new_var1*(lik_linear + slab1_prior_mean[variant_iter]/slab1_prior_var[variant_iter])
            log_w1 = np.log(rho*pi1) + 0.5*np.log(new_var1/slab1_prior_var[variant_iter]) + (new_mu1*new_mu1)/(2.0*new_var1) - (slab1_prior_mean[variant_iter]*slab1_prior_mean[variant_iter])/(2.0*slab1_prior_var[variant_iter])
        else:
            new_var1 = new_var0
            new_mu1 = new_mu0
            log_w1 = very_negative

        # Normalize
        max_log_w = max(log_w_null, max(log_w0, log_w1))
        w_null = np.exp(log_w_null - max_log_w)
        w0 = np.exp(log_w0 - max_log_w)
        w1 = np.exp(log_w1 - max_log_w)
        new_q0 = w0/(w_null + w0 + w1)
        new_q1 = w1/(w_null + w0 + w1)

        q0[variant_iter] = new_q0
        q1[variant_iter] = new_q1
        mu0[variant_iter] = new_mu0
        mu1[variant_iter] = new_mu1
        var0[variant_iter] = new_var0
        var1[variant_iter] = new_var1

        # Update LD-weighted sum of effects to reflect new effect of this variant
        new_scaled_beta = (new_q0*new_mu0 + new_q1*new_mu1)/se[variant_iter]
        delta = new_scaled_beta - scaled_beta[variant_iter]
        if delta != 0.0:
            for other_variant_iter in range(n_variants):
                ld_scaled_beta[other_variant_iter] = ld_scaled_beta[other_variant_iter] + ld_mat[variant_iter, other_variant_iter]*delta
        scaled_beta[variant_iter] = new_scaled_beta
    return


def golden_section_maximize(func, lo, hi, n_steps=40):
    # Maximize a function of one variable on [lo, hi]
    golden = (np.sqrt(5.0) - 1.0)/2.0
    x1 = hi - golden*(hi - lo)
    x2 = lo + golden*(hi - lo)
    f1 = func(x1)
    f2 = func(x2)
    for step in range(n_steps):
        if f1 < f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + golden*(hi - lo)
            f2 = func(x2)
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - golden*(hi - lo)
            f1 = func(x1)
    return 0.5*(lo + hi)


class CHROMATIN_INFORMED_EXPRESSION_PREDICTION(object):
    """
    Stage 2 of the chromatin-informed expression prediction model (see chromatin_informed_expression_prediction_model.md). Fit once across all genes.
    Spike-and-slab model of eQTL effects from eQTL summary statistics, where the prior of each variant depends on whether the variant has a chromatin path to the gene
    (path_probs, pred_mean_given_path, pred_var_given_path come from stage 1: chromatin_only_expression_prediction.py. They are held fixed. No feedback from eQTL data to chromatin)
        no path: beta_j ~ pi0*N(0, sigma2_0) + (1 - pi0)*delta_0
        path:    beta_j ~ pi1*N(lambda*pred_mean_given_path_j, sigma2_1 + lambda^2*pred_var_given_path_j) + (1 - pi1)*delta_0
    pi0, pi1, sigma2_0, sigma2_1, lambda are shared across genes.
    Posterior mean eQTL effects are the expression prediction weights. Inclusion probabilities are NOT fine-mapping PIPs (not valid under tight LD).
    """

    def __init__(self, tol=1e-4, max_iter=300, min_path_prob=0.01):
        self.tol = tol
        self.max_iter = max_iter
        self.min_path_prob = min_path_prob  # Variants with probability of chromatin path less than this are treated as having no chromatin path

        # Filled in by fit()
        self.gene_to_data = None     # dict: gene_id -> dict with keys variant_ids_file, ld_file, eqtl_effects, eqtl_se, path_probs, pred_mean_given_path, pred_var_given_path
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

            # Now update hyperparameters across all genes
            self.update_pis()
            self.update_slab0_variance()
            self.update_lambda_and_slab1_variance()

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
        return np.asarray([self.pi0, self.pi1, self.sigma2_0, self.sigma2_1, self.lambda_scale])

    def print_hyperparameter_summary(self):
        """
        Print current values of hyperparameters (shared across genes).
        """
        print('##########################')
        print('Iteration ' + str(self.n_iter))
        print('lambda=' + str(self.lambda_scale))
        print('pi0=' + str(self.pi0) + '  pi1=' + str(self.pi1) + '  enrichment (pi1/pi0)=' + str(self.pi1/self.pi0))
        print('sigma2_0=' + str(self.sigma2_0) + '  sigma2_1=' + str(self.sigma2_1))
        sys.stdout.flush()
        return

    def get_posterior_mean_eqtl_effects(self, gene_id):
        """
        Posterior mean eQTL effects (expression prediction weights) for single gene.
        """
        return self.q0[gene_id]*self.mu0[gene_id] + self.q1[gene_id]*self.mu1[gene_id]

    def update_eqtl_effects(self, gene_id, ld_mat):
        """
        Update variational posterior parameters for eQTL effect sizes for single gene.
        """
        # eQTL likelihood info
        eqtl_effects = self.gene_to_data[gene_id]['eqtl_effects']
        eqtl_se = self.gene_to_data[gene_id]['eqtl_se']

        # Prior on eqtls with a chromatin path
        slab1_prior_mean = self.lambda_scale*self.pred_mean_given_path[gene_id]
        slab1_prior_var = self.sigma2_1 + np.square(self.lambda_scale)*self.pred_var_given_path[gene_id]

        # RSS likelihood: eqtl_effects ~ N(S R S^-1 beta, S R S) with S = diag(eqtl_se)
        # beta is on scale of standardized genotype and un-standardized expression (same scale as eqtl_effects)
        eqtl_z = eqtl_effects/eqtl_se
        scaled_beta = self.get_posterior_mean_eqtl_effects(gene_id)/eqtl_se  # E[beta]/se
        ld_scaled_beta = np.dot(ld_mat, scaled_beta)
        # Loop through variants (updates are sequential). Updates q0, q1, mu0, mu1, var0, var1 in place
        two_slab_variant_sweep(eqtl_z, eqtl_se, ld_mat, self.path_probs[gene_id], self.sigma2_0, slab1_prior_mean, slab1_prior_var, self.pi0, self.pi1, self.q0[gene_id], self.q1[gene_id], self.mu0[gene_id], self.mu1[gene_id], self.var0[gene_id], self.var1[gene_id], scaled_beta, ld_scaled_beta)
        return

    def update_pis(self):
        """
        Update prior inclusion probabilities of variants without (pi0) and with (pi1) a chromatin path (shared across genes).
        """
        q0_sum = 0.0
        q1_sum = 0.0
        no_path_sum = 0.0
        path_sum = 0.0
        # Loop through genes
        for gene_id in self.gene_ids:
            rho = self.path_probs[gene_id]
            q_null = 1.0 - self.q0[gene_id] - self.q1[gene_id]
            # Posterior probability of a chromatin path for variants with no effect
            path_given_null = rho*(1.0 - self.pi1)/((1.0 - rho)*(1.0 - self.pi0) + rho*(1.0 - self.pi1))
            q0_sum += np.sum(self.q0[gene_id])
            q1_sum += np.sum(self.q1[gene_id])
            path_sum += np.sum(self.q1[gene_id] + q_null*path_given_null)
            no_path_sum += np.sum(self.q0[gene_id] + q_null*(1.0 - path_given_null))

        if no_path_sum > 0:
            self.pi0 = np.clip(q0_sum/no_path_sum, 1e-12, 1.0 - 1e-12)
        if path_sum > 0:
            self.pi1 = np.clip(q1_sum/path_sum, 1e-12, 1.0 - 1e-12)
        return

    def update_slab0_variance(self):
        """
        Update slab variance of eQTL effects without a chromatin path (shared across genes).
        """
        sq_sum = 0.0
        q0_sum = 0.0
        # Loop through genes
        for gene_id in self.gene_ids:
            sq_sum += np.sum(self.q0[gene_id]*(np.square(self.mu0[gene_id]) + self.var0[gene_id]))
            q0_sum += np.sum(self.q0[gene_id])

        if q0_sum > 0:
            self.sigma2_0 = sq_sum/q0_sum
        return

    def update_lambda_and_slab1_variance(self):
        """
        Update scale (lambda) and slab variance (sigma2_1) of eQTL effects with a chromatin path (shared across genes).
        Maximizes expected log prior: sum_j q1_j*[-.5*log(sigma2_1 + lambda^2*w_j) - ((mu1_j - lambda*m_j)^2 + var1_j)/(2*(sigma2_1 + lambda^2*w_j))], with lambda ~ N(0, 10^2)
        where m_j, w_j are mean and variance of chromatin-predicted eQTL effect given a path
        """
        # Prior on lambda
        lambda_prior_var = 100.0

        # Variants with non-negligible probability of effect with chromatin path (across all genes)
        q1 = []
        mu1 = []
        var1 = []
        pred_mean = []
        pred_var = []
        for gene_id in self.gene_ids:
            indices = self.q1[gene_id] > 1e-10
            q1.append(self.q1[gene_id][indices])
            mu1.append(self.mu1[gene_id][indices])
            var1.append(self.var1[gene_id][indices])
            pred_mean.append(self.pred_mean_given_path[gene_id][indices])
            pred_var.append(self.pred_var_given_path[gene_id][indices])
        q1 = np.hstack(q1)
        mu1 = np.hstack(mu1)
        var1 = np.hstack(var1)
        pred_mean = np.hstack(pred_mean)
        pred_var = np.hstack(pred_var)
        if len(q1) == 0 or np.sum(q1*np.square(pred_mean)) == 0.0:
            return

        def objective(lambda_scale, sigma2_1):
            slab_var = sigma2_1 + np.square(lambda_scale)*pred_var
            return np.sum(q1*(-0.5*np.log(slab_var) - (np.square(mu1 - lambda_scale*pred_mean) + var1)/(2.0*slab_var))) - np.square(lambda_scale)/(2.0*lambda_prior_var)

        # Starting values: closed form maximizer when variance of chromatin-predicted eQTL effects is ignored
        sigma2_floor = 1e-8*self.sigma2_0
        start_lambda = np.sum(q1*mu1*pred_mean)/np.sum(q1*np.square(pred_mean))
        start_sigma2_1 = max(np.sum(q1*(np.square(mu1 - start_lambda*pred_mean) + var1))/np.sum(q1), sigma2_floor)
        candidates = [(self.lambda_scale, self.sigma2_1), (start_lambda, start_sigma2_1)]

        # Alternate one-dimensional maximizations
        new_lambda = start_lambda
        new_sigma2_1 = start_sigma2_1
        for round_iter in range(5):
            if new_lambda != 0.0:
                new_lambda = golden_section_maximize(lambda x: objective(x, new_sigma2_1), new_lambda - np.abs(new_lambda), new_lambda + np.abs(new_lambda))
            new_sigma2_1 = max(np.exp(golden_section_maximize(lambda x: objective(new_lambda, np.exp(x)), np.log(new_sigma2_1) - 3.0, np.log(new_sigma2_1) + 3.0)), sigma2_floor)
        candidates.append((new_lambda, new_sigma2_1))

        # Keep best
        objective_values = [objective(candidate[0], candidate[1]) for candidate in candidates]
        self.lambda_scale, self.sigma2_1 = candidates[np.argmax(objective_values)]
        return

    def initialize_model(self):
        """
        Initialize model parameters and hyperparameters.
        """
        self.converged = False
        self.n_iter = 0

        self.n_genes = len(self.gene_ids)
        # Prior variables
        self.pi0 = 0.1 # Prior inclusion probability of eqtl effect for variants without a chromatin path
        self.pi1 = 0.1 # Prior inclusion probability of eqtl effect for variants with a chromatin path
        self.sigma2_0 = 0.1 # Slab variance of eqtl effects without a chromatin path
        self.sigma2_1 = 0.1 # Slab variance of eqtl effects with a chromatin path (around lambda*chromatin-predicted effect)
        self.lambda_scale = 0.0 # Scale factor converting chromatin-predicted eQTL effects to eQTL effects

        self.q0 = {} # Variational posterior probability of eqtl effect without a chromatin path
        self.q1 = {} # Variational posterior probability of eqtl effect with a chromatin path
        self.mu0 = {} # Variational posterior mean eqtl effect size given effect without a chromatin path
        self.mu1 = {} # Variational posterior mean eqtl effect size given effect with a chromatin path
        self.var0 = {} # Variational posterior variance given effect without a chromatin path
        self.var1 = {} # Variational posterior variance given effect with a chromatin path
        self.path_probs = {} # Probability variant has a chromatin path (fixed; from stage 1)
        self.pred_mean_given_path = {} # Mean of chromatin-predicted eQTL effect given a path (fixed; from stage 1)
        self.pred_var_given_path = {} # Variance of chromatin-predicted eQTL effect given a path (fixed; from stage 1)

        # Loop through genes to initialize gene-specific variables
        n_variants_with_path = 0
        n_variants_total = 0
        for gene_id in self.gene_ids:
            n_variants = len(np.loadtxt(self.gene_to_data[gene_id]['variant_ids_file'], dtype=str, ndmin=1))
            # Quick error check
            if n_variants != len(self.gene_to_data[gene_id]['eqtl_effects']) or n_variants != len(self.gene_to_data[gene_id]['path_probs']):
                print("Error: n_variants != len(eqtl_effects) or len(path_probs) for gene_id %s" % (gene_id))
                sys.exit(1)

            # Variants with small probability of chromatin path are treated as having no chromatin path
            path_probs = np.copy(self.gene_to_data[gene_id]['path_probs']).astype(float)
            pred_mean_given_path = np.copy(self.gene_to_data[gene_id]['pred_mean_given_path']).astype(float)
            pred_var_given_path = np.copy(self.gene_to_data[gene_id]['pred_var_given_path']).astype(float)
            no_path = path_probs < self.min_path_prob
            path_probs[no_path] = 0.0
            pred_mean_given_path[no_path] = 0.0
            pred_var_given_path[no_path] = 0.0
            self.path_probs[gene_id] = np.minimum(path_probs, 1.0)
            self.pred_mean_given_path[gene_id] = pred_mean_given_path
            self.pred_var_given_path[gene_id] = pred_var_given_path
            n_variants_with_path += np.sum(no_path == False)
            n_variants_total += n_variants

            self.q0[gene_id] = np.ones(n_variants) * 0.1
            self.q1[gene_id] = np.zeros(n_variants)
            self.mu0[gene_id] = np.zeros(n_variants)
            self.mu1[gene_id] = np.zeros(n_variants)
            self.var0[gene_id] = np.ones(n_variants)
            self.var1[gene_id] = np.ones(n_variants)
        print(str(n_variants_with_path) + ' of ' + str(n_variants_total) + ' gene-variant pairs have probability of chromatin path >= ' + str(self.min_path_prob))
        sys.stdout.flush()
        return
