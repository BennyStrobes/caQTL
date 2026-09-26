import numpy as np
import pdb
import sys
from susie_rss import SUSIE_RSS


class CHROMATIN_ONLY_EXPRESSION_PREDICTION(object):
    """
    Stage 1 of the chromatin-informed expression prediction model (see chromatin_informed_expression_prediction_model.md). Uses caQTL and peak-gene data only (no eQTL data).
    caQTL effects of each peak are fit with SuSiE (z-score model; no sample size). Peak-gene effects are fit with a spike and slab (peak_pi and sigma2_peak_gene shared across genes).
    Chromatin-predicted eQTL effect of variant j on the gene is m_j = sum_k g_k*b_kj (g_k peak-gene effect, b_kj caQTL effect).
    pred_eqtl (E[m_j]) are expression prediction weights (on scale of standardized genotype, up to an unknown scale factor lambda shared across genes).
    Also computes, for each variant, the probability it has a chromatin path to the gene (path_probs), and the mean and variance of m_j given a path.
    """

    def __init__(self, L=5, tol=1e-4, max_iter=300):
        self.L = L                   # Number of SuSiE single effects per peak
        self.tol = tol
        self.max_iter = max_iter

        # Filled in by fit()
        self.gene_to_data = None     # dict: gene_id -> dict with keys variant_ids_file, ld_file, caqtl_effects, caqtl_se, peak_effects, peak_se
        self.gene_ids = None         # sorted array of gene ids

    def fit(self, gene_to_data):
        self.gene_to_data = gene_to_data
        self.gene_ids = np.sort(np.asarray(list(gene_to_data.keys())))

        # Fit peak-gene effects (hyperparameters shared across genes)
        self.fit_peak_gene_effects()

        # Fit caQTL effects and compute chromatin-predicted eQTL effects
        self.pred_eqtl = {} # E[m_j]: chromatin-predicted eQTL effect
        self.path_probs = {} # Probability variant has a chromatin path to the gene (caQTL variant for a peak with a peak-gene effect)
        self.pred_mean_given_path = {} # E[m_j | path]
        self.pred_var_given_path = {} # Var[m_j | path]
        # Loop through genes
        for gene_iter, gene_id in enumerate(self.gene_ids):
            # Extract LD for gene
            ld_mat = np.load(self.gene_to_data[gene_id]['ld_file'])
            self.compute_predicted_eqtl_effects(gene_id, ld_mat)
            if np.mod(gene_iter, 100) == 0:
                print(str(gene_iter) + ' of ' + str(len(self.gene_ids)) + ' genes')
                sys.stdout.flush()
        return self

    def fit_peak_gene_effects(self):
        """
        Spike and slab on peak-gene effects: peak_effects ~ N(g, peak_se^2), g ~ peak_pi*N(0, sigma2_peak_gene) + (1 - peak_pi)*delta_0
        """
        # All peak-gene pairs across genes
        peak_effects = np.hstack([self.gene_to_data[gene_id]['peak_effects'] for gene_id in self.gene_ids])
        peak_se = np.hstack([self.gene_to_data[gene_id]['peak_se'] for gene_id in self.gene_ids])

        self.peak_pi = 0.1
        self.sigma2_peak_gene = 0.1
        peak_gene_alphas = np.zeros(len(peak_effects))
        peak_gene_mus = np.zeros(len(peak_effects))
        peak_gene_vars = np.zeros(len(peak_effects))
        if len(peak_effects) > 0:
            for itera in range(self.max_iter):
                prev_hyperparameters = np.asarray([self.peak_pi, self.sigma2_peak_gene])
                # Update variational posteriors
                peak_gene_vars = 1.0/(1.0/np.square(peak_se) + 1.0/self.sigma2_peak_gene)
                peak_gene_mus = peak_gene_vars*peak_effects/np.square(peak_se)
                log_odds = np.log(self.peak_pi) - np.log(1.0 - self.peak_pi) + 0.5*np.log(peak_gene_vars/self.sigma2_peak_gene) + np.square(peak_gene_mus)/(2.0*peak_gene_vars)
                peak_gene_alphas = 1.0/(1.0 + np.exp(-np.clip(log_odds, -500, 500)))
                # Update hyperparameters
                self.peak_pi = np.mean(peak_gene_alphas)
                if np.sum(peak_gene_alphas) > 0:
                    self.sigma2_peak_gene = np.sum(peak_gene_alphas*(np.square(peak_gene_mus) + peak_gene_vars))/np.sum(peak_gene_alphas)
                hyperparameters = np.asarray([self.peak_pi, self.sigma2_peak_gene])
                if np.max(np.abs(hyperparameters - prev_hyperparameters)/np.maximum(np.abs(prev_hyperparameters), 1e-300)) < self.tol:
                    break
        print('peak_pi=' + str(self.peak_pi) + '  sigma2_peak_gene=' + str(self.sigma2_peak_gene))
        sys.stdout.flush()

        # Split back up into genes
        self.peak_gene_alphas = {}
        self.peak_gene_mus = {}
        self.peak_gene_vars = {}
        start = 0
        for gene_id in self.gene_ids:
            n_peaks = len(self.gene_to_data[gene_id]['peak_effects'])
            self.peak_gene_alphas[gene_id] = peak_gene_alphas[start:(start + n_peaks)]
            self.peak_gene_mus[gene_id] = peak_gene_mus[start:(start + n_peaks)]
            self.peak_gene_vars[gene_id] = peak_gene_vars[start:(start + n_peaks)]
            start = start + n_peaks
        return

    def compute_predicted_eqtl_effects(self, gene_id, ld_mat):
        """
        Fit caQTL effects of each peak with SuSiE and compute moments of chromatin-predicted eQTL effect m_j = sum_k g_k*b_kj for single gene.
        """
        caqtl_effects = self.gene_to_data[gene_id]['caqtl_effects']
        caqtl_se = self.gene_to_data[gene_id]['caqtl_se']
        n_peaks_per_gene = caqtl_effects.shape[0]
        n_variants_per_gene = caqtl_effects.shape[1]

        peak_gene_effects = self.peak_gene_alphas[gene_id]*self.peak_gene_mus[gene_id]  # E[g]
        peak_gene_effects_sq = self.peak_gene_alphas[gene_id]*(np.square(self.peak_gene_mus[gene_id]) + self.peak_gene_vars[gene_id])  # E[g^2]

        pred_eqtl = np.zeros(n_variants_per_gene)  # E[m]
        pred_eqtl_var = np.zeros(n_variants_per_gene)  # Var[m]
        log_no_path_prob = np.zeros(n_variants_per_gene)  # log probability variant has no chromatin path through any peak
        # Loop through peaks
        for peak_iter in range(n_peaks_per_gene):
            peak_caqtl_se = caqtl_se[peak_iter, :]
            # SuSiE z-score model (sample size of None): effects are on z-score scale (caQTL effect divided by standard error)
            susie_fit = SUSIE_RSS(L=self.L).fit(caqtl_effects[peak_iter, :], peak_caqtl_se, ld_mat, None)
            variant_peak_effects = susie_fit.posterior_mean*peak_caqtl_se  # E[b]
            variant_peak_effects_sq = (np.square(susie_fit.posterior_sd) + np.square(susie_fit.posterior_mean))*np.square(peak_caqtl_se)  # E[b^2]

            pred_eqtl = pred_eqtl + peak_gene_effects[peak_iter]*variant_peak_effects
            pred_eqtl_var = pred_eqtl_var + peak_gene_effects_sq[peak_iter]*variant_peak_effects_sq - np.square(peak_gene_effects[peak_iter]*variant_peak_effects)
            # Path through this peak: variant is a caQTL for the peak and the peak has a peak-gene effect
            log_no_path_prob = log_no_path_prob + np.log(np.maximum(1.0 - self.peak_gene_alphas[gene_id][peak_iter]*susie_fit.pip, 1e-300))

        path_probs = 1.0 - np.exp(log_no_path_prob)
        pred_eqtl_sq = np.square(pred_eqtl) + pred_eqtl_var  # E[m^2]

        # m_j is zero when there is no path. So E[m | path] = E[m]/P(path) and E[m^2 | path] = E[m^2]/P(path)
        has_path = path_probs > 0.0
        pred_mean_given_path = np.zeros(n_variants_per_gene)
        pred_var_given_path = np.zeros(n_variants_per_gene)
        pred_mean_given_path[has_path] = pred_eqtl[has_path]/path_probs[has_path]
        pred_var_given_path[has_path] = np.maximum(pred_eqtl_sq[has_path]/path_probs[has_path] - np.square(pred_mean_given_path[has_path]), 0.0)

        self.pred_eqtl[gene_id] = pred_eqtl
        self.path_probs[gene_id] = path_probs
        self.pred_mean_given_path[gene_id] = pred_mean_given_path
        self.pred_var_given_path[gene_id] = pred_var_given_path
        return
