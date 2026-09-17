import numpy as np
from susie_rss import SUSIE_RSS


class SUSIE_RSS_MIXTURE(SUSIE_RSS):
    """
    SuSiE-RSS (z-score mode) in which the prior on the effect of element j is a Gaussian mixture,
        b_j ~ sum_c w_jc N(m_jc, s2_jc),
    instead of N(0, V). Used by Model L (caqtl_mediated_fine_mapping_model_proposal_predicted_mean.md), where the components
    are the possible chromatin predictions of a variant's causal eQTL effect. The prior variance is not optimized per effect:
    an effect is either on (this prior) or off (zero effect on every element), decided by SuSiE's own rule (on when the
    single-effect log likelihood ratio against no effect is positive) if switch_off_effects is True, and always on otherwise.
    V is 1 for an effect that is on and 0 for one that is off, so pip / credible-set code of the base class applies unchanged.
    The single-effect posterior under the mixture prior is exact (mixture of Gaussians per element), so the base class's KL
    identity and ELBO hold. After the fit, component_posteriors() stores, per effect and element, the posterior component
    weights (omega), and the component posterior means and variances, recomputed from the final residual of each effect.
    """

    def __init__(self, L=10, switch_off_effects=True, tol=1e-3, max_iter=100, coverage=0.95, min_abs_corr=0.5):
        super().__init__(L=L, prior_variance=1.0, estimate_prior_variance=switch_off_effects, estimate_residual_variance=False, zscore_residual_variance=False, tol=tol, max_iter=max_iter, coverage=coverage, min_abs_corr=min_abs_corr)
        self.omega = None        # L x p x C posterior component weights given the element is selected (0 for an effect that is off)
        self.comp_mean = None    # L x p x C posterior mean of the effect under each component
        self.comp_var = None     # L x p x C posterior variance of the effect under each component

    def fit(self, beta_hat, beta_se, ld_mat, n, mix_w, mix_m, mix_s2, prior_weights=None):
        # mix_w, mix_m, mix_s2: p x C arrays (weights sum to 1 per element; unused components get weight 0 and any positive variance)
        if n is not None:
            raise ValueError('SUSIE_RSS_MIXTURE runs in z-score mode only (n must be None)')
        mix_w = np.asarray(mix_w, dtype=float)
        mix_m = np.asarray(mix_m, dtype=float)
        mix_s2 = np.asarray(mix_s2, dtype=float)
        p = len(beta_hat)
        if mix_w.ndim != 2 or mix_w.shape[0] != p or mix_m.shape != mix_w.shape or mix_s2.shape != mix_w.shape:
            raise ValueError('mix_w, mix_m and mix_s2 must be p x C arrays')
        if np.any(mix_w < 0.0) or np.any(np.abs(np.sum(mix_w, axis=1) - 1.0) > 1e-6):
            raise ValueError('mixture weights must be non-negative and sum to 1 for every element')
        if np.any(mix_s2 <= 0.0) or not np.all(np.isfinite(mix_m)):
            raise ValueError('mixture variances must be positive and means finite')
        self.mix_w = mix_w
        self.mix_m = mix_m
        self.mix_s2 = mix_s2
        with np.errstate(divide='ignore'):
            self.log_mix_w = np.where(mix_w > 0.0, np.log(np.maximum(mix_w, 1e-300)), -np.inf)
        super().fit(beta_hat, beta_se, ld_mat, None, prior_weights=prior_weights)
        self.component_posteriors()
        return self

    def mixture_single_effect(self, Xtr):
        # Per element: log BF_j = log sum_c w_jc N(betahat_j; m_jc, s2_jc + shat2_j) / N(betahat_j; 0, shat2_j), the posterior component
        # weights omega_jc, and the component posterior mean / variance of the effect
        betahat = Xtr / self.dXtX
        shat2 = self.sigma2 / self.dXtX
        tot = self.mix_s2 + shat2[:, None]
        lc = 0.5 * np.log(shat2[:, None] / tot) - (betahat[:, None] - self.mix_m)**2 / (2.0 * tot) + (betahat**2 / (2.0 * shat2))[:, None] + self.log_mix_w
        mx = np.max(lc, axis=1)
        log_bf = mx + np.log(np.sum(np.exp(lc - mx[:, None]), axis=1))
        omega = np.exp(lc - log_bf[:, None])
        post_var = 1.0 / (1.0 / self.mix_s2 + (self.dXtX / self.sigma2)[:, None])
        post_mean = post_var * ((Xtr / self.sigma2)[:, None] + self.mix_m / self.mix_s2)
        return log_bf, omega, post_mean, post_var

    def single_effect_regression(self, Xtr, V):
        log_bf, omega, post_mean, post_var = self.mixture_single_effect(Xtr)
        log_post = log_bf + self.log_prior_weights
        max_log_post = np.max(log_post)
        weights = np.exp(log_post - max_log_post)
        alpha = weights / np.sum(weights)
        lbf_model = max_log_post + np.log(np.sum(weights))     # log(sum_j pi_j BF_j): log BF of the single effect vs no effect
        if self.estimate_prior_variance and lbf_model <= 0.0:
            # switched off: zero effect everywhere (same state as V = 0 in the base class)
            return self.prior_weights.copy(), np.zeros(self.p), np.zeros(self.p), 0.0, 0.0
        mu = np.sum(omega * post_mean, axis=1)
        mu2 = np.sum(omega * (post_var + post_mean**2), axis=1)
        return alpha, mu, mu2, lbf_model, 1.0

    def component_posteriors(self):
        L = self.L
        p = self.p
        C = self.mix_w.shape[1]
        self.omega = np.zeros((L, p, C))
        self.comp_mean = np.zeros((L, p, C))
        self.comp_var = np.zeros((L, p, C))
        for l in range(L):
            if self.V[l] <= 0.0:
                continue
            Xtr = self.Xty - self.XtXr + np.dot(self.XtX, self.alpha[l, :] * self.mu[l, :])   # residual excluding effect l
            log_bf, omega, post_mean, post_var = self.mixture_single_effect(Xtr)
            self.omega[l] = omega
            self.comp_mean[l] = post_mean
            self.comp_var[l] = post_var
