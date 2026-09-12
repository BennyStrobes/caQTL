import numpy as np


class SUSIE_RSS(object):
    """
    Sum of Single Effects (SuSiE) fine-mapping from summary statistics.

    fit() takes marginal effect sizes and standard errors from univariate OLS on standardized genotypes
    (each SNP mean 0, variance 1), the in-sample LD (correlation) matrix, and the sample size n. These are
    converted exactly to the sufficient statistics X'X, X'y, y'y of the individual-level regression, so the
    fit is identical to running SuSiE (Wang et al. 2020, JRSSB) on the individual-level data with the IBSS
    algorithm. If n is None, the z-score model of Zou et al. (2022, PLoS Genet) is used instead
    (X'X = R, X'y = z, residual variance fixed at 1).

    Defaults mirror susieR: L = 10 single effects, prior variance of each effect initialized at 0.2 * var(y)
    and then estimated by maximizing the single-effect likelihood, residual variance estimated, and
    convergence declared when the ELBO increases by less than 1e-3.
    """

    def __init__(self, L=10, prior_variance=0.2, estimate_prior_variance=True, estimate_residual_variance=True, tol=1e-3, max_iter=100, coverage=0.95, min_abs_corr=0.5):
        self.L = L
        self.prior_variance = prior_variance                    # initial prior variance of each effect, as a fraction of var(y)
        self.estimate_prior_variance = estimate_prior_variance
        self.estimate_residual_variance = estimate_residual_variance
        self.tol = tol
        self.max_iter = max_iter
        self.coverage = coverage                                # credible set coverage
        self.min_abs_corr = min_abs_corr                        # minimum purity (min |r| between members) of a reported credible set

        # Filled in by fit()
        self.alpha = None            # L x p: posterior probability that single effect l is SNP j
        self.mu = None               # L x p: posterior mean of effect l given it is SNP j
        self.mu2 = None              # L x p: posterior second moment of effect l given it is SNP j
        self.V = None                # L: prior variance of each single effect (0 = effect switched off)
        self.sigma2 = None           # residual variance
        self.lbf = None              # L: log Bayes factor of each single effect vs no effect
        self.pip = None              # p: posterior inclusion probability of each SNP
        self.posterior_mean = None   # p: posterior mean effect size of each SNP
        self.posterior_sd = None     # p: posterior sd of the effect size of each SNP
        self.credible_sets = None    # list of dicts with keys effect_index, snp_indices, coverage, purity
        self.elbo = None             # ELBO after each IBSS iteration
        self.converged = None
        self.n_iter = None

    def fit(self, beta_hat, beta_se, ld_mat, n, prior_weights=None):
        beta_hat = np.asarray(beta_hat, dtype=float)
        beta_se = np.asarray(beta_se, dtype=float)
        ld_mat = np.asarray(ld_mat, dtype=float)
        p = len(beta_hat)
        if beta_hat.ndim != 1 or beta_se.shape != beta_hat.shape:
            raise ValueError('beta_hat and beta_se must be 1D vectors of the same length')
        if ld_mat.shape != (p, p):
            raise ValueError('ld_mat must be a p x p matrix')
        if np.any(np.isnan(beta_hat)) or np.any(beta_se <= 0.0) or np.any(np.isnan(beta_se)):
            raise ValueError('beta_hat must be non-missing and beta_se positive')
        if n is not None and n <= 2:
            raise ValueError('n must be greater than 2')
        self.p = p
        self.ld_mat = ld_mat

        # Convert summary statistics to sufficient statistics
        if n is None:
            # z-score model z_hat ~ N(R z, R): equivalent to X'X = R, X'y = z with residual variance fixed at 1.
            # n and y'y are placeholders that only enter the ELBO as constants (they give var_y = 1).
            self.XtX = ld_mat
            self.Xty = beta_hat / beta_se
            self.yty = 1.0
            self.n = 2
            estimate_residual_variance = False
        else:
            # OLS on standardized genotypes: X'X = (n-1) R and X'y = (n-1) beta_hat. y'y follows from the OLS
            # standard error identity se_j^2 = (y'y - beta_hat_j^2 (n-1)) / ((n-2)(n-1)), which holds for every j.
            self.XtX = (n - 1.0) * ld_mat
            self.Xty = (n - 1.0) * beta_hat
            yty_per_snp = (n - 1.0) * (beta_se**2 * (n - 2.0) + beta_hat**2)
            self.yty = np.median(yty_per_snp)
            self.n = n
            estimate_residual_variance = self.estimate_residual_variance
            if np.max(np.abs(yty_per_snp / self.yty - 1.0)) > 0.05:
                print('warning: the y variance implied by (beta_hat, beta_se) differs across SNPs by more than 5%. Are these OLS summary statistics on standardized genotypes?')
        self.dXtX = np.diag(self.XtX).copy()
        self.var_y = self.yty / (self.n - 1.0)

        # Prior probability that a single effect is on each SNP
        if prior_weights is None:
            self.prior_weights = np.ones(p) / p
        else:
            self.prior_weights = np.asarray(prior_weights, dtype=float) / np.sum(prior_weights)
        self.log_prior_weights = np.log(self.prior_weights)

        # Initialize variational parameters
        L = self.L
        self.alpha = np.ones((L, p)) / p
        self.mu = np.zeros((L, p))
        self.mu2 = np.zeros((L, p))
        self.V = np.ones(L) * self.prior_variance * self.var_y
        self.sigma2 = self.var_y
        self.lbf = np.zeros(L)
        self.KL = np.zeros(L)
        self.XtXr = np.zeros(p)      # X'X times the current posterior mean of b (sum of all L effects)
        elbo = [-np.inf]
        self.converged = False

        # IBSS: iteratively fit each single effect to the residual left by the other L-1 effects
        for iteration in range(self.max_iter):
            for l in range(L):
                # X'r for the residual excluding effect l
                self.XtXr = self.XtXr - np.dot(self.XtX, self.alpha[l, :] * self.mu[l, :])
                Xtr = self.Xty - self.XtXr
                # Single effect regression of the residual on each SNP
                alpha_l, mu_l, mu2_l, lbf_l, V_l = self.single_effect_regression(Xtr, self.V[l])
                self.alpha[l, :] = alpha_l
                self.mu[l, :] = mu_l
                self.mu2[l, :] = mu2_l
                self.V[l] = V_l
                self.lbf[l] = lbf_l
                # KL divergence from the prior to the posterior of effect l: -lbf + E_q[log p(r|b) - log p(r|0)]
                self.KL[l] = -lbf_l - 0.5 / self.sigma2 * (-2.0 * np.sum(alpha_l * mu_l * Xtr) + np.sum(self.dXtX * alpha_l * mu2_l))
                # Add the updated effect l back
                self.XtXr = self.XtXr + np.dot(self.XtX, alpha_l * mu_l)

            # Convergence check on the ELBO
            elbo.append(self.get_elbo())
            if elbo[-1] - elbo[-2] < self.tol:
                self.converged = True
                break

            # Residual variance update (maximizes the ELBO)
            if estimate_residual_variance:
                self.sigma2 = self.get_expected_rss() / self.n
                if self.sigma2 <= 0.0:
                    raise ValueError('estimated residual variance is negative: summary statistics and LD are inconsistent')

        self.n_iter = iteration + 1
        self.elbo = np.array(elbo[1:])
        if not self.converged:
            print('warning: IBSS did not converge in ' + str(self.max_iter) + ' iterations')

        # Summaries
        self.pip = self.get_pips()
        self.posterior_mean = np.sum(self.alpha * self.mu, axis=0)
        self.posterior_sd = np.sqrt(np.maximum(np.sum(self.alpha * self.mu2 - (self.alpha * self.mu)**2, axis=0), 0.0))
        self.credible_sets = self.get_credible_sets()
        return self

    def single_effect_regression(self, Xtr, V):
        # Bayesian regression of the residual r on a single SNP: b_j ~ N(0, V) for the one SNP j drawn with P(j) = prior_weights_j
        betahat = Xtr / self.dXtX               # OLS effect of each SNP on the residual
        shat2 = self.sigma2 / self.dXtX         # and its sampling variance
        if self.estimate_prior_variance:
            V = self.optimize_prior_variance(betahat, shat2, V)
        log_bf = self.log_bayes_factors(V, betahat, shat2)
        log_post = log_bf + self.log_prior_weights
        max_log_post = np.max(log_post)
        weights = np.exp(log_post - max_log_post)
        alpha = weights / np.sum(weights)
        lbf_model = max_log_post + np.log(np.sum(weights))     # log(sum_j pi_j BF_j): log BF of the single effect vs no effect
        if V > 0.0:
            post_var = 1.0 / (1.0 / V + self.dXtX / self.sigma2)
        else:
            post_var = np.zeros(self.p)
        mu = post_var * Xtr / self.sigma2
        mu2 = post_var + mu**2
        return alpha, mu, mu2, lbf_model, V

    def log_bayes_factors(self, V, betahat, shat2):
        # log BF_j = log N(betahat_j; 0, V + shat2_j) - log N(betahat_j; 0, shat2_j)
        return 0.5 * np.log(shat2 / (V + shat2)) + 0.5 * (betahat**2 / shat2) * (V / (V + shat2))

    def ser_loglik(self, V, betahat, shat2):
        # log likelihood ratio of the single effect model with prior variance V vs no effect: log(sum_j pi_j BF_j(V))
        log_post = self.log_bayes_factors(V, betahat, shat2) + self.log_prior_weights
        max_log_post = np.max(log_post)
        return max_log_post + np.log(np.sum(np.exp(log_post - max_log_post)))

    def optimize_prior_variance(self, betahat, shat2, V_current):
        # Maximize the single effect log likelihood over log(V) on [-30, 15] by golden-section search
        # (susieR does the same with Brent's method). Keep the current V if the search does not beat it,
        # and set V to exactly 0 if no effect is at least as likely.
        lo = -30.0
        hi = 15.0
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = hi - golden * (hi - lo)
        x2 = lo + golden * (hi - lo)
        f1 = self.ser_loglik(np.exp(x1), betahat, shat2)
        f2 = self.ser_loglik(np.exp(x2), betahat, shat2)
        for _ in range(50):
            if f1 < f2:
                lo = x1
                x1 = x2
                f1 = f2
                x2 = lo + golden * (hi - lo)
                f2 = self.ser_loglik(np.exp(x2), betahat, shat2)
            else:
                hi = x2
                x2 = x1
                f2 = f1
                x1 = hi - golden * (hi - lo)
                f1 = self.ser_loglik(np.exp(x1), betahat, shat2)
        V_new = np.exp(0.5 * (lo + hi))
        if self.ser_loglik(V_new, betahat, shat2) < self.ser_loglik(V_current, betahat, shat2):
            V_new = V_current
        if self.ser_loglik(0.0, betahat, shat2) >= self.ser_loglik(V_new, betahat, shat2):
            V_new = 0.0
        return V_new

    def get_expected_rss(self):
        # E_q ||y - X b||^2 under the variational posterior
        b_bar_l = self.alpha * self.mu                 # L x p posterior mean of each single effect
        b_bar = np.sum(b_bar_l, axis=0)                # p posterior mean of b
        erss = self.yty - 2.0 * np.dot(b_bar, self.Xty) + np.dot(b_bar, np.dot(self.XtX, b_bar))
        erss = erss - np.sum(np.dot(b_bar_l, self.XtX) * b_bar_l)    # replace sum_l b_l' X'X b_l ...
        erss = erss + np.sum(self.dXtX * self.alpha * self.mu2)       # ... with its expectation sum_l sum_j alpha_lj mu2_lj X'X_jj
        return erss

    def get_elbo(self):
        # Evidence lower bound: expected log likelihood minus the KL divergences of the L single effects
        expected_loglik = -0.5 * self.n * np.log(2.0 * np.pi * self.sigma2) - 0.5 / self.sigma2 * self.get_expected_rss()
        return expected_loglik - np.sum(self.KL)

    def get_pips(self, prior_tol=1e-9):
        # PIP_j = 1 - prod_l (1 - alpha_lj), over single effects whose prior variance was not estimated to be zero
        keep = self.V > prior_tol
        if np.sum(keep) == 0:
            return np.zeros(self.p)
        return 1.0 - np.prod(1.0 - self.alpha[keep, :], axis=0)

    def get_credible_sets(self, prior_tol=1e-9):
        # Credible set of each single effect: the smallest set of SNPs whose alpha sums to at least the coverage.
        # Sets are dropped if the effect's prior variance is zero, if purity (min |r| between members) is below
        # min_abs_corr, or if they duplicate an earlier set.
        credible_sets = []
        seen = set()
        for l in range(self.L):
            if self.V[l] <= prior_tol:
                continue
            ordering = np.argsort(-self.alpha[l, :])
            n_in_cs = min(np.searchsorted(np.cumsum(self.alpha[l, ordering]), self.coverage) + 1, self.p)
            snp_indices = np.sort(ordering[:n_in_cs])
            purity = np.min(np.abs(self.ld_mat[np.ix_(snp_indices, snp_indices)]))
            if purity < self.min_abs_corr:
                continue
            key = tuple(snp_indices)
            if key in seen:
                continue
            seen.add(key)
            credible_sets.append({'effect_index': l, 'snp_indices': snp_indices, 'coverage': np.sum(self.alpha[l, snp_indices]), 'purity': purity})
        return credible_sets
