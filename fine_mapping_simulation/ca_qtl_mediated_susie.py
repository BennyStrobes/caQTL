import numpy as np
from susie_rss import SUSIE_RSS


#############################################################
# Generic Bayesian single effect regression (SER)
# A single effect is one SNP j (prior weight pi_j) carrying an effect ~ N(0, V). Given a per-SNP score S_j and
# precision P_j, so that the log likelihood in the effect b is S_j b - 0.5 P_j b^2, the posterior over (j, b) is
# closed form. With S = X'r / sigma2 and P = X'X_jj / sigma2 this is the SuSiE SER; here S and P may also pool
# evidence from several likelihood terms.
#############################################################
def ser_log_bayes_factors(V, betahat, shat2):
    return 0.5 * np.log(shat2 / (V + shat2)) + 0.5 * (betahat**2 / shat2) * (V / (V + shat2))


def ser_loglik(V, betahat, shat2, log_prior_weights):
    log_post = ser_log_bayes_factors(V, betahat, shat2) + log_prior_weights
    max_log_post = np.max(log_post)
    return max_log_post + np.log(np.sum(np.exp(log_post - max_log_post)))


def optimize_ser_prior_variance(betahat, shat2, log_prior_weights, V_current):
    # Maximize the SER log likelihood over log(V) on [-30, 15] by golden-section search (as in susie_rss.py)
    lo = -30.0
    hi = 15.0
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - golden * (hi - lo)
    x2 = lo + golden * (hi - lo)
    f1 = ser_loglik(np.exp(x1), betahat, shat2, log_prior_weights)
    f2 = ser_loglik(np.exp(x2), betahat, shat2, log_prior_weights)
    for _ in range(50):
        if f1 < f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + golden * (hi - lo)
            f2 = ser_loglik(np.exp(x2), betahat, shat2, log_prior_weights)
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - golden * (hi - lo)
            f1 = ser_loglik(np.exp(x1), betahat, shat2, log_prior_weights)
    V_new = np.exp(0.5 * (lo + hi))
    if ser_loglik(V_new, betahat, shat2, log_prior_weights) < ser_loglik(V_current, betahat, shat2, log_prior_weights):
        V_new = V_current
    if ser_loglik(0.0, betahat, shat2, log_prior_weights) >= ser_loglik(V_new, betahat, shat2, log_prior_weights):
        V_new = 0.0
    return V_new


def single_effect_regression(S, P, V, log_prior_weights, estimate_prior_variance):
    # Returns alpha (p), mu (p), mu2 (p), post_var (p), log BF of the single effect vs no effect, and V
    betahat = S / P
    shat2 = 1.0 / P
    if estimate_prior_variance:
        V = optimize_ser_prior_variance(betahat, shat2, log_prior_weights, V)
    log_post = ser_log_bayes_factors(V, betahat, shat2) + log_prior_weights
    max_log_post = np.max(log_post)
    weights = np.exp(log_post - max_log_post)
    alpha = weights / np.sum(weights)
    lbf_model = max_log_post + np.log(np.sum(weights))
    if V > 0.0:
        post_var = 1.0 / (1.0 / V + P)
    else:
        post_var = np.zeros(len(S))
    mu = post_var * S
    mu2 = post_var + mu**2
    return alpha, mu, mu2, post_var, lbf_model, V


def ser_kl(alpha, mu, post_var, V, log_prior_weights):
    # KL(q || prior) of a single effect: SNP assignment term plus the Gaussian effect term given the assignment
    kl = np.sum(alpha * (np.log(alpha + 1e-300) - log_prior_weights))
    if V > 0.0:
        kl = kl + np.sum(alpha * 0.5 * ((post_var + mu**2) / V - 1.0 - np.log(post_var / V)))
    return kl


def xlogx_ratio(q, pi):
    # q * log(q / pi) with the convention 0 log 0 = 0
    return np.where(q > 0.0, q * (np.log(q + 1e-300) - np.log(pi)), 0.0)


def spike_slab_kl(q, m, v, pi, tau2):
    # KL(q || prior) for g = z h, z ~ Bern(pi), h ~ N(0, tau2), with q(z = 1) = q and q(h | z = 1) = N(m, v)
    kl = xlogx_ratio(q, pi) + xlogx_ratio(1.0 - q, 1.0 - pi)
    kl = kl + q * 0.5 * ((v + m**2) / tau2 - 1.0 - np.log(v / tau2))
    return kl


class CAQTL_MEDIATED_SUSIE(object):
    """
    Joint fine-mapping of a gene's eQTL with the caQTLs of its nearby peaks, fit across genes.

    Per gene, with X the standardized genotypes of the cis SNPs:
        A_k    = X b_k + e_k                          caQTL of peak k,   b_k = sum_l b_kl  (SuSiE single effects)
        E      = X (sum_k g_k b_k + d) + e            eQTL,              d   = sum_m d_m   (SuSiE single effects)
        ghat_k ~ N(lambda g_k, se_k^2 + tau_u^2)      observed peak-gene link
        g_k    = z_k h_k,  z_k ~ Bern(pi),  h_k ~ N(0, tau_g^2)    spike-and-slab peak-gene effect
    pi and tau_g^2 (and optionally the link scale lambda and link bias variance tau_u^2) are shared across genes
    and estimated by EM. Everything else is gene specific: SuSiE prior variances per single effect, residual
    variances per trait. Inference is mean-field variational: every single effect and every g_k has an exact
    closed-form conditional posterior, updated by coordinate ascent on the ELBO (IBSS extended to the mediated
    model). LD is loaded from disk one gene at a time; all other per-gene quantities stay in memory.

    Inputs: per gene, marginal OLS effects and SEs on standardized genotypes for the eQTL (p) and for each
    peak's caQTL (K x p), the LD matrix file, link estimates and SEs (K), plus the two sample sizes.
    Main output: the eQTL PIP of each SNP, the posterior probability that it affects expression directly or
    through a linked peak.
    """

    def __init__(self, L_eqtl=10, L_caqtl=5, link_prior_prob=0.2, link_prior_variance=None, estimate_link_prior=True, estimate_link_scale=False, estimate_link_bias_variance=False, estimate_prior_variance=True, estimate_residual_variance=True, n_inner_iter=5, max_outer_iter=100, tol=1e-3, verbose=True):
        self.L_eqtl = L_eqtl                                  # single effects for the direct eQTL component d
        self.L_caqtl = L_caqtl                                # single effects per peak for b_k
        self.link_prior_prob = link_prior_prob                # pi: initial (or fixed) prior probability that a link is non-zero
        self.link_prior_variance = link_prior_variance        # tau_g^2: initial slab variance; None = method of moments from the links
        self.estimate_link_prior = estimate_link_prior        # update pi and tau_g^2 across genes
        self.estimate_link_scale = estimate_link_scale        # estimate lambda (otherwise fixed at 1)
        self.estimate_link_bias_variance = estimate_link_bias_variance   # estimate tau_u^2 (otherwise fixed at 0)
        self.estimate_prior_variance = estimate_prior_variance           # SuSiE prior variances per single effect
        self.estimate_residual_variance = estimate_residual_variance     # residual variance per trait per gene
        self.n_inner_iter = n_inner_iter                      # coordinate ascent passes per gene per outer iteration
        self.max_outer_iter = max_outer_iter
        self.tol = tol
        self.verbose = verbose

        # Filled in by fit()
        self.link_scale = 1.0
        self.link_bias_variance = 0.0
        self.elbo = None
        self.converged = None
        self.n_iter = None
        self.gene_results = None

    ###############################
    # Fitting across genes
    ###############################
    def fit(self, input_data, n_eqtl, n_caqtl):
        # input_data: dict with lists over genes: eqtl_effects (p), eqtl_ses (p), caqtl_effects (K x p), caqtl_ses (K x p),
        # peak_gene_effects (K), peak_gene_ses (K), ld_file_names
        self.n_eqtl = n_eqtl
        self.n_caqtl = n_caqtl
        n_genes = len(input_data['ld_file_names'])
        self.data = [self.prepare_gene_data(input_data, g) for g in range(n_genes)]
        self.state = [None] * n_genes

        # Initialize shared link hyperparameters
        self.pi = self.link_prior_prob
        if self.link_prior_variance is None:
            ghat = np.concatenate([d['ghat'] for d in self.data])
            gse = np.concatenate([d['gse'] for d in self.data])
            self.tau2_g = max((np.mean(ghat**2) - np.mean(gse**2)) / self.pi, 1e-3)
        else:
            self.tau2_g = self.link_prior_variance
        self.link_scale = 1.0
        self.link_bias_variance = 0.0

        self.elbo = []
        self.converged = False
        for outer_iter in range(self.max_outer_iter):
            genetic_elbo = 0.0
            for g in range(n_genes):
                R = np.load(self.data[g]['ld_file']).astype(float)
                if self.state[g] is None:
                    # Multi-start: direct effects initialized at null (the mediated pathway gets first claim on shared signal)
                    # or from eQTL-only SuSiE (the direct component does). Keep whichever reaches the higher ELBO.
                    state_null = self.initialize_gene(self.data[g], R)
                    state_susie = self.copy_state(state_null)
                    self.set_direct_effects_from_susie(self.data[g], state_susie, R)
                    candidates = []
                    for state in [state_null, state_susie]:
                        genetic = self.update_gene(self.data[g], state, R)
                        candidates.append((genetic + self.link_terms(state['q'], state['m'], state['v'], self.data[g]['ghat'], self.data[g]['gse']), genetic, state))
                    best = max(candidates, key=lambda c: c[0])
                    self.state[g] = best[2]
                    genetic_elbo = genetic_elbo + best[1]
                else:
                    genetic_elbo = genetic_elbo + self.update_gene(self.data[g], self.state[g], R)
                del R
            # Shared hyperparameter updates (EM), then the exact total ELBO under the updated hyperparameters
            self.update_link_hyperparameters()
            self.elbo.append(genetic_elbo + self.link_elbo_terms())
            if self.verbose:
                print('outer iter ' + str(outer_iter + 1) + ': ELBO ' + str(round(self.elbo[-1], 3)) + ', pi ' + str(round(self.pi, 4)) + ', tau2_g ' + str(round(self.tau2_g, 4)) + ', lambda ' + str(round(self.link_scale, 4)) + ', tau2_u ' + str(round(self.link_bias_variance, 5)), flush=True)
            if outer_iter > 0 and self.elbo[-1] - self.elbo[-2] < self.tol * n_genes:
                self.converged = True
                break
        self.n_iter = outer_iter + 1
        self.elbo = np.array(self.elbo)
        if not self.converged:
            print('warning: did not converge in ' + str(self.max_outer_iter) + ' outer iterations')

        # Summaries
        self.gene_results = [self.summarize_gene(self.data[g], self.state[g]) for g in range(n_genes)]
        return self

    def prepare_gene_data(self, input_data, g):
        # Sufficient statistics that do not need LD (X'X = c R is formed from LD when the gene is processed)
        beta_E = np.asarray(input_data['eqtl_effects'][g], dtype=float)
        se_E = np.asarray(input_data['eqtl_ses'][g], dtype=float)
        beta_A = np.atleast_2d(np.asarray(input_data['caqtl_effects'][g], dtype=float))
        se_A = np.atleast_2d(np.asarray(input_data['caqtl_ses'][g], dtype=float))
        nE = self.n_eqtl
        nA = self.n_caqtl
        data = {}
        data['ld_file'] = input_data['ld_file_names'][g]
        data['p'] = len(beta_E)
        data['K'] = beta_A.shape[0]
        data['cE'] = nE - 1.0
        data['cA'] = nA - 1.0
        # OLS on standardized genotypes: X'y = (n-1) beta_hat, and y'y from the SE identity (see susie_rss.py)
        data['XtE'] = data['cE'] * beta_E
        data['EtE'] = np.median(data['cE'] * (se_E**2 * (nE - 2.0) + beta_E**2))
        data['XtA'] = data['cA'] * beta_A
        data['AtA'] = np.median(data['cA'] * (se_A**2 * (nA - 2.0) + beta_A**2), axis=1)
        data['beta_A'] = beta_A
        data['se_A'] = se_A
        data['ghat'] = np.asarray(input_data['peak_gene_effects'][g], dtype=float)
        data['gse'] = np.asarray(input_data['peak_gene_ses'][g], dtype=float)
        data['log_prior_weights'] = np.log(np.ones(data['p']) / data['p'])
        return data

    ###############################
    # Per-gene initialization and coordinate ascent
    ###############################
    def initialize_gene(self, data, R):
        p = data['p']
        K = data['K']
        LA = self.L_caqtl
        LE = self.L_eqtl
        state = {}
        # caQTL effects: SuSiE on each peak's caQTL sumstats alone
        state['b_alpha'] = np.zeros((K, LA, p))
        state['b_mu'] = np.zeros((K, LA, p))
        state['b_mu2'] = np.zeros((K, LA, p))
        state['b_post_var'] = np.zeros((K, LA, p))
        state['b_V'] = np.zeros((K, LA))
        state['sigma2_A'] = np.zeros(K)
        for k in range(K):
            fit = SUSIE_RSS(L=LA, estimate_prior_variance=self.estimate_prior_variance, estimate_residual_variance=self.estimate_residual_variance).fit(data['beta_A'][k], data['se_A'][k], R, self.n_caqtl)
            state['b_alpha'][k] = fit.alpha
            state['b_mu'][k] = fit.mu
            state['b_mu2'][k] = fit.mu2
            state['b_post_var'][k] = fit.mu2 - fit.mu**2
            state['b_V'][k] = fit.V
            state['sigma2_A'][k] = fit.sigma2
        # Direct eQTL effects: start at null so the mediated pathway gets first claim on shared signal
        var_E = data['EtE'] / data['cE']
        state['d_alpha'] = np.ones((LE, p)) / p
        state['d_mu'] = np.zeros((LE, p))
        state['d_mu2'] = np.zeros((LE, p))
        state['d_post_var'] = np.zeros((LE, p))
        state['d_V'] = np.ones(LE) * 0.2 * var_E
        state['sigma2_E'] = var_E
        # Links: spike-and-slab posterior given the observed link alone
        state['q'] = np.zeros(K)
        state['m'] = np.zeros(K)
        state['v'] = np.zeros(K)
        for k in range(K):
            w = data['gse'][k]**2 + self.link_bias_variance
            S_g = self.link_scale * data['ghat'][k] / w
            P_g = self.link_scale**2 / w
            state['q'][k], state['m'][k], state['v'][k] = self.spike_slab_posterior(S_g, P_g)
        state['elbo'] = -np.inf
        state['init'] = 'null'
        return state

    def copy_state(self, state):
        return {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in state.items()}

    def set_direct_effects_from_susie(self, data, state, R):
        # Overwrite the direct eQTL component with an eQTL-only SuSiE fit
        fit = SUSIE_RSS(L=self.L_eqtl, estimate_prior_variance=self.estimate_prior_variance, estimate_residual_variance=self.estimate_residual_variance).fit(data['XtE'] / data['cE'], np.sqrt((data['EtE'] / data['cE'] - (data['XtE'] / data['cE'])**2) / (self.n_eqtl - 2.0)), R, self.n_eqtl)
        state['d_alpha'] = fit.alpha.copy()
        state['d_mu'] = fit.mu.copy()
        state['d_mu2'] = fit.mu2.copy()
        state['d_post_var'] = fit.mu2 - fit.mu**2
        state['d_V'] = fit.V.copy()
        state['sigma2_E'] = fit.sigma2
        state['init'] = 'susie'
        return state

    def spike_slab_posterior(self, S_g, P_g):
        v = 1.0 / (1.0 / self.tau2_g + P_g)
        m = v * S_g
        lbf = 0.5 * np.log(v / self.tau2_g) + 0.5 * v * S_g**2
        log_odds = np.log(self.pi) - np.log(1.0 - self.pi) + lbf
        q = 1.0 / (1.0 + np.exp(-log_odds))
        return q, m, v

    def update_gene(self, data, state, R):
        # Coordinate ascent passes for one gene. Returns the gene's ELBO contribution excluding the link terms
        # (those depend on the shared hyperparameters and are added after the global update).
        p = data['p']
        K = data['K']
        LA = self.L_caqtl
        LE = self.L_eqtl
        cE = data['cE']
        cA = data['cA']
        lpw = data['log_prior_weights']

        # Expected effects and their LD products, per single effect and summed
        Eb_l = state['b_alpha'] * state['b_mu']                       # K x LA x p
        REb_l = np.einsum('klp,pq->klq', Eb_l, R)                     # R times each caQTL single effect
        Ed_m = state['d_alpha'] * state['d_mu']                       # LE x p
        REd_m = np.dot(Ed_m, R)
        Eb = np.sum(Eb_l, axis=1)                                     # K x p
        REb = np.sum(REb_l, axis=1)
        Ed = np.sum(Ed_m, axis=0)
        REd = np.sum(REd_m, axis=0)
        Eg = state['q'] * state['m']                                  # E[g_k]
        Eg2 = state['q'] * (state['m']**2 + state['v'])               # E[g_k^2]

        def expected_bRb(k):
            # E[b_k' R b_k] under the SuSiE posterior of peak k
            return np.dot(Eb[k], REb[k]) - np.sum(Eb_l[k] * REb_l[k]) + np.sum(state['b_alpha'][k] * state['b_mu2'][k])

        def expected_dRd():
            return np.dot(Ed, REd) - np.sum(Ed_m * REd_m) + np.sum(state['d_alpha'] * state['d_mu2'])

        for inner_iter in range(self.n_inner_iter):
            # (1) Links g_k: evidence from the eQTL row (covariance of the peak's caQTL profile with the eQTL residual) and the link row
            for k in range(K):
                resid_score = data['XtE'] - cE * (np.dot(Eg, REb) - Eg[k] * REb[k] + REd)    # E[X'(E - X rest_k)]
                w = data['gse'][k]**2 + self.link_bias_variance
                S_g = np.dot(Eb[k], resid_score) / state['sigma2_E'] + self.link_scale * data['ghat'][k] / w
                P_g = cE * expected_bRb(k) / state['sigma2_E'] + self.link_scale**2 / w
                state['q'][k], state['m'][k], state['v'][k] = self.spike_slab_posterior(S_g, P_g)
                Eg[k] = state['q'][k] * state['m'][k]
                Eg2[k] = state['q'][k] * (state['m'][k]**2 + state['v'][k])

            # (2) Direct eQTL single effects d_m: SER on the eQTL residual after the mediated component and the other direct effects
            R_Emed = np.dot(Eg, REb)                                   # R times the expected mediated effect
            for m in range(LE):
                Xtr = data['XtE'] - cE * (R_Emed + REd - REd_m[m])
                S = Xtr / state['sigma2_E']
                P = np.ones(p) * cE / state['sigma2_E']
                alpha, mu, mu2, post_var, lbf, V = single_effect_regression(S, P, state['d_V'][m], lpw, self.estimate_prior_variance)
                state['d_alpha'][m] = alpha
                state['d_mu'][m] = mu
                state['d_mu2'][m] = mu2
                state['d_post_var'][m] = post_var
                state['d_V'][m] = V
                Ed_m[m] = alpha * mu
                REd = REd - REd_m[m]
                REd_m[m] = np.dot(R, Ed_m[m])
                REd = REd + REd_m[m]
            Ed = np.sum(Ed_m, axis=0)

            # (3) caQTL single effects b_kl: SER pooling the caQTL row and the eQTL row (through g_k)
            for k in range(K):
                # eQTL residual excluding peak k entirely: E[X'(E - X(sum_{k' != k} g_k' b_k' + d))]
                resid_other = data['XtE'] - cE * (np.dot(Eg, REb) - Eg[k] * REb[k] + REd)
                for l in range(LA):
                    score_A = (data['XtA'][k] - cA * (REb[k] - REb_l[k, l])) / state['sigma2_A'][k]
                    score_E = (Eg[k] * resid_other - Eg2[k] * cE * (REb[k] - REb_l[k, l])) / state['sigma2_E']
                    S = score_A + score_E
                    P = np.ones(p) * (cA / state['sigma2_A'][k] + Eg2[k] * cE / state['sigma2_E'])
                    alpha, mu, mu2, post_var, lbf, V = single_effect_regression(S, P, state['b_V'][k, l], lpw, self.estimate_prior_variance)
                    state['b_alpha'][k, l] = alpha
                    state['b_mu'][k, l] = mu
                    state['b_mu2'][k, l] = mu2
                    state['b_post_var'][k, l] = post_var
                    state['b_V'][k, l] = V
                    Eb_l[k, l] = alpha * mu
                    REb[k] = REb[k] - REb_l[k, l]
                    REb_l[k, l] = np.dot(R, Eb_l[k, l])
                    REb[k] = REb[k] + REb_l[k, l]
                Eb[k] = np.sum(Eb_l[k], axis=0)

            # (4) Residual variances, from the expected residual sums of squares
            bRb = np.array([expected_bRb(k) for k in range(K)])
            erss_A = data['AtA'] - 2.0 * np.sum(Eb * data['XtA'], axis=1) + cA * bRb
            Emed = np.dot(Eg, Eb)
            Ebeta = Emed + Ed
            EbetaRbeta = np.sum(Eg2 * bRb) + np.dot(Eg, np.dot(np.dot(Eb, REb.T), Eg)) - np.sum(Eg**2 * np.sum(Eb * REb, axis=1)) + 2.0 * np.dot(Emed, REd) + expected_dRd()
            erss_E = data['EtE'] - 2.0 * np.dot(Ebeta, data['XtE']) + cE * EbetaRbeta
            if self.estimate_residual_variance:
                state['sigma2_A'] = erss_A / self.n_caqtl
                state['sigma2_E'] = erss_E / self.n_eqtl
                if np.any(state['sigma2_A'] <= 0.0) or state['sigma2_E'] <= 0.0:
                    raise ValueError('estimated residual variance is negative: summary statistics and LD are inconsistent')

            # (5) Gene ELBO without the link terms
            elbo = np.sum(-0.5 * self.n_caqtl * np.log(2.0 * np.pi * state['sigma2_A']) - 0.5 * erss_A / state['sigma2_A'])
            elbo = elbo - 0.5 * self.n_eqtl * np.log(2.0 * np.pi * state['sigma2_E']) - 0.5 * erss_E / state['sigma2_E']
            for k in range(K):
                for l in range(LA):
                    elbo = elbo - ser_kl(state['b_alpha'][k, l], state['b_mu'][k, l], state['b_post_var'][k, l], state['b_V'][k, l], lpw)
            for m in range(LE):
                elbo = elbo - ser_kl(state['d_alpha'][m], state['d_mu'][m], state['d_post_var'][m], state['d_V'][m], lpw)
            converged = (elbo - state['elbo']) < self.tol
            state['elbo'] = elbo
            if converged:
                break
        return state['elbo']

    ###############################
    # Shared link hyperparameters (EM) and the link part of the ELBO
    ###############################
    def stacked_link_quantities(self):
        q = np.concatenate([s['q'] for s in self.state])
        m = np.concatenate([s['m'] for s in self.state])
        v = np.concatenate([s['v'] for s in self.state])
        ghat = np.concatenate([d['ghat'] for d in self.data])
        gse = np.concatenate([d['gse'] for d in self.data])
        return q, m, v, ghat, gse

    def update_link_hyperparameters(self):
        q, m, v, ghat, gse = self.stacked_link_quantities()
        Eg = q * m
        Eg2 = q * (m**2 + v)
        if self.estimate_link_prior:
            self.pi = np.clip(np.mean(q), 1e-6, 1.0 - 1e-6)
            self.tau2_g = max(np.sum(q * (m**2 + v)) / max(np.sum(q), 1e-300), 1e-8)
        if self.estimate_link_scale:
            w = gse**2 + self.link_bias_variance
            self.link_scale = np.sum(ghat * Eg / w) / np.sum(Eg2 / w)
        if self.estimate_link_bias_variance:
            # Maximize the link log likelihood over log(tau_u^2) by golden-section search, checking tau_u^2 = 0 too
            def f(tau2_u):
                w = gse**2 + tau2_u
                return np.sum(-0.5 * np.log(w) - (ghat**2 - 2.0 * self.link_scale * ghat * Eg + self.link_scale**2 * Eg2) / (2.0 * w))
            lo = -30.0
            hi = 5.0
            golden = (np.sqrt(5.0) - 1.0) / 2.0
            x1 = hi - golden * (hi - lo)
            x2 = lo + golden * (hi - lo)
            f1 = f(np.exp(x1))
            f2 = f(np.exp(x2))
            for _ in range(50):
                if f1 < f2:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + golden * (hi - lo)
                    f2 = f(np.exp(x2))
                else:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - golden * (hi - lo)
                    f1 = f(np.exp(x1))
            tau2_u = np.exp(0.5 * (lo + hi))
            self.link_bias_variance = tau2_u if f(tau2_u) > f(0.0) else 0.0

    def link_terms(self, q, m, v, ghat, gse):
        # Expected log likelihood of observed links minus the KL of their spike-and-slab posteriors, under the current hyperparameters
        Eg = q * m
        Eg2 = q * (m**2 + v)
        w = gse**2 + self.link_bias_variance
        loglik = np.sum(-0.5 * np.log(2.0 * np.pi * w) - (ghat**2 - 2.0 * self.link_scale * ghat * Eg + self.link_scale**2 * Eg2) / (2.0 * w))
        return loglik - np.sum(spike_slab_kl(q, m, v, self.pi, self.tau2_g))

    def link_elbo_terms(self):
        q, m, v, ghat, gse = self.stacked_link_quantities()
        return self.link_terms(q, m, v, ghat, gse)

    ###############################
    # Summaries
    ###############################
    def summarize_gene(self, data, state, prior_tol=1e-9):
        K = data['K']
        p = data['p']
        res = {}
        # caQTL PIPs per peak, over single effects with non-zero prior variance
        res['caqtl_pip'] = np.zeros((K, p))
        for k in range(K):
            keep = state['b_V'][k] > prior_tol
            if np.sum(keep) > 0:
                res['caqtl_pip'][k] = 1.0 - np.prod(1.0 - state['b_alpha'][k][keep], axis=0)
        keep = state['d_V'] > prior_tol
        res['direct_pip'] = 1.0 - np.prod(1.0 - state['d_alpha'][keep], axis=0) if np.sum(keep) > 0 else np.zeros(p)
        # Mediated PIP: some peak has a non-zero link and this SNP as a caQTL causal variant
        res['mediated_pip'] = 1.0 - np.prod(1.0 - state['q'][:, None] * res['caqtl_pip'], axis=0)
        res['eqtl_pip'] = 1.0 - (1.0 - res['direct_pip']) * (1.0 - res['mediated_pip'])
        res['link_prob'] = state['q'].copy()                   # posterior probability the link is non-zero
        res['link_mean'] = state['q'] * state['m']             # posterior mean of g_k
        res['link_slab_mean'] = state['m'].copy()
        res['link_slab_var'] = state['v'].copy()
        Eb = np.sum(state['b_alpha'] * state['b_mu'], axis=1)
        Ed = np.sum(state['d_alpha'] * state['d_mu'], axis=0)
        res['eqtl_posterior_mean'] = np.dot(res['link_mean'], Eb) + Ed
        res['direct_posterior_mean'] = Ed
        res['caqtl_posterior_mean'] = Eb
        res['sigma2_A'] = state['sigma2_A'].copy()
        res['sigma2_E'] = state['sigma2_E']
        res['elbo'] = state['elbo']
        res['init'] = state['init']
        return res
