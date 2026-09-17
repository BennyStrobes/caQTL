import numpy as np
from susie_rss import SUSIE_RSS, zscore_quadratic_form


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
    If the sample sizes are None, the z-score model of Zou et al. (2022) is used for both traits: X'X = R,
    X'y = z = beta / se, residual variances fixed at 1. Effects are then on the z scale (standardized effect
    times sqrt(n)); the ratio sqrt(n_eqtl / n_caqtl) is absorbed by the link scale lambda, so estimate it.
    With zscore_residual_variance=True the z-score model gets a residual variance per trait per gene,
    z ~ N(R b, sigma2 R), estimated by maximum likelihood: y'y = z'R^-1 z and n = p make the Gaussian
    log likelihood exact (up to a constant), so sigma2 = E[RSS] / p. sigma2 > 1 means the z-scores are
    overdispersed relative to the LD null (polygenic background, LD mismatch).
    Genes with no linked peaks (K = 0) reduce to eQTL-only SuSiE.
    With link_inclusion=True (Model D) each link also carries an inclusion indicator u_k ~ Bern(pi_u) and the eQTL is
    E = X (sum_k u_k g_k b_k + d) + e: the hurdle estimate sets the implied mediated effect (ghat_k / lambda) and its sign,
    and the eQTL data alone decide whether that effect is included. A strong hurdle link with no eQTL support is excluded
    at no cost. A peak mediates when z_k u_k = 1; pi_u is the fraction of non-zero links that mediate.
    Main output: the eQTL PIP of each SNP, the posterior probability that it affects expression directly or
    through a linked peak.
    """

    def __init__(self, L_eqtl=10, L_caqtl=5, link_prior_prob=0.2, link_prior_variance=None, estimate_link_prior=True, estimate_link_scale=False, link_scale_init=1.0, estimate_link_bias_variance=False, link_bias_variance_init=0.0, estimate_prior_variance=True, estimate_residual_variance=True, zscore_residual_variance=False, link_inclusion=False, link_inclusion_prob_init=0.5, estimate_link_inclusion_prob=True, link_inclusion_prob_delay=0, link_inclusion_prior_mean=0.2, link_inclusion_prior_strength=0.0, n_inner_iter=5, max_outer_iter=100, tol=1e-3, hyper_tol=1e-3, verbose=True):
        self.L_eqtl = L_eqtl                                  # single effects for the direct eQTL component d
        self.L_caqtl = L_caqtl                                # single effects per peak for b_k
        self.link_prior_prob = link_prior_prob                # pi: initial (or fixed) prior probability that a link is non-zero
        self.link_prior_variance = link_prior_variance        # tau_g^2: initial slab variance; None = method of moments from the links
        self.estimate_link_prior = estimate_link_prior        # update pi and tau_g^2 across genes
        self.estimate_link_scale = estimate_link_scale        # estimate lambda (otherwise fixed at link_scale_init)
        self.link_scale_init = link_scale_init                # initial (or fixed) lambda; e.g. 1 / slope of eQTL effects on caQTL x link products
        self.estimate_link_bias_variance = estimate_link_bias_variance   # estimate tau_u^2 (otherwise fixed at link_bias_variance_init)
        self.link_bias_variance_init = link_bias_variance_init           # tau_u^2 init (or fixed value); measurement variance of a link is se_k^2 + tau_u^2 in both models
        self.estimate_prior_variance = estimate_prior_variance           # SuSiE prior variances per single effect
        self.estimate_residual_variance = estimate_residual_variance     # residual variance per trait per gene
        self.zscore_residual_variance = zscore_residual_variance         # z-score mode: estimate residual variances too (y'y = z'R^-1 z, n = p) instead of fixing them at 1
        # Link inclusion indicator (Model D): u_k | z_k = 1 ~ Bern(pi_u); the mediated term g_k b_k enters the eQTL model only when u_k = 1
        self.link_inclusion = link_inclusion
        self.link_inclusion_prob_init = link_inclusion_prob_init     # pi_u init (or fixed value)
        self.estimate_link_inclusion_prob = estimate_link_inclusion_prob
        self.link_inclusion_prob_delay = link_inclusion_prob_delay   # keep pi_u at its initial value for this many outer iterations before estimating it
        # Beta hyperprior on pi_u, parameterized as a prior mean and a strength in pseudo-links per gene (total pseudo-count =
        # strength x number of genes, so the prior keeps its weight as the gene set grows): pi_u ~ Beta(mean c n + 1, (1 - mean) c n + 1).
        # The MAP update is then (sum of inclusion probabilities + mean c n) / (sum of non-zero probabilities + c n). Strength 0 = no prior.
        self.link_inclusion_prior_mean = link_inclusion_prior_mean
        self.link_inclusion_prior_strength = link_inclusion_prior_strength
        self.n_inner_iter = n_inner_iter                      # coordinate ascent passes per gene per outer iteration
        self.max_outer_iter = max_outer_iter
        self.tol = tol
        self.hyper_tol = hyper_tol                            # also require every shared hyperparameter to move by less than this relative amount
        self.verbose = verbose

        # Filled in by fit()
        self.link_scale = 1.0
        self.link_bias_variance = 0.0
        self.pi_u = 1.0
        self.update_pi_u_now = False
        self.elbo = None
        self.converged = None
        self.n_iter = None
        self.gene_results = None

    ###############################
    # Fitting across genes
    ###############################
    def fit(self, input_data, n_eqtl=None, n_caqtl=None):
        # input_data: dict with lists over genes: eqtl_effects (p), eqtl_ses (p), caqtl_effects (K x p), caqtl_ses (K x p),
        # peak_gene_effects (K), peak_gene_ses (K), ld_file_names
        # n_eqtl / n_caqtl None: z-score model (see class docstring). Placeholder n = 2 only enters the ELBO as a constant.
        self.zscore_mode = (n_eqtl is None) or (n_caqtl is None)
        if self.zscore_mode and not ((n_eqtl is None) and (n_caqtl is None)):
            raise ValueError('n_eqtl and n_caqtl must both be given or both be None')
        self.n_eqtl = 2 if self.zscore_mode else n_eqtl
        self.n_caqtl = 2 if self.zscore_mode else n_caqtl
        self.n_eqtl_fit = None if self.zscore_mode else n_eqtl      # what is passed to SUSIE_RSS.fit
        self.n_caqtl_fit = None if self.zscore_mode else n_caqtl
        self.zscore_resid = self.zscore_mode and self.zscore_residual_variance and self.estimate_residual_variance
        n_genes = len(input_data['ld_file_names'])
        self.n_genes = n_genes
        self.data = [self.prepare_gene_data(input_data, g) for g in range(n_genes)]
        self.state = [None] * n_genes
        self.gene_elbo_increment = np.zeros(n_genes)   # change in each gene's ELBO during the most recent outer iteration

        # Initialize shared link hyperparameters
        self.pi = self.link_prior_prob
        if self.link_prior_variance is None:
            self.tau2_g = self.initial_link_prior_variance()
        else:
            self.tau2_g = self.link_prior_variance
        self.link_scale = self.link_scale_init
        self.link_bias_variance = float(self.link_bias_variance_init)
        self.pi_u = 1.0
        if self.link_inclusion:
            self.pi_u = float(np.clip(self.link_inclusion_prob_init, 1e-6, 1.0 - 1e-6))
            if self.verbose:
                print('link inclusion: pi_u init ' + str(self.pi_u) + ((' (estimated after ' + str(self.link_inclusion_prob_delay) + ' outer iterations)') if self.estimate_link_inclusion_prob else ' (fixed)') + ((', Beta hyperprior mean ' + str(self.link_inclusion_prior_mean) + ', strength ' + str(self.link_inclusion_prior_strength) + ' pseudo-links per gene (' + str(int(round(self.link_inclusion_prior_strength * self.n_genes))) + ' total)') if self.link_inclusion_prior_strength > 0.0 else ', no hyperprior'), flush=True)

        self.elbo = []
        self.converged = False
        for outer_iter in range(self.max_outer_iter):
            genetic_elbo = 0.0
            for g in range(n_genes):
                R = np.load(self.data[g]['ld_file']).astype(float)
                if not np.all(np.isfinite(R)):
                    raise ValueError('LD matrix contains non-finite values: ' + self.data[g]['ld_file'])
                if self.state[g] is None:
                    if self.zscore_resid:
                        self.set_zscore_yty(self.data[g], R)
                    # Multi-start: direct effects initialized at null (the mediated pathway gets first claim on shared signal)
                    # or from eQTL-only SuSiE with all mediated terms zeroed (the direct component does; links must earn
                    # inclusion against the residual). Keep whichever reaches the higher ELBO after one gene update.
                    state_null = self.initialize_gene(self.data[g], R)
                    state_susie = self.copy_state(state_null)
                    self.set_direct_effects_from_susie(self.data[g], state_susie, R)
                    candidates = []
                    for state in [state_null, state_susie]:
                        genetic = self.update_gene(self.data[g], state, R)
                        candidates.append((genetic + self.link_terms(self.link_state_dict(state, self.data[g])), genetic, state))
                    best = max(candidates, key=lambda c: c[0])
                    self.state[g] = best[2]
                    genetic_elbo = genetic_elbo + best[1]
                else:
                    elbo_before = self.state[g]['elbo']
                    genetic_elbo = genetic_elbo + self.update_gene(self.data[g], self.state[g], R)
                    self.gene_elbo_increment[g] = self.state[g]['elbo'] - elbo_before
                if not np.isfinite(self.state[g]['elbo']) or not np.all(np.isfinite(self.state[g]['q'])):
                    raise ValueError('non-finite ELBO or link posterior for gene with LD file ' + self.data[g]['ld_file'])
                del R
            # Shared hyperparameter updates (EM), then the exact total ELBO under the updated hyperparameters
            hyper_before = np.array([self.pi, self.tau2_g, self.link_scale, self.link_bias_variance, self.pi_u])
            self.update_pi_u_now = self.estimate_link_inclusion_prob and outer_iter >= self.link_inclusion_prob_delay
            self.update_link_hyperparameters()
            hyper_after = np.array([self.pi, self.tau2_g, self.link_scale, self.link_bias_variance, self.pi_u])
            hyper_change = np.max(np.abs(hyper_after - hyper_before) / np.maximum(np.abs(hyper_before), 1e-12))
            self.elbo.append(genetic_elbo + self.link_elbo_terms() + self.log_pi_u_prior())
            if self.verbose:
                print('outer iter ' + str(outer_iter + 1) + ': ELBO ' + str(round(self.elbo[-1], 3)) + ', pi ' + str(round(self.pi, 4)) + ', tau2_g ' + '%.3e' % self.tau2_g + ', lambda ' + str(round(self.link_scale, 4)) + ', tau2_u ' + '%.3e' % self.link_bias_variance + (', pi_u ' + str(round(self.pi_u, 4)) if self.link_inclusion else ''), flush=True)
            if outer_iter > 0 and self.elbo[-1] - self.elbo[-2] < self.tol * n_genes and hyper_change < self.hyper_tol:
                self.converged = True
                break
        self.n_iter = outer_iter + 1
        self.elbo = np.array(self.elbo)
        if not self.converged:
            print('warning: did not converge in ' + str(self.max_outer_iter) + ' outer iterations')

        # Summaries
        self.gene_results = [self.summarize_gene(self.data[g], self.state[g]) for g in range(n_genes)]
        return self

    def initial_link_prior_variance(self):
        # Method of moments for tau_g^2 from the links: E[ghat^2] = pi lambda^2 tau_g^2 + E[se^2]. Robust to a few links with
        # huge SEs (they would make the excess negative): only links with finite ghat and 0 < se <= 10 x median(se) are used.
        # If the excess is not positive the links look like pure noise; start from a fraction of their spread instead.
        # Starting at the 1e-8 floor is fatal (the prior precision crushes every link posterior and the EM update returns
        # the floor again), so that is an error rather than a warning.
        ghat = np.concatenate([d['ghat'] for d in self.data])
        gse = np.concatenate([d['gse'] for d in self.data])
        if len(ghat) == 0:
            return 1e-8
        ok = np.isfinite(ghat) & np.isfinite(gse) & (gse > 0.0)
        if np.sum(ok) == 0:
            raise ValueError('no link has a finite estimate and a positive SE')
        ok = ok & (gse <= 10.0 * np.median(gse[ok]))
        excess = np.mean(ghat[ok]**2) - np.mean(gse[ok]**2)
        if excess <= 0.0:
            print('warning: link estimates are not overdispersed relative to their SEs (mean ghat^2 <= mean se^2); initializing tau_g^2 from 0.1 x mean ghat^2', flush=True)
            excess = 0.1 * np.mean(ghat[ok]**2)
        tau2_g = excess / self.pi / self.link_scale_init**2
        if not np.isfinite(tau2_g) or tau2_g <= 1e-7:
            raise ValueError('initial tau_g^2 = ' + str(tau2_g) + ' is at the floor; the links carry no usable signal at link_scale_init = ' + str(self.link_scale_init))
        if self.verbose:
            print('initial tau2_g ' + '%.3e' % tau2_g + ' from ' + str(int(np.sum(ok))) + ' of ' + str(len(ghat)) + ' links (pi ' + str(self.pi) + ', lambda ' + str(self.link_scale_init) + ')', flush=True)
        return tau2_g

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
        if self.zscore_mode:
            # z-score model: X'X = R, X'y = z, residual variance 1 (y'y placeholders give var_y = 1 and only shift the ELBO by a constant).
            # With zscore_resid, y'y = z'R^-1 z and n = p are filled in by set_zscore_yty when the gene's LD is first loaded.
            data['cE'] = 1.0
            data['cA'] = 1.0
            data['XtE'] = beta_E / se_E
            data['EtE'] = 1.0
            data['XtA'] = beta_A / se_A
            data['AtA'] = np.ones(data['K'])
            data['nE'] = data['p'] if self.zscore_resid else 2
            data['nA'] = data['p'] if self.zscore_resid else 2
            data['beta_A'] = beta_A / se_A
            data['se_A'] = np.ones(beta_A.shape)
            data['beta_E'] = beta_E / se_E
            data['se_E'] = np.ones(data['p'])
        else:
            data['cE'] = nE - 1.0
            data['cA'] = nA - 1.0
            data['nE'] = nE
            data['nA'] = nA
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

    def set_zscore_yty(self, data, R):
        # z-score residual variance model: y'y = z'R^-1 z per trait, from one factorization of the gene's LD
        Z = np.column_stack([data['XtE'][:, None], data['XtA'].T]) if data['K'] > 0 else data['XtE'][:, None]
        quad = zscore_quadratic_form(R, Z)
        data['EtE'] = float(quad[0])
        data['AtA'] = np.asarray(quad[1:], dtype=float)
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
            fit = SUSIE_RSS(L=LA, estimate_prior_variance=self.estimate_prior_variance, estimate_residual_variance=self.estimate_residual_variance, zscore_residual_variance=self.zscore_resid).fit(data['beta_A'][k], data['se_A'][k], R, self.n_caqtl_fit, zscore_yty=(data['AtA'][k] if self.zscore_resid else None))
            state['b_alpha'][k] = fit.alpha
            state['b_mu'][k] = fit.mu
            state['b_mu2'][k] = fit.mu2
            state['b_post_var'][k] = fit.mu2 - fit.mu**2
            state['b_V'][k] = fit.V
            state['sigma2_A'][k] = fit.sigma2
        # Direct eQTL effects: start at null so the mediated pathway gets first claim on shared signal
        var_E = data['EtE'] / max(data['nE'] - 1.0, 1.0)
        state['d_alpha'] = np.ones((LE, p)) / p
        state['d_mu'] = np.zeros((LE, p))
        state['d_mu2'] = np.zeros((LE, p))
        state['d_post_var'] = np.zeros((LE, p))
        state['d_V'] = np.ones(LE) * 0.2 * var_E
        state['sigma2_E'] = var_E
        # Links: posterior given the observed link alone (no eQTL evidence yet)
        for key in self.link_state_keys:
            state[key] = np.zeros(K)
        for k in range(K):
            self.update_link_posterior(data, state, k, 0.0, 0.0)
        state['elbo'] = -np.inf
        state['init'] = 'null'
        return state

    def copy_state(self, state):
        return {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in state.items()}

    def set_direct_effects_from_susie(self, data, state, R):
        # Overwrite the direct eQTL component with an eQTL-only SuSiE fit, and zero every link's mediated term so the direct fit
        # is the only thing subtracted from the eQTL residual when the links are updated first (no double counting of shared
        # signal). The links' own hurdle-side state is left as is; each link factor is recomputed at its first update anyway.
        if self.zscore_mode:
            fit = SUSIE_RSS(L=self.L_eqtl, estimate_prior_variance=self.estimate_prior_variance, estimate_residual_variance=self.estimate_residual_variance, zscore_residual_variance=self.zscore_resid).fit(data['beta_E'], data['se_E'], R, None, zscore_yty=(data['EtE'] if self.zscore_resid else None))
        else:
            fit = SUSIE_RSS(L=self.L_eqtl, estimate_prior_variance=self.estimate_prior_variance, estimate_residual_variance=self.estimate_residual_variance).fit(data['XtE'] / data['cE'], np.sqrt((data['EtE'] / data['cE'] - (data['XtE'] / data['cE'])**2) / (self.n_eqtl - 2.0)), R, self.n_eqtl)
        state['d_alpha'] = fit.alpha.copy()
        state['d_mu'] = fit.mu.copy()
        state['d_mu2'] = fit.mu2.copy()
        state['d_post_var'] = fit.mu2 - fit.mu**2
        state['d_V'] = fit.V.copy()
        state['sigma2_E'] = fit.sigma2
        state['Eg'][:] = 0.0
        state['Eg2'][:] = 0.0
        if self.link_inclusion:
            state['r10'][:] = state['q']     # all non-zero mass in the excluded cell
            state['incl'][:] = 0.0
        state['init'] = 'susie'
        return state

    def spike_slab_posterior(self, S_g, P_g):
        v = 1.0 / (1.0 / self.tau2_g + P_g)
        m = v * S_g
        lbf = 0.5 * np.log(v / self.tau2_g) + 0.5 * v * S_g**2
        log_odds = np.log(self.pi) - np.log(1.0 - self.pi) + lbf
        q = 1.0 / (1.0 + np.exp(-log_odds))
        return q, m, v

    def update_link_posterior(self, data, state, k, S_E, P_E):
        # Posterior of link k given the eQTL-side score S_E and precision P_E (E[b_k' X'r] / sigma2_E and E[b_k' X'X b_k] / sigma2_E)
        # and the observed link. Writes into state: q = P(z_k = 1); incl = P(z_k u_k = 1) (= q without link inclusion);
        # Eg, Eg2 = E[u_k g_k], E[u_k g_k^2] (what the eQTL model sees); Egh, Eg2h = E[g_k], E[g_k^2] (what the link term sees);
        # m, v = slab moments of the cell that mediates; m0, v0 and r10 = moments and probability of the excluded cell (Model D).
        ghat = data['ghat'][k]
        w = data['gse'][k]**2 + self.link_bias_variance
        lam = self.link_scale
        S_h = lam * ghat / w
        P_h = lam**2 / w
        if not self.link_inclusion:
            q, m, v = self.spike_slab_posterior(S_E + S_h, P_E + P_h)
            state['q'][k] = q
            state['incl'][k] = q
            state['m'][k] = m
            state['v'][k] = v
            state['r10'][k] = 0.0
            state['m0'][k] = 0.0
            state['v0'][k] = 0.0
            state['Eg'][k] = q * m
            state['Eg2'][k] = q * (m**2 + v)
            state['Egh'][k] = state['Eg'][k]
            state['Eg2h'][k] = state['Eg2'][k]
            return
        # Model D cells: z=0 | z=1,u=0 (h from the hurdle alone) | z=1,u=1 (h from hurdle + eQTL). Common factor N(ghat; 0, w) dropped.
        v_h = 1.0 / (1.0 / self.tau2_g + P_h)
        m_h = v_h * S_h
        S1 = S_E + S_h
        P1 = P_E + P_h
        v1 = 1.0 / (1.0 / self.tau2_g + P1)
        m1 = v1 * S1
        lbf_h = 0.5 * np.log(v_h / self.tau2_g) + 0.5 * v_h * S_h**2                 # slab (hurdle only) vs spike
        lbf_u = 0.5 * np.log(v1 / v_h) + 0.5 * v1 * S1**2 - 0.5 * v_h * S_h**2         # included vs excluded: the eQTL evidence
        lw = np.array([
            np.log(1.0 - self.pi),
            np.log(self.pi) + np.log(1.0 - self.pi_u) + lbf_h,
            np.log(self.pi) + np.log(self.pi_u) + lbf_h + lbf_u])
        r = np.exp(lw - np.max(lw))
        r = r / np.sum(r)
        state['q'][k] = r[1] + r[2]
        state['incl'][k] = r[2]
        state['r10'][k] = r[1]
        state['m'][k] = m1
        state['v'][k] = v1
        state['m0'][k] = m_h
        state['v0'][k] = v_h
        state['Eg'][k] = r[2] * m1
        state['Eg2'][k] = r[2] * (m1**2 + v1)
        state['Egh'][k] = r[1] * m_h + r[2] * m1
        state['Eg2h'][k] = r[1] * (m_h**2 + v_h) + r[2] * (m1**2 + v1)

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
        Eg = state['Eg'].copy()                                       # E[g_k]
        Eg2 = state['Eg2'].copy()                                     # E[g_k^2]

        def expected_bRb(k):
            # E[b_k' R b_k] under the SuSiE posterior of peak k
            return np.dot(Eb[k], REb[k]) - np.sum(Eb_l[k] * REb_l[k]) + np.sum(state['b_alpha'][k] * state['b_mu2'][k])

        def expected_dRd():
            return np.dot(Ed, REd) - np.sum(Ed_m * REd_m) + np.sum(state['d_alpha'] * state['d_mu2'])

        for inner_iter in range(self.n_inner_iter):
            # (1) Links g_k: evidence from the eQTL row (covariance of the peak's caQTL profile with the eQTL residual) and the link row
            for k in range(K):
                resid_score = data['XtE'] - cE * (np.dot(Eg, REb) - Eg[k] * REb[k] + REd)    # E[X'(E - X rest_k)]
                S_E = np.dot(Eb[k], resid_score) / state['sigma2_E']
                P_E = cE * expected_bRb(k) / state['sigma2_E']
                self.update_link_posterior(data, state, k, S_E, P_E)
                Eg[k] = state['Eg'][k]
                Eg2[k] = state['Eg2'][k]

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
            if self.estimate_residual_variance and (not self.zscore_mode or self.zscore_resid):
                state['sigma2_A'] = erss_A / data['nA']
                state['sigma2_E'] = erss_E / data['nE']
                if np.any(state['sigma2_A'] <= 0.0) or state['sigma2_E'] <= 0.0:
                    raise ValueError('estimated residual variance is negative: summary statistics and LD are inconsistent')

            # (5) Gene ELBO without the link terms
            elbo = np.sum(-0.5 * data['nA'] * np.log(2.0 * np.pi * state['sigma2_A']) - 0.5 * erss_A / state['sigma2_A'])
            elbo = elbo - 0.5 * data['nE'] * np.log(2.0 * np.pi * state['sigma2_E']) - 0.5 * erss_E / state['sigma2_E']
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
    link_state_keys = ['q', 'incl', 'm', 'v', 'r10', 'm0', 'v0', 'Eg', 'Eg2', 'Egh', 'Eg2h']

    def link_state_dict(self, state, data):
        L = {key: state[key] for key in self.link_state_keys}
        L['ghat'] = data['ghat']
        L['gse'] = data['gse']
        return L

    def stacked_link_quantities(self):
        L = {key: np.concatenate([s[key] for s in self.state]) for key in self.link_state_keys}
        L['ghat'] = np.concatenate([d['ghat'] for d in self.data])
        L['gse'] = np.concatenate([d['gse'] for d in self.data])
        return L

    def update_link_hyperparameters(self):
        # One EM step per outer iteration for each shared hyperparameter: the mixing proportions are the sums of the current
        # posterior cell probabilities (pi_u with the Beta pseudo-counts), tau_g^2 the mean second moment of non-zero links
        L = self.stacked_link_quantities()
        if len(L['q']) == 0:
            return
        q = L['q']
        Egh = L['Egh']       # E[g_k]: the link term sees every non-zero link, included or not
        Eg2h = L['Eg2h']
        ghat = L['ghat']
        gse = L['gse']
        if self.estimate_link_prior:
            self.pi = float(np.clip(np.mean(q), 1e-6, 1.0 - 1e-6))
            self.tau2_g = max(np.sum(Eg2h) / max(np.sum(q), 1e-300), 1e-8)
        if self.link_inclusion and self.update_pi_u_now:
            pseudo = self.link_inclusion_prior_strength * self.n_genes
            self.pi_u = float(np.clip((np.sum(L['incl']) + self.link_inclusion_prior_mean * pseudo) / max(np.sum(q) + pseudo, 1e-300), 1e-6, 1.0 - 1e-6))
        if self.estimate_link_scale:
            # Excluded links (g_k = ghat_k / lambda) vote for the current lambda; included links, whose g_k is pinned by the eQTL, calibrate it
            w = gse**2 + self.link_bias_variance
            self.link_scale = np.sum(ghat * Egh / w) / np.sum(Eg2h / w)
        if self.estimate_link_bias_variance:
            # Maximize the expected link log likelihood over log(tau_u^2) by golden-section search, checking tau_u^2 = 0 too
            def f(tau2_u):
                w = gse**2 + tau2_u
                return np.sum(-0.5 * np.log(w) - (ghat**2 - 2.0 * self.link_scale * ghat * Egh + self.link_scale**2 * Eg2h) / (2.0 * w))
            lo = -30.0
            hi = np.log(max(np.var(ghat), 1e-8))  # cap: tau_u^2 <= var(ghat)
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

    def link_terms(self, L):
        # Link part of the ELBO under the current hyperparameters: expected log likelihood of the observed links minus the KL of
        # the link posteriors. L: dict of (stacked or per-gene) link posterior arrays plus ghat, gse (see link_state_dict).
        ghat = L['ghat']
        gse = L['gse']
        lam = self.link_scale
        w = gse**2 + self.link_bias_variance
        if not self.link_inclusion:
            q, m, v = L['q'], L['m'], L['v']
            Eg = q * m
            Eg2 = q * (m**2 + v)
            loglik = np.sum(-0.5 * np.log(2.0 * np.pi * w) - (ghat**2 - 2.0 * lam * ghat * Eg + lam**2 * Eg2) / (2.0 * w))
            return loglik - np.sum(spike_slab_kl(q, m, v, self.pi, self.tau2_g))
        # Model D: sum over the three cells of r [E log p(ghat | g) + log p(z, u) - log r - KL(q(h | cell) || p(h))]
        r11 = L['incl']
        r10 = L['r10']
        r0 = np.maximum(1.0 - r10 - r11, 0.0)
        m1, v1, m_h, v_h = L['m'], L['v'], L['m0'], L['v0']
        def elogN(m, v):
            return -0.5 * np.log(2.0 * np.pi * w) - (ghat**2 - 2.0 * lam * ghat * m + lam**2 * (m**2 + v)) / (2.0 * w)
        def kl(m, v):
            return 0.5 * ((v + m**2) / self.tau2_g - 1.0 - np.log(v / self.tau2_g))
        def cell(r, value):
            return np.where(r > 0.0, r * (value - np.log(r + 1e-300)), 0.0)
        total = cell(r0, elogN(0.0, 0.0) + np.log(1.0 - self.pi))
        total = total + cell(r10, elogN(m_h, v_h) + np.log(self.pi) + np.log(1.0 - self.pi_u) - kl(m_h, v_h))
        total = total + cell(r11, elogN(m1, v1) + np.log(self.pi) + np.log(self.pi_u) - kl(m1, v1))
        return np.sum(total)

    def log_pi_u_prior(self):
        # log Beta(pi_u; mean c n + 1, (1 - mean) c n + 1) up to a constant; 0 without the hyperprior
        if not self.link_inclusion or self.link_inclusion_prior_strength <= 0.0:
            return 0.0
        pseudo = self.link_inclusion_prior_strength * self.n_genes
        return self.link_inclusion_prior_mean * pseudo * np.log(self.pi_u) + (1.0 - self.link_inclusion_prior_mean) * pseudo * np.log(1.0 - self.pi_u)

    def link_elbo_terms(self):
        return self.link_terms(self.stacked_link_quantities())

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
        res['mediated_pip'] = 1.0 - np.prod(1.0 - state['incl'][:, None] * res['caqtl_pip'], axis=0)
        res['eqtl_pip'] = 1.0 - (1.0 - res['direct_pip']) * (1.0 - res['mediated_pip'])
        res['link_prob'] = state['q'].copy()                   # posterior probability the link is non-zero
        res['link_mean'] = state['Eg'].copy()                  # posterior mean of g_k
        res['link_slab_mean'] = state['m'].copy()              # slab moments of the mediating cell (z_k = 1, and u_k = 1 with link inclusion)
        res['link_slab_var'] = state['v'].copy()
        res['link_inclusion_prob'] = state['incl'].copy()      # posterior probability the link mediates the eQTL, P(z_k u_k = 1) (= link_prob without link inclusion)
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
