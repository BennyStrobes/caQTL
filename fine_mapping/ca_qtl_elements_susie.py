import numpy as np
from susie_rss import SUSIE_RSS
from ca_qtl_mediated_susie import CAQTL_MEDIATED_SUSIE, ser_log_bayes_factors, spike_slab_kl


def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


class CAQTL_ELEMENTS_SUSIE(CAQTL_MEDIATED_SUSIE):
    """
    Model E (see caqtl_mediated_fine_mapping_model_proposal_elements.md): the eQTL is one SuSiE over an augmented pool of
    elements, the p cis variants plus, for each linked peak whose peak-gene link is non-zero, the peak's genetic component
    m_k = X b_k. Direct and mediated explanations compete inside the same categorical posterior.

    Per gene:
        A_k    = X b_k + e_k                              caQTL of peak k (SuSiE single effects, jointly fit as in Model A)
        E      = sum_l beta_l w_{gamma_l} + e             eQTL: single effect l selects one element gamma_l
                 element j <= p: variant j, predictor x_j, free effect beta_l ~ N(0, V_l)
                 element p + k:  peak k, predictor m_k, effect FIXED at the link value g_k (tied), or free (free_peak_effects)
        g_k    = z_k h_k, z_k ~ Bern(pi), h_k ~ N(0, tau_g^2);  ghat_k ~ N(lambda g_k, se_k^2 + tau_u^2)
        P(gamma_l = j) prop. to 1,  P(gamma_l = p + k) prop. to rho_p * P(z_k = 1):  a peak is in the pool only if its link is non-zero
    rho_p (the enrichment of real-link peak elements over variants), pi, tau_g^2 and tau_u^2 are shared across genes (EM);
    lambda is fixed. Single start: caQTL-only fits per peak, links from the hurdle alone, eQTL single effects empty.
    """

    def __init__(self, L_eqtl=10, L_caqtl=5, link_prior_prob=0.2, link_prior_variance=None, estimate_link_prior=True, link_scale_init=1.0, estimate_link_bias_variance=True, link_bias_variance_init=0.0, peak_weight_ratio_init=1.0, estimate_peak_weight_ratio=True, free_peak_effects=False, pool_gate='running', estimate_link_scale=False, link_scale_prior_sd=None, estimate_prior_variance=True, estimate_residual_variance=True, zscore_residual_variance=False, n_inner_iter=5, max_outer_iter=100, tol=1e-3, hyper_tol=1e-3, verbose=True):
        super().__init__(L_eqtl=L_eqtl, L_caqtl=L_caqtl, link_prior_prob=link_prior_prob, link_prior_variance=link_prior_variance, estimate_link_prior=estimate_link_prior, estimate_link_scale=False, link_scale_init=link_scale_init, estimate_link_bias_variance=estimate_link_bias_variance, link_bias_variance_init=link_bias_variance_init, estimate_prior_variance=estimate_prior_variance, estimate_residual_variance=estimate_residual_variance, zscore_residual_variance=zscore_residual_variance, link_inclusion=False, n_inner_iter=n_inner_iter, max_outer_iter=max_outer_iter, tol=tol, hyper_tol=hyper_tol, verbose=verbose)
        self.peak_weight_ratio_init = peak_weight_ratio_init     # rho_p = omega_p / omega_v: prior weight of a real-link peak element relative to a variant
        self.estimate_peak_weight_ratio = estimate_peak_weight_ratio
        self.free_peak_effects = free_peak_effects               # E-free: a selected peak gets a free effect (symmetric) instead of g_k; the link then only gates the pool
        # Pool gate: the prior weight of peak k is rho_p x qw_k. 'hurdle': qw_k = P(z_k = 1 | hurdle alone), fixed at initialization.
        # 'running': qw_k is refreshed once per outer iteration from the full link posterior q_k (which pools the eQTL evidence about g_k), and
        # held fixed within the iteration, so every gene pass and M-step is a coordinate ascent on a fixed-weight objective; the objective is
        # redefined at each refresh and the printed ELBO can step down there by the refresh amount (reported as pool_refresh in the log).
        if pool_gate not in ('running', 'hurdle'):
            raise ValueError("pool_gate must be 'running' or 'hurdle'")
        self.pool_gate = pool_gate
        # lambda as a random variable: variational factor q(lambda) = N(m, v) with prior N(link_scale_init, link_scale_prior_sd^2)
        # (prior sd defaults to a quarter of link_scale_init). The update uses links weighted by their selection probability:
        # the hurdle term only fixes lambda x g_k, so an unselected link cannot inform lambda (and its spike-slab shrinkage would
        # make it vote for a drifting lambda, Model A's runaway); selected peaks have g_k pinned by the expression data and calibrate it.
        # E[lambda] and E[lambda^2] enter the link posteriors, the link ELBO terms and the tau_u^2 step. The posterior sd reflects the
        # selected peaks only, and still ignores the lambda-g_k coupling the mean-field factorization drops, so read it as a lower bound.
        # lambda and tau_u^2 are not jointly identified: a wrong lambda is absorbed as extra link noise instead of being corrected,
        # so estimating lambda requires tau_u^2 fixed (0, or a value from a fixed-lambda run via link_bias_variance_init).
        if estimate_link_scale and estimate_link_bias_variance:
            raise ValueError('estimate_link_scale requires the link bias variance to be fixed (estimate_link_bias_variance=False)')
        self.estimate_link_scale = estimate_link_scale
        self.link_scale_prior_sd = link_scale_prior_sd       # None: 0.25 x link_scale_init (E[lambda^2] enters every link's hurdle precision, so a vague prior with nothing selected would inflate them)
        self.link_scale_var = 0.0
        self.rho_p = peak_weight_ratio_init

    ###############################
    # Fitting across genes (single start)
    ###############################
    def fit(self, input_data, n_eqtl=None, n_caqtl=None):
        self.zscore_mode = (n_eqtl is None) or (n_caqtl is None)
        if self.zscore_mode and not ((n_eqtl is None) and (n_caqtl is None)):
            raise ValueError('n_eqtl and n_caqtl must both be given or both be None')
        self.n_eqtl = 2 if self.zscore_mode else n_eqtl
        self.n_caqtl = 2 if self.zscore_mode else n_caqtl
        self.n_eqtl_fit = None if self.zscore_mode else n_eqtl
        self.n_caqtl_fit = None if self.zscore_mode else n_caqtl
        self.zscore_resid = self.zscore_mode and self.zscore_residual_variance and self.estimate_residual_variance
        n_genes = len(input_data['ld_file_names'])
        self.n_genes = n_genes
        self.data = [self.prepare_gene_data(input_data, g) for g in range(n_genes)]
        self.state = [None] * n_genes
        self.gene_elbo_increment = np.zeros(n_genes)

        self.pi = self.link_prior_prob
        self.tau2_g = self.initial_link_prior_variance() if self.link_prior_variance is None else self.link_prior_variance
        self.link_scale = self.link_scale_init
        self.link_scale_var = 0.0
        self.link_scale_prior_sd_used = self.link_scale_prior_sd if self.link_scale_prior_sd is not None else 0.25 * float(self.link_scale_init)
        self.link_bias_variance = float(self.link_bias_variance_init)
        self.pi_u = 1.0
        self.update_pi_u_now = False
        self.rho_p = float(self.peak_weight_ratio_init)
        if self.verbose:
            print('elements model: lambda ' + (('estimated as a random variable from the selected peaks, init ' + str(self.link_scale) + ', prior sd ' + str(round(self.link_scale_prior_sd_used, 4))) if self.estimate_link_scale else ('fixed at ' + str(self.link_scale))) + ', peak weight ratio init ' + str(self.rho_p) + (' (estimated)' if self.estimate_peak_weight_ratio else ' (fixed)') + ', peak effects ' + ('free' if self.free_peak_effects else 'tied to the link value') + ', pool gate ' + self.pool_gate, flush=True)

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
                    self.state[g] = self.initialize_gene(self.data[g], R)
                    genetic_elbo = genetic_elbo + self.update_gene(self.data[g], self.state[g], R)
                else:
                    elbo_before = self.state[g]['elbo']
                    genetic_elbo = genetic_elbo + self.update_gene(self.data[g], self.state[g], R)
                    self.gene_elbo_increment[g] = self.state[g]['elbo'] - elbo_before
                if not np.isfinite(self.state[g]['elbo']) or not np.all(np.isfinite(self.state[g]['q'])):
                    raise ValueError('non-finite ELBO or link posterior for gene with LD file ' + self.data[g]['ld_file'])
                del R
            # Pool-weight refresh (running gate): qw <- q once per outer iteration; the gene ELBOs are re-evaluated under the new weights
            weight_change = 0.0
            pool_refresh = 0.0
            if self.pool_gate == 'running':
                for g in range(n_genes):
                    st = self.state[g]
                    if len(st['q']) == 0:
                        continue
                    weight_change = max(weight_change, float(np.max(np.abs(st['q'] - st['qw']))))
                    st['qw'] = st['q'].copy()
                    new_term = self.prior_term(self.data[g], st)
                    pool_refresh = pool_refresh + new_term - st['prior_term']
                    st['prior_term'] = new_term
                genetic_elbo = genetic_elbo + pool_refresh
            hyper_before = np.array([self.pi, self.tau2_g, self.link_bias_variance, self.rho_p, self.link_scale])
            self.update_link_hyperparameters()
            self.update_peak_weight_ratio()
            # Re-evaluate the categorical prior term of every gene under the new rho_p (an M-step on that term, so this is an ascent)
            for g in range(n_genes):
                new_term = self.prior_term(self.data[g], self.state[g])
                genetic_elbo = genetic_elbo + new_term - self.state[g]['prior_term']
                self.state[g]['prior_term'] = new_term
            hyper_after = np.array([self.pi, self.tau2_g, self.link_bias_variance, self.rho_p, self.link_scale])
            hyper_change = np.max(np.abs(hyper_after - hyper_before) / np.maximum(np.abs(hyper_before), 1e-12))
            self.elbo.append(genetic_elbo + self.link_elbo_terms())
            if self.verbose:
                print('outer iter ' + str(outer_iter + 1) + ': ELBO ' + str(round(self.elbo[-1], 3)) + ', pi ' + str(round(self.pi, 4)) + ', tau2_g ' + '%.3e' % self.tau2_g + ', lambda ' + str(round(self.link_scale, 4)) + (' (sd ' + '%.3g' % np.sqrt(self.link_scale_var) + ')' if self.estimate_link_scale else '') + ', tau2_u ' + '%.3e' % self.link_bias_variance + ', peak_weight_ratio ' + str(round(self.rho_p, 4)) + (', pool_refresh ' + '%.3f' % pool_refresh + ', max_dqw ' + '%.1e' % weight_change if self.pool_gate == 'running' else ''), flush=True)
            # With the running gate the objective is redefined at each refresh, so the ELBO test uses the absolute change and the
            # pool weights must also have stopped moving
            if outer_iter > 0 and abs(self.elbo[-1] - self.elbo[-2]) < self.tol * n_genes and hyper_change < self.hyper_tol and weight_change < self.hyper_tol:
                self.converged = True
                break
        self.n_iter = outer_iter + 1
        self.elbo = np.array(self.elbo)
        if not self.converged:
            print('warning: did not converge in ' + str(self.max_outer_iter) + ' outer iterations')
        self.gene_results = [self.summarize_gene(self.data[g], self.state[g]) for g in range(n_genes)]
        return self

    ###############################
    # Per-gene initialization: caQTL-only fits, links from the hurdle alone, eQTL single effects empty
    ###############################
    def initialize_gene(self, data, R):
        p = data['p']
        K = data['K']
        LA = self.L_caqtl
        LE = self.L_eqtl
        state = {}
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
        # Links from the hurdle alone (Model A's spike-slab posterior with no eQTL evidence)
        for key in self.link_state_keys:
            state[key] = np.zeros(K)
        for k in range(K):
            self.update_link_posterior(data, state, k, 0.0, 0.0)
        state['qw'] = state['q'].copy()      # pool-weight link probability: starts at P(z_k = 1 | hurdle alone); refreshed from q_k each outer iteration with the running gate, fixed with the hurdle gate
        # eQTL single effects: empty. d_alpha over the augmented pool (p variants then K peaks), variant effect moments, peak free-effect moments
        var_E = data['EtE'] / max(data['nE'] - 1.0, 1.0)
        state['d_alpha'] = np.tile(self.prior_weights(data, state), (LE, 1))
        state['d_mu'] = np.zeros((LE, p))
        state['d_mu2'] = np.zeros((LE, p))
        state['d_post_var'] = np.zeros((LE, p))
        state['pk_mu'] = np.zeros((LE, K))
        state['pk_mu2'] = np.zeros((LE, K))
        state['pk_post_var'] = np.zeros((LE, K))
        state['d_V'] = np.ones(LE) * 0.2 * var_E
        state['d_on'] = np.ones(LE, dtype=bool)        # effect switched on (its evidence over the whole pool beats the null); off effects contribute nothing
        state['sigma2_E'] = var_E
        state['peak_mass'] = 0.0
        state['n_on'] = 0
        state['elbo'] = -np.inf
        state['init'] = 'empty'
        return state

    def prior_weights(self, data, state):
        # Normalized prior weights over the augmented pool: 1 per variant, rho_p * qw_k per peak (qw: see pool_gate)
        w = np.concatenate([np.ones(data['p']), self.rho_p * state['qw']])
        return w / np.sum(w)

    def link_scale_sq(self):
        return self.link_scale**2 + self.link_scale_var     # E[lambda^2]

    def update_link_posterior(self, data, state, k, S_E, P_E):
        # Spike-slab posterior of g_k pooling the hurdle term (with E[lambda], E[lambda^2]) and the eQTL-side evidence
        ghat = data['ghat'][k]
        w = data['gse'][k]**2 + self.link_bias_variance
        S_h = self.link_scale * ghat / w
        P_h = self.link_scale_sq() / w
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

    def link_terms(self, L):
        # Expected log likelihood of the observed links (with E[lambda], E[lambda^2]) minus the KL of the link posteriors;
        # plus, when lambda is a random variable, minus KL(q(lambda) || prior)
        ghat = L['ghat']
        gse = L['gse']
        w = gse**2 + self.link_bias_variance
        q, m, v = L['q'], L['m'], L['v']
        Eg = q * m
        Eg2 = q * (m**2 + v)
        loglik = np.sum(-0.5 * np.log(2.0 * np.pi * w) - (ghat**2 - 2.0 * self.link_scale * ghat * Eg + self.link_scale_sq() * Eg2) / (2.0 * w))
        out = loglik - np.sum(spike_slab_kl(q, m, v, self.pi, self.tau2_g))
        if self.estimate_link_scale and self.link_scale_var > 0.0:
            s0 = self.link_scale_prior_sd_used**2
            out = out - 0.5 * ((self.link_scale_var + (self.link_scale - self.link_scale_init)**2) / s0 - 1.0 - np.log(self.link_scale_var / s0))
        return out

    def update_link_hyperparameters(self):
        # One EM step for pi, tau_g^2, the lambda factor (selection-weighted) and tau_u^2
        L = self.stacked_link_quantities()
        if len(L['q']) == 0:
            return
        q = L['q']
        Egh = L['Egh']
        Eg2h = L['Eg2h']
        ghat = L['ghat']
        gse = L['gse']
        if self.estimate_link_prior:
            self.pi = float(np.clip(np.mean(q), 1e-6, 1.0 - 1e-6))
            self.tau2_g = max(np.sum(Eg2h) / max(np.sum(q), 1e-300), 1e-8)
        if self.estimate_link_scale:
            # q(lambda) = N(m, v): precision = prior precision + sum_k s_k E[g_k^2] / w_k, mean = (prior mean x prior precision + sum_k s_k ghat_k E[g_k] / w_k) / precision,
            # with s_k the probability that some on single effect selects peak k
            sel = np.concatenate([self.selection_probability(st) for st in self.state])
            w = gse**2 + self.link_bias_variance
            prior_prec = 1.0 / self.link_scale_prior_sd_used**2
            prec = prior_prec + np.sum(sel * Eg2h / w)
            self.link_scale = float((prior_prec * self.link_scale_init + np.sum(sel * ghat * Egh / w)) / prec)
            self.link_scale_var = float(1.0 / prec)
        if self.estimate_link_bias_variance:
            lam2 = self.link_scale_sq()
            def f(tau2_u):
                w = gse**2 + tau2_u
                return np.sum(-0.5 * np.log(w) - (ghat**2 - 2.0 * self.link_scale * ghat * Egh + lam2 * Eg2h) / (2.0 * w))
            lo = -30.0
            hi = np.log(max(np.var(ghat), 1e-8))
            golden = (np.sqrt(5.0) - 1.0) / 2.0
            x1 = hi - golden * (hi - lo)
            x2 = lo + golden * (hi - lo)
            f1 = f(np.exp(x1))
            f2 = f(np.exp(x2))
            for _ in range(50):
                if f1 < f2:
                    lo, x1, f1 = x1, x2, f2
                    x2 = lo + golden * (hi - lo)
                    f2 = f(np.exp(x2))
                else:
                    hi, x2, f2 = x2, x1, f1
                    x1 = hi - golden * (hi - lo)
                    f1 = f(np.exp(x1))
            tau2_u = np.exp(0.5 * (lo + hi))
            self.link_bias_variance = tau2_u if f(tau2_u) > f(0.0) else 0.0

    def selection_probability(self, state):
        # P(some on single effect selects peak k), length K
        K = len(state['q'])
        if K == 0:
            return np.zeros(0)
        p = state['d_alpha'].shape[1] - K
        on = state['d_on']
        if np.sum(on) == 0:
            return np.zeros(K)
        return 1.0 - np.prod(1.0 - state['d_alpha'][on, p:], axis=0)

    ###############################
    # Per-gene coordinate ascent
    ###############################
    def update_gene(self, data, state, R):
        p = data['p']
        K = data['K']
        LA = self.L_caqtl
        LE = self.L_eqtl
        cE = data['cE']
        cA = data['cA']
        lpw_A = data['log_prior_weights']
        tied = not self.free_peak_effects

        Eb_l = state['b_alpha'] * state['b_mu']                       # K x LA x p
        REb_l = np.einsum('klp,pq->klq', Eb_l, R)
        Eb = np.sum(Eb_l, axis=1)                                     # K x p
        REb = np.sum(REb_l, axis=1)
        bRb = np.array([np.dot(Eb[k], REb[k]) - np.sum(Eb_l[k] * REb_l[k]) + np.sum(state['b_alpha'][k] * state['b_mu2'][k]) for k in range(K)])

        def peak_mean(l):
            # expected effect of each peak element in single effect l (given selection): g_k (tied) or the free effect
            return state['Eg'] if tied else state['pk_mu'][l]

        def peak_mean2(l):
            return state['Eg2'] if tied else state['pk_mu2'][l]

        def contrib(l):
            # R x expected contribution of single effect l (variants + peaks), length p; zero for an effect that is off
            if not state['d_on'][l]:
                return np.zeros(p)
            a_pk = state['d_alpha'][l, p:]
            out = np.dot(R, state['d_alpha'][l, :p] * state['d_mu'][l])
            if K > 0:
                out = out + np.dot(a_pk * peak_mean(l), REb)
            return out

        Rcontrib = np.array([contrib(l) for l in range(LE)]) if LE > 0 else np.zeros((0, p))
        Rtotal = np.sum(Rcontrib, axis=0)
        SE_lk = np.zeros((LE, K))      # eQTL-side score of peak k in effect l (for the link update)
        PE_k = cE * bRb / state['sigma2_E'] if K > 0 else np.zeros(0)

        for inner_iter in range(self.n_inner_iter):
            # (1) eQTL single effects: SER over p variants + K real-link peaks
            lpw = np.log(self.prior_weights(data, state) + 1e-300)
            for l in range(LE):
                Xtr = data['XtE'] - cE * (Rtotal - Rcontrib[l])
                S_var = Xtr / state['sigma2_E']
                P_var = cE / state['sigma2_E']
                betahat = S_var / P_var
                shat2 = np.ones(p) / P_var
                if K > 0:
                    S_pk = np.dot(Eb, Xtr) / state['sigma2_E']
                    SE_lk[l] = S_pk
                    if tied:
                        lbf_pk = S_pk * state['Eg'] - 0.5 * PE_k * state['Eg2']     # expected log-likelihood ratio of adding g_k m_k
                    else:
                        betahat_pk = S_pk / PE_k
                        shat2_pk = 1.0 / PE_k
                else:
                    lbf_pk = np.zeros(0)

                def total_loglik(V):
                    lbf_v = ser_log_bayes_factors(V, betahat, shat2) if V > 0.0 else np.zeros(p)
                    if K > 0 and not tied:
                        lbf_p = ser_log_bayes_factors(V, betahat_pk, shat2_pk) if V > 0.0 else np.zeros(K)
                    else:
                        lbf_p = lbf_pk
                    return logsumexp(np.concatenate([lbf_v, lbf_p]) + lpw), lbf_v, lbf_p

                V = state['d_V'][l]
                if self.estimate_prior_variance:
                    lo, hi = -30.0, 15.0
                    golden = (np.sqrt(5.0) - 1.0) / 2.0
                    x1 = hi - golden * (hi - lo)
                    x2 = lo + golden * (hi - lo)
                    f1 = total_loglik(np.exp(x1))[0]
                    f2 = total_loglik(np.exp(x2))[0]
                    for _ in range(50):
                        if f1 < f2:
                            lo, x1, f1 = x1, x2, f2
                            x2 = lo + golden * (hi - lo)
                            f2 = total_loglik(np.exp(x2))[0]
                        else:
                            hi, x2, f2 = x2, x1, f1
                            x1 = hi - golden * (hi - lo)
                            f1 = total_loglik(np.exp(x1))[0]
                    V_new = np.exp(0.5 * (lo + hi))
                    if total_loglik(V_new)[0] < total_loglik(V)[0]:
                        V_new = V
                    if total_loglik(0.0)[0] >= total_loglik(V_new)[0]:
                        V_new = 0.0
                    V = V_new
                ll, lbf_v, lbf_p = total_loglik(V)
                on = ll > 0.0                       # SuSiE's rule generalized to the augmented pool: keep the effect only if it beats the null
                state['d_on'][l] = on
                if not on:
                    V = 0.0
                    lbf_v = np.zeros(p)
                    lbf_p = np.zeros(K)
                log_post = np.concatenate([lbf_v, lbf_p]) + lpw
                alpha = np.exp(log_post - np.max(log_post))
                alpha = alpha / np.sum(alpha)
                state['d_alpha'][l] = alpha
                state['d_V'][l] = V
                if V > 0.0:
                    post_var = 1.0 / (1.0 / V + P_var)
                    state['d_post_var'][l] = post_var
                    state['d_mu'][l] = post_var * S_var
                    if K > 0 and not tied:
                        pv = 1.0 / (1.0 / V + PE_k)
                        state['pk_post_var'][l] = pv
                        state['pk_mu'][l] = pv * S_pk
                        state['pk_mu2'][l] = pv + state['pk_mu'][l]**2
                else:
                    state['d_post_var'][l] = 0.0
                    state['d_mu'][l] = 0.0
                    if K > 0 and not tied:
                        state['pk_post_var'][l] = 0.0
                        state['pk_mu'][l] = 0.0
                        state['pk_mu2'][l] = 0.0
                state['d_mu2'][l] = state['d_post_var'][l] + state['d_mu'][l]**2
                Rtotal = Rtotal - Rcontrib[l]
                Rcontrib[l] = contrib(l)
                Rtotal = Rtotal + Rcontrib[l]

            # (2) Links: hurdle term plus selection-weighted eQTL evidence (tied); hurdle only (free).
            # The eQTL-side scores are recomputed against the residuals as they stand now (the ones recorded during the SER step
            # are stale once later effects have moved), so this update is exact given the current state.
            if K > 0:
                if tied:
                    for l in range(LE):
                        Xtr = data['XtE'] - cE * (Rtotal - Rcontrib[l])
                        SE_lk[l] = np.dot(Eb, Xtr) / state['sigma2_E']
                for k in range(K):
                    if tied:
                        a = state['d_alpha'][:, p + k] * state['d_on']
                        self.update_link_posterior(data, state, k, float(np.sum(a * SE_lk[:, k])), float(np.sum(a) * PE_k[k]))
                    else:
                        self.update_link_posterior(data, state, k, 0.0, 0.0)
                if tied:
                    Rcontrib = np.array([contrib(l) for l in range(LE)])
                    Rtotal = np.sum(Rcontrib, axis=0)

            # (3) caQTL single effects: pool the caQTL residual with the eQTL residual weighted by the peak's expected eQTL coefficient
            for k in range(K):
                a_k = state['d_alpha'][:, p + k] * state['d_on']                       # selection probability per (on) effect
                Eg_e = float(np.sum(a_k * np.array([peak_mean(l)[k] for l in range(LE)])))   # E[coefficient of b_k in the eQTL]
                Eg2_e = float(np.sum(a_k * np.array([peak_mean2(l)[k] for l in range(LE)])))
                resid_other = data['XtE'] - cE * (Rtotal - Eg_e * REb[k])              # eQTL residual excluding peak k entirely
                for l in range(LA):
                    score_A = (data['XtA'][k] - cA * (REb[k] - REb_l[k, l])) / state['sigma2_A'][k]
                    score_E = (Eg_e * resid_other - Eg2_e * cE * (REb[k] - REb_l[k, l])) / state['sigma2_E']
                    S = score_A + score_E
                    P = np.ones(p) * (cA / state['sigma2_A'][k] + Eg2_e * cE / state['sigma2_E'])
                    from ca_qtl_mediated_susie import single_effect_regression
                    alpha, mu, mu2, post_var, lbf, V = single_effect_regression(S, P, state['b_V'][k, l], lpw_A, self.estimate_prior_variance)
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
                bRb[k] = np.dot(Eb[k], REb[k]) - np.sum(Eb_l[k] * REb_l[k]) + np.sum(state['b_alpha'][k] * state['b_mu2'][k])
                PE_k[k] = cE * bRb[k] / state['sigma2_E']
            if K > 0:
                Rcontrib = np.array([contrib(l) for l in range(LE)])
                Rtotal = np.sum(Rcontrib, axis=0)

            # (4) Residual variances and the gene ELBO (link terms are added globally)
            erss_A = data['AtA'] - 2.0 * np.sum(Eb * data['XtA'], axis=1) + cA * bRb
            Econtrib = np.array([(state['d_alpha'][l, :p] * state['d_mu'][l] + (np.dot(state['d_alpha'][l, p:] * peak_mean(l), Eb) if K > 0 else 0.0)) if state['d_on'][l] else np.zeros(p) for l in range(LE)])
            Ebeta = np.sum(Econtrib, axis=0)
            within = 0.0
            for l in range(LE):
                if not state['d_on'][l]:
                    continue
                within = within + np.sum(state['d_alpha'][l, :p] * state['d_mu2'][l])
                if K > 0:
                    within = within + np.sum(state['d_alpha'][l, p:] * peak_mean2(l) * bRb)
            EbetaRbeta = np.dot(Ebeta, Rtotal) - np.sum(Econtrib * Rcontrib) + within
            erss_E = data['EtE'] - 2.0 * np.dot(Ebeta, data['XtE']) + cE * EbetaRbeta
            if self.estimate_residual_variance and (not self.zscore_mode or self.zscore_resid):
                state['sigma2_A'] = erss_A / data['nA']
                state['sigma2_E'] = erss_E / data['nE']
                if np.any(state['sigma2_A'] <= 0.0) or state['sigma2_E'] <= 0.0:
                    raise ValueError('estimated residual variance is negative: summary statistics and LD are inconsistent')
                PE_k = cE * bRb / state['sigma2_E'] if K > 0 else np.zeros(0)
            elbo = np.sum(-0.5 * data['nA'] * np.log(2.0 * np.pi * state['sigma2_A']) - 0.5 * erss_A / state['sigma2_A'])
            elbo = elbo - 0.5 * data['nE'] * np.log(2.0 * np.pi * state['sigma2_E']) - 0.5 * erss_E / state['sigma2_E']
            from ca_qtl_mediated_susie import ser_kl
            for k in range(K):
                for l in range(LA):
                    elbo = elbo - ser_kl(state['b_alpha'][k, l], state['b_mu'][k, l], state['b_post_var'][k, l], state['b_V'][k, l], lpw_A)
            for l in range(LE):
                elbo = elbo - self.effect_kl(state, l, lpw, p, K, tied)
            converged = (elbo - state['elbo']) < self.tol
            state['elbo'] = elbo
            if converged:
                break
        state['peak_mass'] = float(np.sum(state['d_alpha'][state['d_on'], p:])) if K > 0 else 0.0
        state['n_on'] = int(np.sum(state['d_on']))
        state['prior_term'] = self.prior_term(data, state)
        return state['elbo']

    def prior_term(self, data, state):
        # sum over on effects of E_q[log prior weight of the selected element]; its change under a new rho_p is the change in the gene ELBO
        if state['n_on'] == 0:
            return 0.0
        lpw = np.log(self.prior_weights(data, state) + 1e-300)
        return float(np.sum(state['d_alpha'][state['d_on']] * lpw[None, :]))

    def effect_kl(self, state, l, lpw, p, K, tied):
        # KL of single effect l: categorical over the pool, plus the Gaussian effect term for the elements that own an effect
        if not state['d_on'][l]:
            return 0.0
        alpha = state['d_alpha'][l]
        kl = np.sum(alpha * (np.log(alpha + 1e-300) - lpw))
        V = state['d_V'][l]
        if V > 0.0:
            kl = kl + np.sum(alpha[:p] * 0.5 * ((state['d_post_var'][l] + state['d_mu'][l]**2) / V - 1.0 - np.log(state['d_post_var'][l] / V)))
            if K > 0 and not tied:
                kl = kl + np.sum(alpha[p:] * 0.5 * ((state['pk_post_var'][l] + state['pk_mu'][l]**2) / V - 1.0 - np.log(state['pk_post_var'][l] / V)))
        return kl

    ###############################
    # Shared hyperparameters
    ###############################
    def update_peak_weight_ratio(self):
        # Maximize sum_genes [M_g log rho - L log(p_g + rho Q_g)] over log rho, where M_g is the posterior mass on peak
        # elements over the L single effects and Q_g = sum_k q_k the expected number of real-link peaks (the pool size that rho weights)
        if not self.estimate_peak_weight_ratio:
            return
        M = np.array([s['peak_mass'] for s in self.state])
        Q = np.array([np.sum(s['qw']) for s in self.state])
        P = np.array([d['p'] for d in self.data], dtype=float)
        N_on = np.array([s['n_on'] for s in self.state], dtype=float)
        if np.sum(Q) <= 0.0 or np.sum(N_on) <= 0.0:
            return
        def f(log_rho):
            rho = np.exp(log_rho)
            return np.sum(M * log_rho - N_on * np.log(P + rho * Q))
        # rho_p is searched on [1e-3, 1e2]: a real-link peak element at most 100x a variant. Beyond that the prior concentrates on a
        # handful of peaks per gene and a chance fit can switch an effect on (seen with free peak effects on a synthetic set with no direct effects).
        lo, hi = np.log(1e-3), np.log(1e2)
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = hi - golden * (hi - lo)
        x2 = lo + golden * (hi - lo)
        f1, f2 = f(x1), f(x2)
        for _ in range(60):
            if f1 < f2:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + golden * (hi - lo)
                f2 = f(x2)
            else:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - golden * (hi - lo)
                f1 = f(x1)
        self.rho_p = float(np.exp(0.5 * (lo + hi)))

    ###############################
    # Summaries
    ###############################
    def summarize_gene(self, data, state, prior_tol=1e-9):
        K = data['K']
        p = data['p']
        LE = self.L_eqtl
        res = {}
        res['caqtl_pip'] = np.zeros((K, p))
        for k in range(K):
            keep = state['b_V'][k] > prior_tol
            if np.sum(keep) > 0:
                res['caqtl_pip'][k] = 1.0 - np.prod(1.0 - state['b_alpha'][k][keep], axis=0)
        keep = state['d_on']
        res['direct_pip'] = 1.0 - np.prod(1.0 - state['d_alpha'][keep, :p], axis=0) if np.sum(keep) > 0 else np.zeros(p)
        sel = self.selection_probability(state)                                                # P(some on single effect selects peak k)
        res['peak_selection_prob'] = sel
        res['peak_selection_mass'] = np.sum(state['d_alpha'][state['d_on'], p:], axis=0) if (K > 0 and np.sum(state['d_on']) > 0) else np.zeros(K)   # sum over on effects; values well above 1 = the peak is stacked (implied effect too small at this lambda)
        res['link_prob'] = state['q'].copy()
        res['link_inclusion_prob'] = sel * state['q']                                          # P_k = P(selected and link non-zero)
        res['mediated_pip'] = 1.0 - np.prod(1.0 - res['link_inclusion_prob'][:, None] * res['caqtl_pip'], axis=0) if K > 0 else np.zeros(p)
        res['eqtl_pip'] = 1.0 - (1.0 - res['direct_pip']) * (1.0 - res['mediated_pip'])
        res['link_mean'] = state['Eg'].copy()
        res['link_slab_mean'] = state['m'].copy()
        res['link_slab_var'] = state['v'].copy()
        Eb = np.sum(state['b_alpha'] * state['b_mu'], axis=1)
        Ed = np.sum(state['d_alpha'][:, :p] * state['d_mu'] * state['d_on'][:, None], axis=0)
        if K > 0:
            coef = np.array([np.sum(state['d_alpha'][:, p + k] * state['d_on'] * (state['Eg'][k] if not self.free_peak_effects else state['pk_mu'][:, k])) for k in range(K)])
            res['eqtl_posterior_mean'] = Ed + np.dot(coef, Eb)
            res['caqtl_posterior_mean'] = Eb
        else:
            res['eqtl_posterior_mean'] = Ed
            res['caqtl_posterior_mean'] = Eb
        res['direct_posterior_mean'] = Ed
        res['sigma2_A'] = state['sigma2_A'].copy()
        res['sigma2_E'] = state['sigma2_E']
        res['elbo'] = state['elbo']
        res['init'] = state['init']
        return res
