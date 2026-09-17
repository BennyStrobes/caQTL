import numpy as np
from itertools import combinations
from susie_rss import SUSIE_RSS
from susie_rss_mixture import SUSIE_RSS_MIXTURE


def fit_link_spike_slab(ghat, gse, pi_init=0.5, max_iter=1000, tol=1e-9, min_pi=1e-4):
    """
    Empirical Bayes spike-and-slab on the hurdle links: g_k ~ (1 - pi) d0 + pi N(0, tau2), ghat_k ~ N(g_k, se_k^2), EM over all
    links. Returns pi, tau2, and per link P(g_k != 0 | ghat_k), the conditional posterior mean and variance of g_k given non-zero.
    """
    ghat = np.asarray(ghat, dtype=float)
    se2 = np.asarray(gse, dtype=float)**2
    if len(ghat) == 0:
        return pi_init, 1.0, np.zeros(0), np.zeros(0), np.zeros(0)
    tau2 = max(float(np.mean(ghat**2) - np.mean(se2)), 0.01 * float(np.mean(se2)))
    pi = pi_init

    def posterior(pi, tau2):
        log_bf = 0.5 * np.log(se2 / (tau2 + se2)) + 0.5 * (ghat**2 / se2) * (tau2 / (tau2 + se2))
        log_r = np.log(pi) + log_bf - np.logaddexp(np.log(1.0 - pi), np.log(pi) + log_bf)
        return np.exp(log_r), ghat * tau2 / (tau2 + se2), tau2 * se2 / (tau2 + se2)

    for it in range(max_iter):
        r, pm, pv = posterior(pi, tau2)
        pi_new = float(np.clip(np.mean(r), min_pi, 1.0 - min_pi))
        tau2_new = max(float(np.sum(r * (pv + pm**2)) / max(float(np.sum(r)), 1e-12)), 1e-8)
        done = abs(pi_new - pi) < tol and abs(tau2_new - tau2) < tol * max(tau2, 1.0)
        pi, tau2 = pi_new, tau2_new
        if done:
            break
    r, pm, pv = posterior(pi, tau2)
    return pi, tau2, r, pm, pv


class CAQTL_PREDICTED_MEAN_SUSIE(object):
    """
    Model L (caqtl_mediated_fine_mapping_model_proposal_predicted_mean.md, Section 2): the causal eQTL effect of a variant is a
    spike at zero plus a Gaussian centred on the effect predicted through chromatin,
        beta_j = gamma_j (lambda m_j + e_j),  m_j = sum_k b_kj g_k,  e_j ~ N(0, sigma2),
    with b_kj the variant's effect on peak k (spike-and-slab, posterior from the caQTL z-scores computed with SuSiE) and g_k the
    peak's hurdle link to the gene (spike-and-slab, posterior from the hurdle estimate, closed form, hyperparameters by
    empirical Bayes over all links). The caQTL and link posteriors are marginalized inside the eQTL fit: per variant the slab is
    a mixture over which of its links are active, with one component per subset of active links (weight = product of caQTL PIP
    x P(link non-zero) over the subset), mean lambda x the predicted effect of the subset and variance sigma2_linked + lambda^2 x
    the posterior variance of the prediction; the empty subset has mean 0 and variance sigma2_unlinked. lambda (one per hurdle
    significance bin, Gaussian prior N(0, lambda_prior_sd^2)), sigma2_unlinked and sigma2_linked are shared across genes and
    learned by EM over rounds of gene fits. Variants only: nothing is added to the design matrix, and no feedback into the
    caQTL or link posteriors. Z-score mode only.
    """

    def __init__(self, L_eqtl=10, L_caqtl=5, t_bin_edges=(2.0, 4.0), lambda_prior_sd=10.0, lambda_init=0.0, sigma2_init=25.0, single_variance=False, sigma2_floor=0.1, min_caqtl_pip=1e-3, max_peaks_per_variant=3, max_rounds=30, tol=1e-3, switch_off_effects=True, verbose=True):
        self.L_eqtl = L_eqtl
        self.L_caqtl = L_caqtl
        self.t_bin_edges = tuple(sorted(float(x) for x in t_bin_edges)) if (t_bin_edges is not None and len(t_bin_edges) > 0) else ()
        if len(self.t_bin_edges) == 0:
            self.bin_names = ['all']
        else:
            self.bin_names = ['t<' + self._fmt(self.t_bin_edges[0])] + ['t' + self._fmt(a) + '-' + self._fmt(b) for a, b in zip(self.t_bin_edges[:-1], self.t_bin_edges[1:])] + ['t>' + self._fmt(self.t_bin_edges[-1])]
        self.n_bins = len(self.bin_names)
        self.lambda_prior_sd = float(lambda_prior_sd)
        self.lam = np.full(self.n_bins, float(lambda_init))
        self.sigma2_unlinked = float(sigma2_init)
        self.sigma2_linked = float(sigma2_init)
        self.single_variance = single_variance
        self.sigma2_floor = float(sigma2_floor)
        self.min_caqtl_pip = float(min_caqtl_pip)              # a (peak, variant) pair enters a variant's mixture only if caQTL PIP x P(link non-zero) is at least this
        self.max_peaks_per_variant = int(max_peaks_per_variant)  # cap on the peaks in a variant's mixture (2^cap components); the strongest pairs are kept
        self.max_rounds = max_rounds
        self.tol = tol
        self.switch_off_effects = switch_off_effects
        self.verbose = verbose
        self.n_rounds = None
        self.converged = None
        self.gene_results = None
        self.elbo = None

    @staticmethod
    def _fmt(x):
        return str(int(x)) if float(x).is_integer() else ('%.3g' % x)

    def link_bin(self, t):
        if self.n_bins == 1:
            return 0
        return int(np.searchsorted(np.array(self.t_bin_edges), abs(t), side='right'))

    ###############################
    # Fit
    ###############################
    def fit(self, input_data, n_eqtl=None, n_caqtl=None):
        if n_eqtl is not None or n_caqtl is not None:
            raise ValueError('Model L runs in z-score mode only (sample sizes must be None)')
        n_genes = len(input_data['ld_file_names'])
        self.n_genes = n_genes
        self.data = []
        for g in range(n_genes):
            zE = np.asarray(input_data['eqtl_effects'][g], dtype=float) / np.asarray(input_data['eqtl_ses'][g], dtype=float)
            bA = np.atleast_2d(np.asarray(input_data['caqtl_effects'][g], dtype=float))
            sA = np.atleast_2d(np.asarray(input_data['caqtl_ses'][g], dtype=float))
            K = bA.shape[0] if bA.size > 0 else 0
            zA = bA / sA if K > 0 else np.zeros((0, len(zE)))
            ghat = np.asarray(input_data['peak_gene_effects'][g], dtype=float)
            gse = np.asarray(input_data['peak_gene_ses'][g], dtype=float)
            self.data.append({'ld_file': input_data['ld_file_names'][g], 'p': len(zE), 'K': K, 'zE': zE, 'zA': zA, 'ghat': ghat, 'gse': gse})
        # Link layer: empirical Bayes spike-and-slab over every link, then per-link posterior and significance bin
        all_ghat = np.concatenate([d['ghat'] for d in self.data]) if n_genes > 0 else np.zeros(0)
        all_gse = np.concatenate([d['gse'] for d in self.data]) if n_genes > 0 else np.zeros(0)
        self.link_pi, self.link_tau2, r, pm, pv = fit_link_spike_slab(all_ghat, all_gse)
        pos = 0
        for d in self.data:
            K = d['K']
            d['link_prob'] = r[pos:pos + K].copy()
            d['link_mean'] = pm[pos:pos + K].copy()
            d['link_var'] = pv[pos:pos + K].copy()
            d['link_bin'] = np.array([self.link_bin(t) for t in (d['ghat'] / d['gse'])], dtype=int) if K > 0 else np.zeros(0, dtype=int)
            pos += K
        self.n_links = int(len(all_ghat))
        if self.verbose:
            bin_counts = np.zeros(self.n_bins, dtype=int)
            for d in self.data:
                for b in d['link_bin']:
                    bin_counts[b] += 1
            print('link prior (empirical Bayes over ' + str(self.n_links) + ' links): P(link non-zero) ' + '%.3f' % self.link_pi + ', slab variance ' + '%.4g' % self.link_tau2 + ' | links with posterior P(non-zero) > 0.5: ' + str(int(np.sum(r > 0.5))) + ', > 0.9: ' + str(int(np.sum(r > 0.9))) + ' | links per hurdle bin: ' + ', '.join(self.bin_names[b] + ' ' + str(bin_counts[b]) for b in range(self.n_bins)) + ' | variants: ' + str(int(sum(d['p'] for d in self.data))), flush=True)
        self.state = [None] * n_genes
        self.elbo = []
        self.converged = False
        B = self.n_bins
        for rnd in range(self.max_rounds):
            A = np.zeros((B, B))
            rhs = np.zeros(B)
            num0 = 0.0
            den0 = 0.0
            num1 = 0.0
            den1 = 0.0
            total_elbo = 0.0
            n_pred = 0
            for g in range(n_genes):
                data = self.data[g]
                R = np.load(data['ld_file']).astype(float)
                if not np.all(np.isfinite(R)):
                    raise ValueError('LD matrix contains non-finite values: ' + data['ld_file'])
                if self.state[g] is None:
                    self.state[g] = self.initialize_gene(data, R)
                st = self.state[g]
                fit = self.fit_gene(data, st, R)
                del R
                total_elbo = total_elbo + st['elbo']
                n_pred += len(st['pred_idx'])
                # accumulate the sufficient statistics of the hyperparameter updates from the active effects, then drop the full component arrays
                active = np.where(fit.V > 1e-9)[0]
                idx = st['pred_idx']
                ne = st['ne_sub']
                for l in active:
                    W = fit.alpha[l][:, None] * fit.omega[l]                      # p x C: P(effect l is element j and component c)
                    second = fit.comp_var[l] + fit.comp_mean[l]**2
                    num0 += float(np.sum(W[:, 0] * second[:, 0]))
                    den0 += float(np.sum(W[:, 0]))
                    if len(idx) > 0:
                        Wp = W[idx]
                        Wn = np.where(ne, Wp / st['s2_sub'], 0.0)
                        A += np.einsum('jc,jcb,jcd->bd', Wn, st['x_sub'], st['x_sub'])
                        rhs += np.einsum('jc,jcb,jc->b', Wn, st['x_sub'], fit.comp_mean[l][idx])
                        resid2 = fit.comp_var[l][idx] + (fit.comp_mean[l][idx] - st['m_sub'])**2 - (st['s2_sub'] - self.sigma2_linked)   # minus lambda^2 x prediction variance
                        num1 += float(np.sum(np.where(ne, Wp * resid2, 0.0)))
                        den1 += float(np.sum(np.where(ne, Wp, 0.0)))
                del fit
            lam_old = self.lam.copy()
            s0_old = self.sigma2_unlinked
            s1_old = self.sigma2_linked
            self.lam = np.linalg.solve(A + np.eye(B) / self.lambda_prior_sd**2, rhs)
            # slab variances: pooled second moments of the causal effects in each class; left unchanged when the class holds
            # almost no causal mass (the update would be noise)
            min_mass = 0.1
            if self.single_variance:
                if den0 + den1 > min_mass:
                    s = max((num0 + num1) / (den0 + den1), self.sigma2_floor)
                    self.sigma2_unlinked = s
                    self.sigma2_linked = s
            else:
                if den0 > min_mass:
                    self.sigma2_unlinked = max(num0 / den0, self.sigma2_floor)
                if den1 > min_mass:
                    self.sigma2_linked = max(num1 / den1, self.sigma2_floor)
            self.expected_causal_unlinked = den0
            self.expected_causal_linked = den1
            self.elbo.append(total_elbo)
            change = max(float(np.max(np.abs(self.lam - lam_old) / (0.1 + np.abs(lam_old)))), abs(self.sigma2_unlinked - s0_old) / s0_old, abs(self.sigma2_linked - s1_old) / s1_old)
            if self.verbose:
                print('round ' + str(rnd + 1) + ': sum of gene ELBOs ' + str(round(total_elbo, 3)) + ' | lambda per bin: ' + ', '.join(self.bin_names[b] + ' %.3f' % self.lam[b] for b in range(B)) + ' | sigma2 unlinked %.3g, linked %.3g' % (self.sigma2_unlinked, self.sigma2_linked) + ' | expected causal effects: %.2f without prediction, %.2f with' % (den0, den1) + ' | variants with a prediction: ' + str(n_pred) + ' | max change %.2e' % change, flush=True)
            if rnd > 0 and change < self.tol:
                self.converged = True
                break
        self.n_rounds = rnd + 1
        self.elbo = np.array(self.elbo)
        if not self.converged:
            print('warning: hyperparameters did not converge in ' + str(self.max_rounds) + ' rounds')
        self.gene_results = [self.summarize_gene(self.data[g], self.state[g]) for g in range(n_genes)]
        return self

    ###############################
    # Per gene
    ###############################
    def initialize_gene(self, data, R):
        # caQTL posteriors per peak (SuSiE on the caQTL z-scores; conditional moments of b_kj given it is non-zero), then the
        # per-variant mixture components, built once. Kept in compact form (only variants with a prediction).
        p = data['p']
        K = data['K']
        st = {'caqtl_pip': np.zeros((K, p))}
        bbar = np.zeros((K, p))
        b2 = np.zeros((K, p))
        for k in range(K):
            fit = SUSIE_RSS(L=self.L_caqtl, estimate_prior_variance=True).fit(data['zA'][k], np.ones(p), R, None)
            active = fit.V > 1e-9
            pip = fit.pip
            Eb = np.sum(fit.alpha[active] * fit.mu[active], axis=0) if np.sum(active) > 0 else np.zeros(p)
            Eb2 = np.sum(fit.alpha[active] * fit.mu2[active], axis=0) if np.sum(active) > 0 else np.zeros(p)
            ok = pip > 0.0
            st['caqtl_pip'][k] = pip
            bbar[k, ok] = Eb[ok] / pip[ok]
            b2[k, ok] = np.maximum(Eb2[ok] / pip[ok], (Eb[ok] / pip[ok])**2)
        self.build_components(data, st, bbar, b2)
        st['elbo'] = -np.inf
        return st

    def build_components(self, data, st, bbar, b2):
        # For each variant: the linked peaks whose caQTL PIP x P(link non-zero) passes the threshold (strongest max_peaks_per_variant
        # kept); one mixture component per subset S of them, weight prod_{k in S} q_kj prod_{k not in S} (1 - q_kj); per hurdle bin b
        # the predicted effect x_b = sum_{k in S, bin b} bbar_kj gbar_k and its posterior variance v_b (product of independent posteriors).
        # Component 0 is always the empty subset.
        p = data['p']
        K = data['K']
        B = self.n_bins
        q = st['caqtl_pip'] * data['link_prob'][:, None] if K > 0 else np.zeros((0, p))
        g2 = data['link_var'] + data['link_mean']**2 if K > 0 else np.zeros(0)
        pred_idx = []
        rows = []
        for j in range(p):
            ks = np.where(q[:, j] >= self.min_caqtl_pip)[0] if K > 0 else np.zeros(0, dtype=int)
            if len(ks) == 0:
                continue
            if len(ks) > self.max_peaks_per_variant:
                ks = ks[np.argsort(-q[ks, j])[:self.max_peaks_per_variant]]
            pred_idx.append(j)
            rows.append(ks)
        n_pred = len(pred_idx)
        C = 2 ** (max(len(ks) for ks in rows) if n_pred > 0 else 0)
        w = np.zeros((n_pred, C))
        x = np.zeros((n_pred, C, B))
        v = np.zeros((n_pred, C, B))
        ne = np.zeros((n_pred, C), dtype=bool)
        comp_peaks = {}
        for i, (j, ks) in enumerate(zip(pred_idx, rows)):
            c = 0
            for size in range(len(ks) + 1):
                for S in combinations(list(ks), size):
                    notS = [k for k in ks if k not in S]
                    w[i, c] = float(np.prod(q[list(S), j])) * float(np.prod(1.0 - q[notS, j]))
                    for k in S:
                        b = data['link_bin'][k]
                        mean_kj = bbar[k, j] * data['link_mean'][k]
                        x[i, c, b] += mean_kj
                        v[i, c, b] += max(b2[k, j] * g2[k] - mean_kj**2, 0.0)
                    ne[i, c] = size > 0
                    if size > 0:
                        comp_peaks[(i, c)] = np.array(S, dtype=int)
                    c += 1
            w[i] = w[i] / np.sum(w[i])
        st['pred_idx'] = np.array(pred_idx, dtype=int)
        st['w_sub'] = w
        st['x_sub'] = x
        st['v_sub'] = v
        st['ne_sub'] = ne
        st['comp_peaks'] = comp_peaks
        st['C'] = max(C, 1)

    def fit_gene(self, data, st, R):
        p = data['p']
        C = st['C']
        idx = st['pred_idx']
        # current component means and variances from the shared hyperparameters
        m_sub = np.einsum('jcb,b->jc', st['x_sub'], self.lam) if len(idx) > 0 else np.zeros((0, C))
        s2_sub = np.where(st['ne_sub'], self.sigma2_linked + np.einsum('jcb,b->jc', st['v_sub'], self.lam**2), self.sigma2_unlinked) if len(idx) > 0 else np.zeros((0, C))
        st['m_sub'] = m_sub
        st['s2_sub'] = s2_sub
        w = np.zeros((p, C))
        w[:, 0] = 1.0
        m = np.zeros((p, C))
        s2 = np.full((p, C), self.sigma2_unlinked)
        if len(idx) > 0:
            w[idx] = st['w_sub']
            m[idx] = m_sub
            s2[idx] = s2_sub
        fit = SUSIE_RSS_MIXTURE(L=self.L_eqtl, switch_off_effects=self.switch_off_effects).fit(data['zE'], np.ones(p), R, None, w, m, s2)
        active = fit.V > 1e-9
        st['alpha'] = fit.alpha
        st['mu'] = fit.mu
        st['V'] = fit.V
        st['pip'] = fit.pip
        st['omega0'] = fit.omega[:, :, 0]          # P(component is the empty subset | effect l is element j)
        st['elbo'] = float(fit.elbo[-1])
        st['n_iter'] = fit.n_iter
        st['susie_converged'] = fit.converged
        st['credible_sets'] = fit.credible_sets
        # posterior mass of each peak's link carrying a causal effect, per effect
        K = data['K']
        peak_mass = np.zeros((self.L_eqtl, K))
        for (i, c), ks in st['comp_peaks'].items():
            j = idx[i]
            for l in np.where(active)[0]:
                peak_mass[l, ks] += fit.alpha[l, j] * fit.omega[l, j, c]
        st['peak_mass'] = peak_mass
        # prior expectation of the predicted effect per variant (for diagnostics)
        pred = np.zeros(p)
        if len(idx) > 0:
            pred[idx] = np.sum(st['w_sub'] * m_sub, axis=1)
        st['predicted_mean'] = pred
        return fit

    ###############################
    # Summaries
    ###############################
    def summarize_gene(self, data, st):
        p = data['p']
        K = data['K']
        res = {}
        active = st['V'] > 1e-9
        alpha = st['alpha'][active]
        w0 = st['omega0'][active]
        res['eqtl_pip'] = 1.0 - np.prod(1.0 - alpha, axis=0) if np.sum(active) > 0 else np.zeros(p)
        res['direct_pip'] = 1.0 - np.prod(1.0 - alpha * w0, axis=0) if np.sum(active) > 0 else np.zeros(p)           # causal, effect from the no-prediction component
        res['mediated_pip'] = 1.0 - np.prod(1.0 - alpha * (1.0 - w0), axis=0) if np.sum(active) > 0 else np.zeros(p)  # causal, effect from a chromatin-prediction component
        res['eqtl_posterior_mean'] = np.sum(alpha * st['mu'][active], axis=0) if np.sum(active) > 0 else np.zeros(p)
        res['eqtl_predicted_mean'] = st['predicted_mean']
        pw = np.zeros(p)
        if len(st['pred_idx']) > 0:
            pw[st['pred_idx']] = 1.0 - st['w_sub'][:, 0]
        res['prediction_weight'] = pw                                                                   # prior P(variant has a chromatin prediction)
        mass = st['peak_mass'][active] if (K > 0 and np.sum(active) > 0) else np.zeros((0, K))
        res['peak_selection_prob'] = 1.0 - np.prod(1.0 - mass, axis=0) if mass.shape[0] > 0 else np.zeros(K)   # P(some causal effect acts through peak k)
        res['peak_selection_mass'] = np.sum(mass, axis=0) if mass.shape[0] > 0 else np.zeros(K)
        res['link_inclusion_prob'] = res['peak_selection_prob']
        res['link_prob'] = data['link_prob']
        res['link_mean'] = data['link_mean']
        res['caqtl_pip'] = st['caqtl_pip']
        res['peak_group'] = ['peak_' + self.bin_names[b] for b in data['link_bin']]
        res['elbo'] = st['elbo']
        res['init'] = 'predicted_mean'
        return res
