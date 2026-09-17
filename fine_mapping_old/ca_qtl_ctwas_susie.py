import numpy as np
from scipy.stats import norm
from susie_rss import SUSIE_RSS


class CAQTL_CTWAS_SUSIE(object):
    """
    Model J (see caqtl_mediated_fine_mapping_model_proposal_ctwas_groups.md): cTWAS on peaks.

    Per gene: each linked peak's caQTL is fine-mapped once with SuSiE-RSS and frozen; the peak's genetic component
    X E[b_k], standardized to unit variance (s_k^2 = E[b_k' R b_k]), joins the p variants as an element of one SuSiE over
    the eQTL z-scores. Every element has a free effect beta_l ~ N(0, V_l). The prior weight of an element depends on its
    group: variants; peaks binned by hurdle significance |t_k| = |ghat_k / se_k| and, where the sign is meaningful, by
    whether the hurdle's sign agrees with the marginal eQTL x caQTL direction at the peak's caQTL lead. Group weights are
    shared across genes and learned by EM (expected causal elements per element in the group). No lambda, no tie to the
    hurdle size, no refit of the caQTL. Z-score mode only.
    """

    def __init__(self, L_eqtl=10, L_caqtl=5, t_bin_edges=(2.0, 4.0), t_bin_quantiles=0, sign_groups=True, sign_min_z=2.0, sign_gate=False, group_pseudo_elements=10.0, max_rounds=20, group_tol=1e-3, min_group_weight=1e-4, estimate_prior_variance=True, verbose=True):
        self.L_eqtl = L_eqtl
        self.L_caqtl = L_caqtl
        self.t_bin_edges = tuple(sorted(float(x) for x in t_bin_edges))
        self.t_bin_quantiles = int(t_bin_quantiles)   # > 0: replace the edges by quantiles of |t| over all links so the bins hold equal numbers of links
        self.sign_groups = sign_groups
        self.sign_min_z = sign_min_z
        self.sign_gate = sign_gate                          # impose the sign: discordant peaks get prior weight 0 (instead of a learned group weight)
        self.group_pseudo_elements = group_pseudo_elements  # EM shrinkage: each group's rate is pooled with this many pseudo-elements at the variant rate (stabilizes tiny groups)
        self.max_rounds = max_rounds
        self.group_tol = group_tol
        self.min_group_weight = min_group_weight
        self.estimate_prior_variance = estimate_prior_variance
        self.verbose = verbose
        self.build_groups()
        self.n_rounds = None
        self.converged = None
        self.gene_results = None
        self.elbo = None

    @staticmethod
    def _fmt(x):
        return str(int(x)) if float(x).is_integer() else ('%.3g' % x)

    def build_groups(self):
        bins = ['t<' + self._fmt(self.t_bin_edges[0])] + ['t' + self._fmt(a) + '-' + self._fmt(b) for a, b in zip(self.t_bin_edges[:-1], self.t_bin_edges[1:])] + ['t>' + self._fmt(self.t_bin_edges[-1])]
        self.bin_names = bins
        self.groups = ['variant']
        for i, b in enumerate(bins):
            if i == 0 or not self.sign_groups:
                self.groups.append('peak_' + b)
            else:
                for st in ['concordant', 'discordant', 'undetermined']:
                    self.groups.append('peak_' + b + '_' + st)
        self.group_index = {g: i for i, g in enumerate(self.groups)}
        self.pi_group = np.ones(len(self.groups))

    ###############################
    # Grouping
    ###############################
    def peak_group(self, t, d, ze_lead):
        # t: hurdle z of the link; d: marginal direction sign(z_E) sign(z_A) at the caQTL lead (0 if undetermined); ze_lead: eQTL z at the lead
        b = int(np.searchsorted(np.array(self.t_bin_edges), abs(t), side='right'))   # 0 = weakest bin
        name = 'peak_' + self.bin_names[b]
        if b == 0 or not self.sign_groups:
            return name
        if abs(ze_lead) < self.sign_min_z or d == 0:
            return name + '_undetermined'
        return name + ('_concordant' if np.sign(t) == d else '_discordant')

    ###############################
    # Fit
    ###############################
    def fit(self, input_data, n_eqtl=None, n_caqtl=None):
        if n_eqtl is not None or n_caqtl is not None:
            raise ValueError('Model J runs in z-score mode only (sample sizes must be None)')
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
            imputed = np.atleast_2d(np.asarray(input_data['caqtl_imputed_masks'][g])).astype(bool) if ('caqtl_imputed_masks' in input_data and input_data['caqtl_imputed_masks'][g] is not None and K > 0) else np.zeros((K, len(zE)), dtype=bool)
            self.data.append({'ld_file': input_data['ld_file_names'][g], 'p': len(zE), 'K': K, 'zE': zE, 'zA': zA, 'ghat': ghat, 'gse': gse, 'imputed': imputed})
        # Bin edges from quantiles of |t| if requested, then group every peak from the inputs alone (no LD needed) and report the counts
        all_t = np.concatenate([d['ghat'] / d['gse'] for d in self.data if d['K'] > 0]) if any(d['K'] > 0 for d in self.data) else np.zeros(0)
        if self.t_bin_quantiles > 1 and len(all_t) > 0:
            qs = np.quantile(np.abs(all_t), np.linspace(0.0, 1.0, self.t_bin_quantiles + 1)[1:-1])
            self.t_bin_edges = tuple(float(x) for x in qs)
            self.build_groups()
        for d in self.data:
            d['peak_group'] = []
            d['direction'] = np.zeros(d['K'])
            d['lead'] = np.zeros(d['K'], dtype=int)
            for k in range(d['K']):
                typed = ~d['imputed'][k]
                lead = int(np.argmax(np.where(typed, np.abs(d['zA'][k]), -np.inf))) if np.sum(typed) > 0 else int(np.argmax(np.abs(d['zA'][k])))
                dk = float(np.sign(d['zE'][lead]) * np.sign(d['zA'][k, lead]))
                d['lead'][k] = lead
                d['direction'][k] = dk
                d['peak_group'].append(self.peak_group(d['ghat'][k] / d['gse'][k], dk, d['zE'][lead]))
        if self.verbose:
            counts = {g: 0 for g in self.groups[1:]}
            for d in self.data:
                for name in d['peak_group']:
                    counts[name] += 1
            print('hurdle |t| bin edges: ' + ', '.join('%.3g' % e for e in self.t_bin_edges) + ' | variants: ' + str(int(sum(d['p'] for d in self.data))) + ' | peaks per group: ' + ', '.join(g + ' ' + str(counts[g]) for g in self.groups[1:]), flush=True)
        self.state = [None] * n_genes
        self.elbo = []
        self.converged = False
        for rnd in range(self.max_rounds):
            counts = np.zeros(len(self.groups))
            sizes = np.zeros(len(self.groups))
            total_elbo = 0.0
            for g in range(n_genes):
                data = self.data[g]
                R = np.load(data['ld_file']).astype(float)
                if not np.all(np.isfinite(R)):
                    raise ValueError('LD matrix contains non-finite values: ' + data['ld_file'])
                if self.state[g] is None:
                    self.state[g] = self.initialize_gene(data, R)
                st = self.state[g]
                self.fit_gene(data, st, R)
                del R
                total_elbo = total_elbo + st['elbo']
                # group masses over active effects
                active = st['V'] > 1e-9
                if np.sum(active) > 0:
                    mass = np.sum(st['alpha'][active], axis=0)
                else:
                    mass = np.zeros(st['alpha'].shape[1])
                for e, gi in enumerate(st['group_of_element']):
                    counts[gi] += mass[e]
                    sizes[gi] += 1.0
            pi_old = self.pi_group.copy()
            # variant rate first (no shrinkage), then every peak group pooled with pseudo-elements at the variant rate
            pi_v = counts[0] / max(sizes[0], 1.0) if sizes[0] > 0 else pi_old[0]
            kappa = self.group_pseudo_elements
            pi_new = np.array([(counts[i] + kappa * pi_v) / (sizes[i] + kappa) if i > 0 else pi_v for i in range(len(self.groups))])
            self.pi_group = np.maximum(pi_new, self.min_group_weight)
            self.group_counts = counts
            self.group_sizes = sizes
            self.elbo.append(total_elbo)
            change = np.max(np.abs(self.pi_group - pi_old) / np.maximum(np.abs(pi_old), 1e-12))
            if self.verbose:
                enr = self.pi_group / self.pi_group[0]
                print('round ' + str(rnd + 1) + ': sum of gene ELBOs ' + str(round(total_elbo, 3)) + ' | enrichment over variants: ' + ', '.join(self.groups[i] + ' %.2f' % enr[i] for i in range(1, len(self.groups))) + ' | max weight change %.2e' % change, flush=True)
            if rnd > 0 and change < self.group_tol:
                self.converged = True
                break
        self.n_rounds = rnd + 1
        self.elbo = np.array(self.elbo)
        if not self.converged:
            print('warning: group weights did not converge in ' + str(self.max_rounds) + ' rounds')
        self.gene_results = [self.summarize_gene(self.data[g], self.state[g]) for g in range(n_genes)]
        return self

    ###############################
    # Per gene
    ###############################
    def initialize_gene(self, data, R):
        # caQTL-only fits, frozen; standardized peak predictors; groups
        p = data['p']
        K = data['K']
        st = {}
        st['Eb'] = np.zeros((K, p))
        st['caqtl_pip'] = np.zeros((K, p))
        st['s'] = np.ones(K)
        st['REb'] = np.zeros((K, p))
        st['group_of_element'] = [0] * p
        st['peak_group'] = list(data['peak_group'])
        st['direction'] = data['direction'].copy()
        st['lead'] = data['lead'].copy()
        for k in range(K):
            fit = SUSIE_RSS(L=self.L_caqtl, estimate_prior_variance=self.estimate_prior_variance).fit(data['zA'][k], np.ones(p), R, None)
            Eb = np.sum(fit.alpha * fit.mu, axis=0)
            Eb_l = fit.alpha * fit.mu
            REb_l = np.dot(Eb_l, R)
            bRb = np.dot(Eb, np.dot(R, Eb)) - np.sum(Eb_l * REb_l) + np.sum(fit.alpha * fit.mu2)    # E[b' R b]
            st['Eb'][k] = Eb
            st['REb'][k] = np.dot(R, Eb)
            st['s'][k] = np.sqrt(max(bRb, 1e-12))
            st['caqtl_pip'][k] = fit.pip
            st['group_of_element'].append(self.group_index[st['peak_group'][k]])
        st['group_of_element'] = np.array(st['group_of_element'])
        # sign gate (optional): 0 for a discordant peak whose directions are both clear, else 1
        st['gate'] = np.concatenate([np.ones(p), np.array([0.0 if name.endswith('_discordant') else 1.0 for name in st['peak_group']])]) if K > 0 else np.ones(p)
        # augmented sufficient statistics that do not depend on the group weights
        st['z_aug'] = np.concatenate([data['zE'], np.dot(st['Eb'], data['zE']) / st['s']]) if K > 0 else data['zE'].copy()
        st['alpha'] = np.ones((self.L_eqtl, p + K)) / (p + K)
        st['mu'] = np.zeros((self.L_eqtl, p + K))
        st['V'] = np.zeros(self.L_eqtl)
        st['elbo'] = -np.inf
        return st

    def augmented_ld(self, st, R):
        p = st['Eb'].shape[1]
        K = st['Eb'].shape[0]
        if K == 0:
            return R
        cross = st['REb'] / st['s'][:, None]                        # K x p: corr(peak k, variant j)
        pk = np.dot(st['Eb'], st['REb'].T) / np.outer(st['s'], st['s'])   # K x K
        np.fill_diagonal(pk, 1.0)
        R_aug = np.zeros((p + K, p + K))
        R_aug[:p, :p] = R
        R_aug[p:, :p] = cross
        R_aug[:p, p:] = cross.T
        R_aug[p:, p:] = pk
        return R_aug

    def fit_gene(self, data, st, R):
        def K_gate(state):
            return 'gate' in state and np.any(state['gate'] == 0.0)
        R_aug = self.augmented_ld(st, R)
        weights = self.pi_group[st['group_of_element']].copy()
        if self.sign_gate and K_gate(st):
            weights[st['gate'] == 0.0] = 0.0
        fit = SUSIE_RSS(L=self.L_eqtl, estimate_prior_variance=self.estimate_prior_variance).fit(st['z_aug'], np.ones(len(st['z_aug'])), R_aug, None, prior_weights=weights)
        st['alpha'] = fit.alpha
        st['mu'] = fit.mu
        st['V'] = fit.V
        st['pip_aug'] = fit.pip
        st['elbo'] = float(fit.elbo[-1])
        st['n_iter'] = fit.n_iter
        st['susie_converged'] = fit.converged
        st['credible_sets'] = fit.credible_sets

    ###############################
    # Summaries
    ###############################
    def summarize_gene(self, data, st):
        p = data['p']
        K = data['K']
        res = {}
        active = st['V'] > 1e-9
        res['direct_pip'] = 1.0 - np.prod(1.0 - st['alpha'][active, :p], axis=0) if np.sum(active) > 0 else np.zeros(p)
        Pk = 1.0 - np.prod(1.0 - st['alpha'][active, p:], axis=0) if (K > 0 and np.sum(active) > 0) else np.zeros(K)
        res['peak_selection_prob'] = Pk
        res['peak_selection_mass'] = np.sum(st['alpha'][active, p:], axis=0) if (K > 0 and np.sum(active) > 0) else np.zeros(K)
        res['link_inclusion_prob'] = Pk
        res['link_prob'] = 1.0 - 2.0 * norm.sf(np.abs(data['ghat'] / data['gse'])) if K > 0 else np.zeros(0)   # two-sided significance of the hurdle link (no link model here)
        res['caqtl_pip'] = st['caqtl_pip']
        res['mediated_pip'] = 1.0 - np.prod(1.0 - Pk[:, None] * st['caqtl_pip'], axis=0) if K > 0 else np.zeros(p)
        res['eqtl_pip'] = 1.0 - (1.0 - res['direct_pip']) * (1.0 - res['mediated_pip'])
        Ed = np.sum(st['alpha'][active, :p] * st['mu'][active, :p], axis=0) if np.sum(active) > 0 else np.zeros(p)
        coef = np.sum(st['alpha'][active, p:] * st['mu'][active, p:], axis=0) / st['s'] if (K > 0 and np.sum(active) > 0) else np.zeros(K)   # on the original m_k scale
        res['peak_coefficient'] = coef
        res['link_mean'] = coef
        res['eqtl_posterior_mean'] = Ed + (np.dot(coef, st['Eb']) if K > 0 else 0.0)
        res['peak_group'] = list(st['peak_group'])
        res['direction'] = st['direction']
        res['gate'] = st['gate'][p:] if K > 0 else np.zeros(0)
        res['elbo'] = st['elbo']
        res['init'] = 'ctwas'
        res['sigma2_E'] = 1.0
        res['sigma2_A'] = np.ones(K)
        return res
