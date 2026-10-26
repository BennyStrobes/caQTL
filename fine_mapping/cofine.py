import numpy as np
import sys
from numba import njit


###############################################################################
# CoFine: eQTL fine-mapping with chromatin-mediated directional priors (grant Aim 3B, equations 6 and 7).
#
# Model (per gene g; p cis variants, K linked peaks; peaks linked to several genes are treated separately for each gene):
#   delta_kj ~ (1 - pi_delta) delta_0 + pi_delta N(0, sigma2_delta)        causal caQTL effect of variant j on peak k
#   gamma_k  ~ (1 - pi_gamma) delta_0 + pi_gamma N(0, sigma2_gamma)        causal effect of peak k on the gene
#   m_j = sum_k delta_kj gamma_k                                            chromatin-mediated effect of variant j on the gene
#   H_j = 1[exists k: delta_kj != 0 and gamma_k != 0]                      variant j has a chromatin-mediated path to the gene
#   beta_j | H_j = 0 ~ (1 - pi0) delta_0 + pi0 N(0, sigma2_0)                                         (equation 6)
#   beta_j | H_j = 1 ~ (1 - pi1) delta_0 + pi1 N(lambda m_j, tau2)                                    (equation 7)
# Shared across genes: pi_delta, sigma2_delta, pi_gamma, sigma2_gamma, pi0, sigma2_0, pi1, lambda, tau2 (and c2, below).
#
# Likelihoods. eQTL and caQTL marginal effects (standardized genotype; one standard error per gene or peak, s) are transformed
# to the low-rank eigen-space of the LD (SBayesRC-style; as in eb_coloc/multi_causal_variant/real_data/mcvc_tempered.py):
#   w = Lambda^{-1/2} U' b_hat = Q beta + eps,  Q = Lambda^{1/2} U',  eps ~ N(0, resid_var I_k)
# with one residual variance per gene (eQTL) and per peak (caQTL), initialized at s^2. Peak-gene links:
#   gamma_hat_k ~ N(gamma_k, c2 se_k^2)  (diagonal; marginal link statistics)
# c2 = 1, or (learn_link_se_inflation) one shared inflation of the squared link standard errors for anti-conservative link statistics,
# with a scaled-inverse-chi-square(link_se_inflation_prior_df, 1) prior. Most tested peak-gene pairs are null (gamma_k = 0), so c2 is
# identified by the spread of their z-scores (a model-based genomic-control factor for the links).
#
# Gibbs sampler, one sweep per gene:
#   1. gamma_k for each peak, with gamma_k integrated out of its slab: the link statistic and, through equation 7, the current eQTL
#      effects of every variant with delta_kj != 0 are both Gaussian in gamma_k, so the class probabilities and the slab draw are exact.
#   2. delta_k. (all variants of peak k) by tempered Gibbs (Zanella & Roberts 2019, as in mcvc_tempered.py). When gamma_k != 0 the
#      prior of delta_kj also carries p(beta_j | H_j, m_j): if beta_j != 0 this is Gaussian in delta_kj (a pseudo-observation
#      (beta_j - lambda m_j^{-k}) / (lambda gamma_k) with variance tau2 / (lambda gamma_k)^2), so it folds into the slab of delta_kj.
#   2b. Path swaps (Metropolis-Hastings): an active path (delta_kj != 0, gamma_k != 0, beta_j != 0) and its eQTL effect move together
#      to a variant j' in high LD with j, (delta_kj', beta_j') = sign(R_jj') (delta_kj, beta_j). With small tau2, delta_kj and beta_j
#      hold each other in place (each raises the other's prior by orders of magnitude), so blocks 2 and 3 alone rarely move a path
#      between variants in LD and the PIPs and P(H_j = 1) become overconfident.
#   3. beta (all variants) by tempered Gibbs, with the per-variant prior of equations 6 and 7 given the current m and H.
#   Each tempered block runs from the block's current state and then hands on one state drawn from its visited states with
#   probability proportional to their importance weights (weighted reservoir sampling), so the blocks pass approximately exact
#   conditional draws to each other and the hyper-parameter updates can use the sampled state directly.
#   eQTL PIPs are the importance-weighted, Rao-Blackwellized conditional inclusion probabilities of block 3 averaged over the
#   retained sweeps; P(H_j = 1) and posterior means are averages of the handed-on states.
# Hyper-parameters (after each sweep over all genes): Beta draws for the pi's, scaled-inverse-chi-square draws for the variances,
# a normal draw for lambda given tau2 (regression of the active beta_j on m_j among variants with H_j = 1), then tau2 given lambda.
###############################################################################


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True)
def _log_expit(x):
    # log(1 / (1 + exp(-x))) without underflow or cancellation
    if x >= 0.0:
        return -np.log1p(np.exp(-x))
    return x - np.log1p(np.exp(x))


@njit(cache=True)
def _logaddexp(a, b):
    m = max(a, b)
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.exp(a - m) + np.exp(b - m))


_SELECTION_BLOCK_SIZE = 32  # sites per partial sum of the tempered selection weights


@njit(cache=True)
def _sum_block(blk, p_sel, block_sum):
    # block_sum[blk] = sum of p_sel over the sites of block blk (recomputed from the sites rather than updated by differences, so no
    # rounding error accumulates: the weights span hundreds of orders of magnitude)
    start = blk * _SELECTION_BLOCK_SIZE
    end = min(start + _SELECTION_BLOCK_SIZE, p_sel.shape[0])
    s = 0.0
    for k in range(start, end):
        s += p_sel[k]
    block_sum[blk] = s


@njit(cache=True)
def _rebuild_p_sel(log_p_sel, p_sel, ref, block_sum):
    # Re-express the selection weights relative to the largest one: p_sel = exp(log_p_sel - ref), and recompute their block sums
    ref[0] = np.max(log_p_sel)
    for k in range(log_p_sel.shape[0]):
        p_sel[k] = np.exp(log_p_sel[k] - ref[0])
    for blk in range(block_sum.shape[0]):
        _sum_block(blk, p_sel, block_sum)


@njit(cache=True)
def _update_site(a, dot, beta, z, q_norm_sq, V, total_var, half_log_var_ratio, log_w_off, log_w_on, slab_mean, P_on, logit_on, log_p_sel, p_sel, ref, block_sum, acc_P_on, last_sum_w, sum_w, log_one_minus_eps, log_eps_half):
    # Full conditional inclusion probability of site a (own effect added back into its residual dot product) -> P_on[a] (and its
    # log-odds logit_on[a]), and its tempered selection weight p_a = g_a(current class) / P_a(current class) = (1 - eps) + (eps / 2) / P_a(current class)
    # -> log_p_sel[a], held as p_sel[a] = exp(log_p_sel[a] - ref[0]) (the caller then recomputes the sum of a's block). Everything is
    # computed from the log-odds: P(current class) can be far below the smallest double (strong effects in LD), and 1 - P_on would
    # round to 0 and corrupt the selection and the weights.
    # V, total_var, half_log_var_ratio: per-site constants of the block (resid_var / |q_a|^2, slab_var + V, 0.5 log(total_var / V)).
    # Before P_on[a] changes, the weights of the sub-states visited since its last change (sum_w - last_sum_w[a]) are credited to acc_P_on[a]
    acc_P_on[a] += P_on[a] * (sum_w - last_sum_w[a])
    last_sum_w[a] = sum_w
    qn = q_norm_sq[a]
    b = (dot[a] + qn * beta[a]) / qn
    # log BF of a N(slab_mean, slab_var) slab: log N(b; slab_mean, slab_var + V) - log N(b; 0, V)
    log_bf = -half_log_var_ratio[a] - 0.5 * (b - slab_mean[a]) ** 2 / total_var[a] + 0.5 * b * b / V[a]
    x = log_w_on[a] + log_bf - log_w_off[a]
    logit_on[a] = x
    # log P_on and log (1 - P_on) (= _log_expit(x) and _log_expit(-x)), sharing log(1 + exp(-|x|))
    log1p_term = np.log1p(np.exp(-abs(x)))
    if x >= 0.0:
        log_p_on = -log1p_term
        log_p_off = -x - log1p_term
    else:
        log_p_on = x - log1p_term
        log_p_off = -log1p_term
    P_on[a] = np.exp(log_p_on)
    if z[a] == 1:
        log_pc = log_p_on
    else:
        log_pc = log_p_off
    lp = _logaddexp(log_one_minus_eps, log_eps_half - log_pc)
    log_p_sel[a] = lp
    if lp - ref[0] > 500.0:
        _rebuild_p_sel(log_p_sel, p_sel, ref, block_sum)
    else:
        p_sel[a] = np.exp(lp - ref[0])


@njit(cache=True)
def _tgs_block_numba(Qt, G, q_norm_sq, nbr_indptr, nbr_indices, resid, beta, z, log_w_off, log_w_on, slab_mean, slab_var, resid_var,
                     n_sub_steps, epsilon, P_on, logit_on, log_p_sel, p_sel, ref, block_sum, block_stamp, site_consts, dot, acc_P_on, last_sum_w, snap_beta, snap_z, stats):
    # Tempered Gibbs over the spike-and-slab effects of one block (n_snps sites sharing one residual vector). Site j is null
    # (log prior weight log_w_off[j]) or has an effect drawn from N(slab_mean[j], slab_var[j]) (log prior weight log_w_on[j]).
    # On entry resid = w - Q beta. On exit (beta, z, resid) are the last sub-state (the caller replaces them with the snapshot);
    # acc_P_on holds the importance-weighted MEAN of the Rao-Blackwellized inclusion probabilities over the visited sub-states and
    # (snap_beta, snap_z) one visited sub-state drawn with probability proportional to its weight.
    # The weights 1 / Z vary over hundreds of orders of magnitude, so they are accumulated relative to a log scale (log_scale) that
    # is raised whenever a weight would exceed exp(200) on it (the squared weights then stay far below the largest double).
    # Bookkeeping per sub-step avoids passes over all sites where it can (the cost is dominated by the conditionals of the
    # neighbours of a moved site): the selection weights are summed in blocks of _SELECTION_BLOCK_SIZE sites (block_sum; only the
    # blocks of updated sites are re-summed, block_stamp marks those done in the current sub-step), and acc_P_on is accumulated
    # lazily, each site being credited P_on times the weights summed since its last change (last_sum_w), when it changes and at the end.
    # site_consts: work array (3 x n_snps) for the per-site constants of _update_site.
    # stats: [0] number of class changes, [1] log of the sum of the weights, [2] effective sample size of the weights.
    # epsilon = 0 gives random-scan Gibbs (uniform site choice, unit weights).
    n_snps = Qt.shape[0]
    n_components = Qt.shape[1]
    n_blocks = block_sum.shape[0]
    one_minus_eps = 1.0 - epsilon
    eps_half = epsilon / 2.0
    log_one_minus_eps = np.log(one_minus_eps) if one_minus_eps > 0.0 else -np.inf
    log_eps_half = np.log(eps_half) if eps_half > 0.0 else -np.inf
    log_n = np.log(n_snps)
    V = site_consts[0]
    total_var = site_consts[1]
    half_log_var_ratio = site_consts[2]
    for j in range(n_snps):
        s1 = 0.0
        for i in range(n_components):
            s1 += Qt[j, i] * resid[i]
        dot[j] = s1
        V[j] = resid_var / q_norm_sq[j]
        total_var[j] = slab_var[j] + V[j]
        half_log_var_ratio[j] = 0.5 * np.log(total_var[j] / V[j])
    for blk in range(n_blocks):
        block_stamp[blk] = -1
    ref[0] = 0.0
    for j in range(n_snps):
        acc_P_on[j] = 0.0
        last_sum_w[j] = 0.0
        _update_site(j, dot, beta, z, q_norm_sq, V, total_var, half_log_var_ratio, log_w_off, log_w_on, slab_mean, P_on, logit_on, log_p_sel, p_sel, ref, block_sum, acc_P_on, last_sum_w, 0.0, log_one_minus_eps, log_eps_half)
    _rebuild_p_sel(log_p_sel, p_sel, ref, block_sum)
    log_scale = -np.inf
    sum_w = 0.0
    sum_w_sq = 0.0
    n_changes = 0
    for step in range(n_sub_steps):
        # Weight of the current state: 1 / Z, Z = mean_j p_j = exp(ref) * sum_j p_sel_j / n
        s = 0.0
        for blk in range(n_blocks):
            s += block_sum[blk]
        if s < 1e-250:
            _rebuild_p_sel(log_p_sel, p_sel, ref, block_sum)
            s = 0.0
            for blk in range(n_blocks):
                s += block_sum[blk]
        log_w = log_n - ref[0] - np.log(s)
        if log_w > log_scale + 200.0:
            if sum_w > 0.0:
                factor = np.exp(log_scale - log_w)
                for j in range(n_snps):
                    acc_P_on[j] *= factor
                    last_sum_w[j] *= factor
                sum_w *= factor
                sum_w_sq *= factor * factor
            log_scale = log_w
        w = np.exp(log_w - log_scale)
        sum_w += w
        sum_w_sq += w * w
        # Weighted reservoir sampling of the state handed on
        if np.random.random() * sum_w < w:
            for j in range(n_snps):
                snap_beta[j] = beta[j]
                snap_z[j] = z[j]

        # Informed choice of the site to update: j with probability proportional to p_j (first its block, then the site within it)
        u = np.random.random() * s
        acc = 0.0
        blk = n_blocks - 1
        for bb in range(n_blocks):
            if u < acc + block_sum[bb]:
                blk = bb
                break
            acc += block_sum[bb]
        u -= acc
        start = blk * _SELECTION_BLOCK_SIZE
        end = min(start + _SELECTION_BLOCK_SIZE, n_snps)
        acc = 0.0
        j = end - 1
        for a in range(start, end):
            acc += p_sel[a]
            if u < acc:
                j = a
                break

        # Tempered draw of the class from g_j = (1 - eps) P_j + eps / 2, effect from its exact conditional
        g_on = one_minus_eps * P_on[j] + eps_half
        if np.random.random() < g_on:
            z_new = 1
        else:
            z_new = 0
        qn = q_norm_sq[j]
        b_new = 0.0
        if z_new == 1:
            b = (dot[j] + qn * beta[j]) / qn
            post_var = 1.0 / (1.0 / V[j] + 1.0 / slab_var[j])
            b_new = post_var * (b / V[j] + slab_mean[j] / slab_var[j]) + np.sqrt(post_var) * np.random.normal()
        if z_new != z[j]:
            n_changes += 1

        # Apply the move: residuals, all dot products (through the Gram matrix), and the conditionals of j's neighbours and the sums
        # of their blocks (a null site that stays null leaves everything unchanged)
        d = b_new - beta[j]
        z[j] = z_new
        beta[j] = b_new
        if d != 0.0:
            for i in range(n_components):
                resid[i] -= Qt[j, i] * d
            for a in range(n_snps):
                dot[a] -= G[a, j] * d
            for idx in range(nbr_indptr[j], nbr_indptr[j + 1]):
                _update_site(nbr_indices[idx], dot, beta, z, q_norm_sq, V, total_var, half_log_var_ratio, log_w_off, log_w_on, slab_mean, P_on, logit_on, log_p_sel, p_sel, ref, block_sum, acc_P_on, last_sum_w, sum_w, log_one_minus_eps, log_eps_half)
            for idx in range(nbr_indptr[j], nbr_indptr[j + 1]):
                blk = nbr_indices[idx] // _SELECTION_BLOCK_SIZE
                if block_stamp[blk] != step:
                    block_stamp[blk] = step
                    _sum_block(blk, p_sel, block_sum)
    for j in range(n_snps):
        acc_P_on[j] = (acc_P_on[j] + P_on[j] * (sum_w - last_sum_w[j])) / sum_w
    stats[0] = n_changes
    stats[1] = log_scale + np.log(sum_w)
    stats[2] = sum_w * sum_w / sum_w_sq

def log_normal_density(x, mean, var):
    return -0.5 * np.log(2.0 * np.pi * var) - 0.5 * np.square(x - mean) / var


def sample_scaled_inv_chi_square(df, scale, n_obs, sum_sq):
    # Posterior draw of a variance with a scaled-inverse-chi-square(df, scale) prior after n_obs normal deviates with sum of squares sum_sq
    shape = 0.5 * (df + n_obs)
    rate = 0.5 * (df * scale + sum_sq)
    return 1.0 / np.random.gamma(shape, 1.0 / rate)


def sample_beta_with_floor(n_on, n_total, prior_a, prior_b):
    # Inclusion probability from a Beta(prior_a + n_on, prior_b + n_off) draw, floored at 1 / n_total away from 0 and 1 so that
    # no class becomes impossible
    floor = 1.0 / max(n_total, 2.0)
    return np.clip(np.random.beta(prior_a + n_on, prior_b + n_total - n_on), floor, 1.0 - floor)


def low_rank_ld(ld_mat, variance_explained_threshold):
    # Low-rank eigen-decomposition R ~ U Lambda U' keeping the leading eigenvalues that explain variance_explained_threshold of the
    # trace (as eigen_decompose_ld in eb_coloc preprocess_colocalization_data.py). Returns Q = Lambda^{1/2} U' (k x p) and
    # Lambda^{-1/2} U' (k x p), which maps marginal effects b to w
    eigenvalues, eigenvectors = np.linalg.eigh(ld_mat)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    zero_tolerance = eigenvalues[0] * ld_mat.shape[0] * np.finfo(eigenvalues.dtype).eps
    eigenvalues[eigenvalues < zero_tolerance] = 0.0
    cumulative_variance_explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = int(np.searchsorted(cumulative_variance_explained, variance_explained_threshold) + 1)
    n_components = min(n_components, int(np.sum(eigenvalues > 0.0)))
    eigenvalues = eigenvalues[:n_components]
    U = eigenvectors[:, :n_components]
    Q = np.sqrt(eigenvalues)[:, np.newaxis] * U.T
    lambda_inv_sqrt_Ut = (1.0 / np.sqrt(eigenvalues))[:, np.newaxis] * U.T
    return Q, lambda_inv_sqrt_Ut


class COFINE(object):
    def __init__(self, n_iters=1000, burn_in=400, tgs=True, tgs_epsilon=0.5, tgs_sub_steps_per_snp=1.0, tgs_r2_update_threshold=0.01,
                 pi0_init=5e-3, pi1_init=5e-3, pi_delta_init=5e-3, pi_gamma_init=0.1, lambda_init=0.0,
                 pi_prior_a=1.0, pi_prior_b=1.0, var_prior_df=1e-16, lambda_prior_var=1e4,
                 swap_r2=0.5, swaps_per_path=2.0, learn_link_se_inflation=False, link_se_inflation_prior_df=4.0,
                 update_hyperparameters=True, update_residual_vars=True, seed=None, verbose=True):
        # n_iters: total sweeps; burn_in: sweeps discarded before accumulating posterior summaries
        # tgs: tempered Gibbs for the delta and beta blocks (False: random-scan Gibbs); tgs_epsilon, tgs_sub_steps_per_snp,
        #     tgs_r2_update_threshold: as in mcvc_tempered.py
        # swap_r2: path swaps propose variants j' with r2(j, j') >= swap_r2; swaps_per_path: proposals per gene and sweep, per active
        #     path with an eQTL effect (0: no path swaps)
        # pi_prior_a / pi_prior_b: Beta prior on every inclusion probability; var_prior_df: degrees of freedom of the
        #     scaled-inverse-chi-square priors on every variance (scale = initial value; ~0 is vague); lambda_prior_var: N(0, .) prior on lambda
        # learn_link_se_inflation: sample the inflation c2 of the squared link standard errors (initialized at 1; False: fixed at 1);
        #     link_se_inflation_prior_df: degrees of freedom of its scaled-inverse-chi-square prior (scale 1)
        # Slab variances and tau2 are initialized from the data (25 x median squared standard error: effects of ~5 standard errors)
        if burn_in >= n_iters:
            raise ValueError('burn_in must be smaller than n_iters')
        self.n_iters = n_iters
        self.burn_in = burn_in
        self.tgs = tgs
        self.tgs_epsilon = float(tgs_epsilon) if tgs else 0.0
        self.tgs_sub_steps_per_snp = float(tgs_sub_steps_per_snp)
        self.tgs_r2_update_threshold = float(tgs_r2_update_threshold)
        self.swap_r2 = float(swap_r2)
        self.swaps_per_path = float(swaps_per_path)
        self.pi0 = pi0_init
        self.pi1 = pi1_init
        self.pi_delta = pi_delta_init
        self.pi_gamma = pi_gamma_init
        self.lam = lambda_init
        self.pi_prior_a = pi_prior_a
        self.pi_prior_b = pi_prior_b
        self.var_prior_df = var_prior_df
        self.lambda_prior_var = lambda_prior_var
        self.learn_link_se_inflation = learn_link_se_inflation
        self.link_se_inflation_prior_df = link_se_inflation_prior_df
        self.link_se_inflation = 1.0
        self.update_hyperparameters = update_hyperparameters
        self.update_residual_vars = update_residual_vars
        self.seed = seed
        self.verbose = verbose
        self.hyperparameter_names = ['pi0', 'sigma2_0', 'pi1', 'lambda', 'tau2', 'pi_delta', 'sigma2_delta', 'pi_gamma', 'sigma2_gamma']
        if self.learn_link_se_inflation:
            self.hyperparameter_names.append('link_se_inflation')

    def fit(self, genes):
        # genes: list of dicts, one per gene, from prepare_gene_data()
        if self.seed is not None:
            np.random.seed(self.seed)
            _seed_numba(self.seed)
        self.initialize(genes)
        n_post = self.n_iters - self.burn_in
        self.sampled_hyperparameters = np.zeros((n_post, len(self.hyperparameter_names)))
        for itera in range(self.n_iters):
            sweep_stats = np.zeros(3)  # class changes in beta blocks, sum of ESS over beta blocks, number of beta blocks
            self.swap_stats[:] = 0.0
            for g in np.random.permutation(self.n_genes):
                self.sweep_gene(g, sweep_stats)
            if self.update_hyperparameters:
                self.sample_hyperparameters()
            if self.update_residual_vars:
                self.sample_residual_vars()
            if self.verbose:
                n_beta = sum(int(np.sum(st['zE'])) for st in self.state)
                n_path_beta = sum(int(np.sum((st['zE'] == 1) & (st['n_path'] > 0))) for st in self.state)
                n_delta = sum(int(np.sum(st['zC'])) for st in self.state)
                n_gamma = sum(int(np.sum(st['zG'])) for st in self.state)
                print('iter ' + str(itera) + ': ' + ' '.join(name + '=%.3e' % value for name, value in zip(self.hyperparameter_names, self.current_hyperparameters())) + ' n_active=(beta %d, beta on path %d, delta %d, gamma %d)' % (n_beta, n_path_beta, n_delta, n_gamma) + ' mean_resid_var_eqtl=%.3e' % np.mean([st['resid_var_E'] for st in self.state]) + ' tgs_changes_per_gene=%.1f tgs_mean_ess=%.1f' % (sweep_stats[0] / self.n_genes, sweep_stats[1] / max(sweep_stats[2], 1.0)) + ' path_swaps=%d/%d' % (self.swap_stats[1], self.swap_stats[0]), flush=True)
            if itera >= self.burn_in:
                self.sampled_hyperparameters[itera - self.burn_in] = self.current_hyperparameters()
                for st in self.state:
                    # Each sweep's beta block gives an importance-weighted estimate of the conditional PIPs; average them over sweeps
                    st['sum_pip_weighted'] += st['acc_P_on_E']
                    st['sum_pip_weight'] += 1.0
                    st['sum_beta'] += st['beta']
                    st['sum_H'] += (st['n_path'] > 0)
                    st['sum_m'] += st['m']
                    st['sum_link_prob'] += st['P_on_G']
        # Posterior summaries
        self.n_post = n_post
        for st in self.state:
            st['eqtl_pip'] = st['sum_pip_weighted'] / st['sum_pip_weight']
            st['eqtl_posterior_mean'] = st['sum_beta'] / n_post
            st['path_prob'] = st['sum_H'] / n_post
            st['chromatin_effect_posterior_mean'] = st['sum_m'] / n_post
            st['link_prob'] = st['sum_link_prob'] / n_post
        return self

    def current_hyperparameters(self):
        values = [self.pi0, self.sigma2_0, self.pi1, self.lam, self.tau2, self.pi_delta, self.sigma2_delta, self.pi_gamma, self.sigma2_gamma]
        if self.learn_link_se_inflation:
            values.append(self.link_se_inflation)
        return np.asarray(values)

    def initialize(self, genes):
        self.genes = genes
        self.n_genes = len(genes)
        self.state = []
        for gene in genes:
            p = gene['Qt'].shape[0]
            K = gene['wC'].shape[0]
            G = np.ascontiguousarray(gene['Qt'] @ gene['Qt'].T)
            # A variant whose LD lies entirely in the discarded eigen-directions has |q_j|^2 ~ 0 (no information in w); the floor
            # keeps its single-site regression finite (its likelihood is then flat and it follows the prior)
            q_norm_sq = np.maximum(np.diag(G).copy(), 1e-12)
            r2 = np.square(G / np.sqrt(np.outer(q_norm_sq, q_norm_sq)))
            np.fill_diagonal(r2, 1.0)
            neighbours = [np.flatnonzero(r2[j] >= self.tgs_r2_update_threshold) for j in range(p)]
            nbr_indptr = np.zeros(p + 1, dtype=np.int64)
            nbr_indptr[1:] = np.cumsum([len(nj) for nj in neighbours])
            swap_neighbours = [np.flatnonzero(r2[j] >= self.swap_r2) for j in range(p)]
            swap_neighbours = [nj[nj != j] for j, nj in enumerate(swap_neighbours)]
            st = {
                'p': p, 'K': K, 'G': G, 'q_norm_sq': q_norm_sq, 'nbr_indptr': nbr_indptr, 'nbr_indices': np.concatenate(neighbours).astype(np.int64),
                'swap_neighbours': swap_neighbours,
                # eQTL block
                'beta': np.zeros(p), 'zE': np.zeros(p, dtype=np.int64), 'residE': gene['wE'].copy(), 'resid_var_E': gene['sE'] ** 2,
                # caQTL blocks (one row per peak)
                'delta': np.zeros((K, p)), 'zC': np.zeros((K, p), dtype=np.int64), 'residC': gene['wC'].copy(), 'resid_var_C': np.square(gene['sC']),
                # peak-gene links
                'gamma': np.zeros(K), 'zG': np.zeros(K, dtype=np.int64), 'P_on_G': np.zeros(K),
                # chromatin-mediated effect and number of chromatin paths of each variant
                'm': np.zeros(p), 'n_path': np.zeros(p, dtype=np.int64),
                # work arrays of the tempered blocks
                'P_on': np.zeros(p), 'logit_on': np.zeros(p), 'log_p_sel': np.zeros(p), 'p_sel': np.zeros(p), 'ref': np.zeros(1), 'dot': np.zeros(p),
                'block_sum': np.zeros((p + _SELECTION_BLOCK_SIZE - 1) // _SELECTION_BLOCK_SIZE), 'block_stamp': np.zeros((p + _SELECTION_BLOCK_SIZE - 1) // _SELECTION_BLOCK_SIZE, dtype=np.int64),
                'site_consts': np.zeros((3, p)), 'acc_P_on': np.zeros(p), 'last_sum_w': np.zeros(p),
                'snap_beta': np.zeros(p), 'snap_z': np.zeros(p, dtype=np.int64),
                'acc_P_on_E': np.zeros(p),
                # posterior accumulators
                'sum_pip_weighted': np.zeros(p), 'sum_pip_weight': 0.0, 'sum_beta': np.zeros(p), 'sum_H': np.zeros(p), 'sum_m': np.zeros(p), 'sum_link_prob': np.zeros(K),
            }
            self.state.append(st)
        # Slab variances initialized at 25 x the median squared standard error (effects of ~5 standard errors); these are also the
        # scales of their (vague) priors, as are the per-gene and per-peak residual variances (initialized at s^2)
        self.sigma2_0 = 25.0 * np.median([gene['sE'] ** 2 for gene in genes])
        self.tau2 = self.sigma2_0
        sC_all = np.concatenate([gene['sC'] for gene in genes] + [np.zeros(0)])
        se_link_all = np.concatenate([gene['gamma_se'] for gene in genes] + [np.zeros(0)])
        self.sigma2_delta = 25.0 * np.median(np.square(sC_all)) if len(sC_all) > 0 else 1.0
        self.sigma2_gamma = 25.0 * np.median(np.square(se_link_all)) if len(se_link_all) > 0 else 1.0
        self.var_prior_scales = {'sigma2_0': self.sigma2_0, 'tau2': self.tau2, 'sigma2_delta': self.sigma2_delta, 'sigma2_gamma': self.sigma2_gamma}
        self.swap_stats = np.zeros(2)  # path swaps proposed and accepted (reset every sweep)
        self.n_variants_total = sum(st['p'] for st in self.state)
        self.n_caqtl_pairs_total = sum(st['p'] * st['K'] for st in self.state)
        self.n_peaks_total = sum(st['K'] for st in self.state)
        print(str(self.n_genes) + ' genes, ' + str(self.n_variants_total) + ' variant-gene pairs, ' + str(self.n_peaks_total) + ' peak-gene links, ' + str(self.n_caqtl_pairs_total) + ' variant-peak pairs', flush=True)
        print('initial slab variances: sigma2_0=%.3e sigma2_delta=%.3e sigma2_gamma=%.3e' % (self.sigma2_0, self.sigma2_delta, self.sigma2_gamma), flush=True)

    def beta_prior_log_density(self, beta, zE, has_path, m):
        # log p(beta_j | H_j, m_j) under equations 6 and 7: probability mass if beta_j = 0, density if beta_j != 0
        pi = np.where(has_path, self.pi1, self.pi0)
        mean = np.where(has_path, self.lam * m, 0.0)
        var = np.where(has_path, self.tau2, self.sigma2_0)
        return np.where(zE == 1, np.log(pi) + log_normal_density(beta, mean, var), np.log1p(-pi))

    def run_block(self, st, Qt, resid, effects, z, log_w_off, log_w_on, slab_mean, slab_var, resid_var, stats):
        # Tempered Gibbs over one block, then hand on the snapshot state (effects, z and resid are updated in place)
        n_sub_steps = max(1, int(round(self.tgs_sub_steps_per_snp * st['p'])))
        _tgs_block_numba(Qt, st['G'], st['q_norm_sq'], st['nbr_indptr'], st['nbr_indices'], resid, effects, z,
                         np.ascontiguousarray(log_w_off), np.ascontiguousarray(log_w_on), np.ascontiguousarray(slab_mean), np.ascontiguousarray(slab_var), resid_var,
                         n_sub_steps, self.tgs_epsilon, st['P_on'], st['logit_on'], st['log_p_sel'], st['p_sel'], st['ref'], st['block_sum'], st['block_stamp'], st['site_consts'], st['dot'], st['acc_P_on'], st['last_sum_w'], st['snap_beta'], st['snap_z'], stats)
        effects[:] = st['snap_beta']
        z[:] = st['snap_z']
        return

    def sweep_gene(self, g, sweep_stats):
        gene = self.genes[g]
        st = self.state[g]
        Qt = gene['Qt']
        stats = np.zeros(3)
        for k in np.random.permutation(st['K']):
            self.update_gamma(g, k)
        for k in np.random.permutation(st['K']):
            self.update_delta_block(g, k, Qt, stats)
        self.swap_paths(g, Qt)
        self.update_beta_block(g, Qt, stats)
        sweep_stats[0] += stats[0]
        sweep_stats[1] += stats[2]
        sweep_stats[2] += 1.0

    def update_beta_block(self, g, Qt, stats):
        st = self.state[g]
        gene = self.genes[g]
        has_path = st['n_path'] > 0
        pi = np.where(has_path, self.pi1, self.pi0)
        slab_mean = np.where(has_path, self.lam * st['m'], 0.0)
        slab_var = np.where(has_path, self.tau2, self.sigma2_0)
        self.run_block(st, Qt, st['residE'], st['beta'], st['zE'], np.log1p(-pi), np.log(pi), slab_mean, slab_var, st['resid_var_E'], stats)
        st['acc_P_on_E'][:] = st['acc_P_on']
        st['residE'][:] = gene['wE'] - Qt.T @ st['beta']

    def update_delta_block(self, g, k, Qt, stats):
        st = self.state[g]
        gene = self.genes[g]
        p = st['p']
        gamma_k = st['gamma'][k]
        delta_k = st['delta'][k]
        delta_old = delta_k.copy()
        log_on = np.full(p, np.log(self.pi_delta))
        log_off = np.full(p, np.log1p(-self.pi_delta))
        slab_mean = np.zeros(p)
        slab_var = np.full(p, self.sigma2_delta)
        if gamma_k != 0.0:
            # delta_kj also enters the prior of beta_j (through m_j and H_j): condition on everything except peak k's contribution
            m_minus = st['m'] - delta_old * gamma_k
            n_path_minus = st['n_path'] - (delta_old != 0.0)
            beta = st['beta']
            zE = st['zE']
            log_off = log_off + self.beta_prior_log_density(beta, zE, n_path_minus > 0, m_minus)
            # delta_kj != 0 puts variant j on a path: beta_j = 0 has prior mass 1 - pi1, beta_j != 0 has density
            # pi1 N(beta_j; lambda m_j^{-k} + lambda gamma_k delta_kj, tau2), Gaussian in delta_kj
            log_on = log_on + np.where(zE == 1, np.log(self.pi1), np.log1p(-self.pi1))
            c = self.lam * gamma_k
            on_path_active = zE == 1
            if c != 0.0:
                y = (beta - self.lam * m_minus) / c
                u = self.tau2 / (c * c)
                log_on = log_on + np.where(on_path_active, -np.log(np.abs(c)) + log_normal_density(y, 0.0, self.sigma2_delta + u), 0.0)
                slab_mean = np.where(on_path_active, self.sigma2_delta * y / (self.sigma2_delta + u), 0.0)
                slab_var = np.where(on_path_active, self.sigma2_delta * u / (self.sigma2_delta + u), self.sigma2_delta)
            else:
                log_on = log_on + np.where(on_path_active, log_normal_density(beta, self.lam * m_minus, self.tau2), 0.0)
        self.run_block(st, Qt, st['residC'][k], delta_k, st['zC'][k], log_off, log_on, slab_mean, slab_var, st['resid_var_C'][k], stats)
        st['residC'][k] = gene['wC'][k] - Qt.T @ delta_k
        if gamma_k != 0.0:
            st['m'] = st['m'] + (delta_k - delta_old) * gamma_k
            st['n_path'] = st['n_path'] - (delta_old != 0.0) + (delta_k != 0.0)

    def swappable_paths(self, st):
        # (k, j) with delta_kj != 0, gamma_k != 0 and beta_j != 0
        return np.argwhere((st['zC'] == 1) & (st['gamma'][:, None] != 0.0) & (st['zE'][None, :] == 1))

    def swap_paths(self, g, Qt):
        # Metropolis-Hastings moves of (delta_kj, beta_j) to (delta_kj', beta_j') = sign(R_jj') (delta_kj, beta_j), for j' drawn uniformly
        # from j's high-LD neighbours with delta_kj' = 0 and beta_j' = 0 (otherwise the proposal is rejected). The move maps the values
        # one to one, so their slab densities cancel; the target changes through the caQTL (peak k) and eQTL likelihoods and the eQTL
        # prior of j and j'. Hastings ratio: (k, j) is drawn uniformly from the swappable paths, whose number changes with the move
        st = self.state[g]
        n_proposals = int(np.ceil(self.swaps_per_path * len(self.swappable_paths(st))))
        linked = st['gamma'] != 0.0
        for _ in range(n_proposals):
            paths = self.swappable_paths(st)
            if len(paths) == 0:
                return
            k, j = paths[np.random.randint(len(paths))]
            nbr = st['swap_neighbours'][j]
            if len(nbr) == 0:
                continue
            jp = nbr[np.random.randint(len(nbr))]
            self.swap_stats[0] += 1
            if st['zC'][k, jp] == 1 or st['zE'][jp] == 1:
                continue
            sign = np.sign(st['G'][j, jp])
            delta_old = st['delta'][k, j]
            beta_old = st['beta'][j]
            residC = st['residC'][k]
            residE = st['residE']
            residC_new = residC + Qt[j] * delta_old - Qt[jp] * (sign * delta_old)
            residE_new = residE + Qt[j] * beta_old - Qt[jp] * (sign * beta_old)
            log_lik = -0.5 * (residC_new @ residC_new - residC @ residC) / st['resid_var_C'][k] - 0.5 * (residE_new @ residE_new - residE @ residE) / st['resid_var_E']
            idx = np.asarray([j, jp])
            m_new = st['m'][idx] + st['gamma'][k] * np.asarray([-delta_old, sign * delta_old])
            n_path_new = st['n_path'][idx] + np.asarray([-1, 1])
            log_prior = np.sum(self.beta_prior_log_density(np.asarray([0.0, sign * beta_old]), np.asarray([0, 1]), n_path_new > 0, m_new)) - np.sum(self.beta_prior_log_density(st['beta'][idx], st['zE'][idx], st['n_path'][idx] > 0, st['m'][idx]))
            # Swappable paths after the move: those through j are lost (beta_j = 0), those through j' are gained (including (k, j'))
            n_paths_new = len(paths) - int(np.sum((st['zC'][:, j] == 1) & linked)) + int(np.sum((st['zC'][:, jp] == 1) & linked)) + 1
            log_hastings = np.log(len(paths)) - np.log(n_paths_new) + np.log(len(nbr)) - np.log(len(st['swap_neighbours'][jp]))
            if np.log(np.random.random()) < log_lik + log_prior + log_hastings:
                self.swap_stats[1] += 1
                st['delta'][k, jp] = sign * delta_old
                st['delta'][k, j] = 0.0
                st['zC'][k, jp] = 1
                st['zC'][k, j] = 0
                st['beta'][jp] = sign * beta_old
                st['beta'][j] = 0.0
                st['zE'][jp] = 1
                st['zE'][j] = 0
                st['residC'][k] = residC_new
                st['residE'][:] = residE_new
                st['m'][idx] = m_new
                st['n_path'][idx] = n_path_new

    def update_gamma(self, g, k):
        st = self.state[g]
        gene = self.genes[g]
        gamma_hat = gene['gamma_hat'][k]
        gamma_se2 = self.link_se_inflation * gene['gamma_se'][k] ** 2
        gamma_old = st['gamma'][k]
        J = np.flatnonzero(st['zC'][k] == 1)
        delta_J = st['delta'][k, J]
        m_minus = st['m'][J] - delta_J * gamma_old
        n_path_minus = st['n_path'][J] - (1 if gamma_old != 0.0 else 0)
        beta_J = st['beta'][J]
        zE_J = st['zE'][J]
        active = zE_J == 1

        # gamma_k = 0: link statistic under the null; variants of J keep the path status they have through other peaks
        log_off = np.log1p(-self.pi_gamma) + log_normal_density(gamma_hat, 0.0, gamma_se2) + np.sum(self.beta_prior_log_density(beta_J, zE_J, n_path_minus > 0, m_minus))
        # gamma_k != 0: every variant of J is on a path. Observations d = x gamma + noise (noise variances D), gamma ~ N(0, sigma2_gamma):
        # the link statistic (x = 1) and, for variants of J with beta_j != 0, beta_j - lambda m_j^{-k} (x = lambda delta_kj, variance tau2)
        d = np.concatenate([[gamma_hat], beta_J[active] - self.lam * m_minus[active]])
        x = np.concatenate([[1.0], self.lam * delta_J[active]])
        D = np.concatenate([[gamma_se2], np.full(int(np.sum(active)), self.tau2)])
        A = np.sum(x * x / D)
        B = np.sum(x * d / D)
        C = np.sum(d * d / D)
        log_det = np.sum(np.log(D)) + np.log1p(self.sigma2_gamma * A)
        quad = C - self.sigma2_gamma * B * B / (1.0 + self.sigma2_gamma * A)
        log_marginal = -0.5 * (len(d) * np.log(2.0 * np.pi) + log_det + quad)
        log_on = np.log(self.pi_gamma) + np.sum(~active) * np.log1p(-self.pi1) + np.sum(active) * np.log(self.pi1) + log_marginal

        P_on = 1.0 / (1.0 + np.exp(np.clip(log_off - log_on, -700.0, 700.0)))
        st['P_on_G'][k] = P_on
        if np.random.random() < P_on:
            post_prec = 1.0 / self.sigma2_gamma + A
            gamma_new = B / post_prec + np.random.normal() / np.sqrt(post_prec)
            st['zG'][k] = 1
        else:
            gamma_new = 0.0
            st['zG'][k] = 0
        st['gamma'][k] = gamma_new
        st['m'][J] = m_minus + delta_J * gamma_new
        st['n_path'][J] = n_path_minus + (1 if gamma_new != 0.0 else 0)

    def sample_hyperparameters(self):
        df = self.var_prior_df
        scales = self.var_prior_scales
        # eQTL effects off and on chromatin paths
        n_off_total = 0.0
        n_off_on = 0.0
        ss_off = 0.0
        n_path_total = 0.0
        n_path_on = 0.0
        beta_path = []
        m_path = []
        # caQTL and link effects
        n_delta_on = 0.0
        ss_delta = 0.0
        n_gamma_on = 0.0
        ss_gamma = 0.0
        for st in self.state:
            has_path = st['n_path'] > 0
            on = st['zE'] == 1
            n_off_total += np.sum(~has_path)
            n_off_on += np.sum(on & ~has_path)
            ss_off += np.sum(np.square(st['beta'][on & ~has_path]))
            n_path_total += np.sum(has_path)
            n_path_on += np.sum(on & has_path)
            beta_path.append(st['beta'][on & has_path])
            m_path.append(st['m'][on & has_path])
            n_delta_on += np.sum(st['zC'])
            ss_delta += np.sum(np.square(st['delta']))
            n_gamma_on += np.sum(st['zG'])
            ss_gamma += np.sum(np.square(st['gamma']))
        beta_path = np.concatenate(beta_path)
        m_path = np.concatenate(m_path)

        self.pi0 = sample_beta_with_floor(n_off_on, n_off_total, self.pi_prior_a, self.pi_prior_b)
        if n_off_on > 0:
            self.sigma2_0 = sample_scaled_inv_chi_square(df, scales['sigma2_0'], n_off_on, ss_off)
        if n_path_total > 0:
            self.pi1 = sample_beta_with_floor(n_path_on, n_path_total, self.pi_prior_a, self.pi_prior_b)
        if len(beta_path) > 0:
            # lambda | tau2: regression of the active on-path eQTL effects on the chromatin-mediated effects, N(0, lambda_prior_var) prior
            post_prec = np.sum(np.square(m_path)) / self.tau2 + 1.0 / self.lambda_prior_var
            self.lam = np.sum(m_path * beta_path) / self.tau2 / post_prec + np.random.normal() / np.sqrt(post_prec)
            self.tau2 = sample_scaled_inv_chi_square(df, scales['tau2'], len(beta_path), np.sum(np.square(beta_path - self.lam * m_path)))
        if self.n_caqtl_pairs_total > 0:
            self.pi_delta = sample_beta_with_floor(n_delta_on, self.n_caqtl_pairs_total, self.pi_prior_a, self.pi_prior_b)
            if n_delta_on > 0:
                self.sigma2_delta = sample_scaled_inv_chi_square(df, scales['sigma2_delta'], n_delta_on, ss_delta)
        if self.n_peaks_total > 0:
            self.pi_gamma = sample_beta_with_floor(n_gamma_on, self.n_peaks_total, self.pi_prior_a, self.pi_prior_b)
            if n_gamma_on > 0:
                self.sigma2_gamma = sample_scaled_inv_chi_square(df, scales['sigma2_gamma'], n_gamma_on, ss_gamma)
            if self.learn_link_se_inflation:
                # c2 | gamma: the standardized link residuals (gamma_hat_k - gamma_k) / se_k are N(0, c2)
                ss_link = sum(np.sum(np.square((gene['gamma_hat'] - st['gamma']) / gene['gamma_se'])) for gene, st in zip(self.genes, self.state))
                self.link_se_inflation = sample_scaled_inv_chi_square(self.link_se_inflation_prior_df, 1.0, self.n_peaks_total, ss_link)

    def sample_residual_vars(self):
        df = self.var_prior_df
        for gene, st in zip(self.genes, self.state):
            k = gene['Qt'].shape[1]
            st['resid_var_E'] = sample_scaled_inv_chi_square(df, gene['sE'] ** 2, k, np.sum(np.square(st['residE'])))
            for kk in range(st['K']):
                st['resid_var_C'][kk] = sample_scaled_inv_chi_square(df, gene['sC'][kk] ** 2, k, np.sum(np.square(st['residC'][kk])))


def prepare_gene_data(ld_mat, eqtl_effects, eqtl_se, caqtl_effects, caqtl_se, peak_gene_effects, peak_gene_se, variance_explained_threshold=0.995):
    # Inputs for one gene on the standardized-genotype scale (as written by prepare_input_data_for_fine_mapping.py):
    #   ld_mat (p x p), eqtl_effects / eqtl_se (p), caqtl_effects / caqtl_se (K x p), peak_gene_effects / peak_gene_se (K)
    # Marginal effects are converted to b = z * s (s the median standard error of the gene or peak), so that b ~ N(R beta, s^2 R)
    # holds with one standard error per gene or peak, and transformed to w = Lambda^{-1/2} U' b
    Q, lambda_inv_sqrt_Ut = low_rank_ld(ld_mat, variance_explained_threshold)
    sE = float(np.median(eqtl_se))
    wE = lambda_inv_sqrt_Ut @ (eqtl_effects / eqtl_se * sE)
    K = caqtl_effects.shape[0]
    sC = np.zeros(K)
    wC = np.zeros((K, Q.shape[0]))
    for k in range(K):
        sC[k] = np.median(caqtl_se[k])
        wC[k] = lambda_inv_sqrt_Ut @ (caqtl_effects[k] / caqtl_se[k] * sC[k])
    return {'Qt': np.ascontiguousarray(Q.T), 'wE': wE, 'sE': sE, 'wC': wC, 'sC': sC,
            'gamma_hat': np.asarray(peak_gene_effects, dtype=float), 'gamma_se': np.asarray(peak_gene_se, dtype=float)}
