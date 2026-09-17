# Model E: variants and peak genetic components as competing elements

Implemented in `ca_qtl_elements_susie.py` (class `CAQTL_ELEMENTS_SUSIE`) with the runner `run_ca_qtl_elements_eqtl_finemapping.py`. Companion to `caqtl_mediated_fine_mapping_model.md` (Model A) and the earlier proposals B, C (retired) and D.

Model E fine-maps a gene's eQTL with one SuSiE over an augmented pool of candidate elements: the cis variants and, for each linked peak, the peak's genetic component (its accessibility predicted from the jointly fit caQTL). The direct explanation of an expression signal (a variant) and the mediated one (a peak) compete inside the same categorical posterior, scored by the same likelihood on the same residual. It is cTWAS (Zhao et al. 2024, variants and imputed gene expression as competing elements) moved from the GWAS-gene level to the eQTL-peak level, with three differences: the peak genetic components are random and jointly fit with the caQTL data; a selected peak's effect is the latent peak-gene link value measured by the hurdle estimate; and a peak's prior weight in the pool is proportional to the probability that its link is non-zero.

## Why this structure

In Models A and D the decision "direct effect at variant $j$ or mediated through peak $k$" was made by whichever component claimed the shared residual first, so initialization and update order decided it. The hyperparameters meant to quantify mediation were unidentified ($\lambda$, $\rho$) or attracted to a boundary ($\pi_u$), and in D an untestable link inherited $\pi_u$ and handed its caQTL variants a mediated PIP for nothing. Putting both hypotheses inside one categorical posterior removes the order dependence, replaces the per-link inclusion probability with a symmetric class weight, and gives untestable peaks only the prior share of a single effect.

## Model

**caQTL side (as in Model A).** For each linked peak $k$,

$$A_k = X b_k + e_k, \qquad b_k = \sum_{l=1}^{L_{ca}} b_{kl} \quad \text{(SuSiE single effects)}.$$

The peak's genetic component $m_k = X b_k$ is random: $b_k$ has a posterior fit from the caQTL data and, whenever the peak is selected by the eQTL model, from the expression data. The eQTL side uses $E[b_k]$ and $E[b_k' R\, b_k]$, so a diffusely fine-mapped peak is automatically a weaker predictor than a sharply fine-mapped one.

**Peak-gene link (as in Model A).**

$$g_k = z_k h_k, \qquad z_k \sim \text{Bern}(\pi), \qquad h_k \sim N(0, \tau_g^2), \qquad \hat g_k \sim N(\lambda g_k,\ se_k^2 + \tau_u^2).$$

$z_k$ says whether the hurdle link is real; it is identified from the hurdle marginal (a null bulk plus a tail) and does not by itself decide mediation. $\lambda$ is fixed (from the large gene set; its EM estimate is degenerate, and the pooled moment estimate on subsets has ranged from 1.2 to 37). $\pi$, $\tau_g^2$, $\tau_u^2$ are shared across genes and estimated by EM.

**eQTL regression over variants and peaks.**

$$E = \sum_{l=1}^{L} \beta_l\, w_{\gamma_l} + e, \qquad \gamma_l \in \{1, \dots, p + K\}$$

Single effect $l$ selects one element. Element $j \le p$ is variant $j$ with predictor $x_j$ and a free effect $\beta_l \sim N(0, V_l)$, $V_l$ estimated per effect as in SuSiE. Element $p + k$ is peak $k$ with predictor $m_k$ and effect **fixed at the link value**, $\beta_l = g_k$: a selected peak contributes $g_k\, m_k$, the latent link times the peak's genetic component. A peak element has no effect size of its own.

**Prior on which element a single effect selects.**

$$P(\gamma_l = j) \propto \omega_v \quad (j \le p), \qquad P(\gamma_l = p + k) \propto \omega_p\, q^w_k,$$

with $q^w_k$ the probability that link $k$ is non-zero (below). Only the ratio $\rho_p = \omega_p / \omega_v$ matters; it is shared across genes and estimated by EM on $[10^{-3}, 10^2]$. $\rho_p$ is the enrichment of real-link peak elements over variants: how much more often an eQTL single effect is best explained by a real peak's genetic component than by an individual variant. It is a symmetric split of posterior mass, with no boundary to be attracted to. The cap of 100 exists because beyond a hundred-fold enrichment the prior concentrates on a few peaks per gene and a chance fit can switch an effect on; on data with direct effects the estimate sits well inside the bracket.

**The pool weight $q^w_k$.** A null link should not compete for single effects, so a peak's prior weight scales with the probability its link is non-zero. $q^w_k$ starts at the hurdle-only value $P(z_k = 1 \mid \hat g_k)$ and is refreshed once per outer iteration from the full link posterior $q_k$, which by then also pools the eQTL evidence about $g_k$; within an outer iteration it is held fixed. (Setting `pool_gate='hurdle'` keeps it at the hurdle-only value for the whole run.) The EM step for $\rho_p$ normalizes the posterior mass on peak elements by $\sum_k q^w_k$, the expected number of real-link peaks, not by $K$, so null links neither dilute $\rho_p$ nor vote on it. No separate hurdle-strength modulation of the prior weight is needed: the spike does that job.

**Single effects can be off.** SuSiE switches a single effect off by setting its prior variance $V_l$ to zero, which makes every variant's Bayes factor 1. A tied peak element's Bayes factor does not depend on $V_l$, so that rule alone would leave a switched-off effect spreading its prior share over the peaks. A single effect is therefore on only if its total evidence over the whole pool, $\log \sum_{\text{elements}} \pi_e\, \text{BF}_e$ at the optimized $V_l$, is positive; otherwise it is off, contributes nothing to the residual, the PIPs or the EM, and its posterior over elements is the prior weights. This is SuSiE's $V = 0$ rule generalized to the augmented pool.

**Direction.** The sign of $g_k m_k$ is that of $g_k$, which the hurdle term pulls toward the sign of $\hat g_k$ with precision $\lambda^2 / se_k^2$; for a peak element to win, the expression residual must run in that direction. No separate sign prior.

## Inference

Mean-field variational inference, coordinate ascent, z-score mode as in Model A. Per gene and inner pass:

1. **eQTL single effects.** For each $l$, on the residual $r_l$ (expression minus every other on effect's expected contribution), evaluate the log Bayes factor of each element. Variant $j$: the SER quantity from $x_j' r_l$ and $x_j' x_j$. Peak $k$: the expected log-likelihood ratio of adding $g_k m_k$ under the current factor of $g_k$, $S_{E,k} E[g_k] - \tfrac12 P_{E,k} E[g_k^2]$, with score $S_{E,k} = E[b_k]' X' r_l / \sigma^2_E$ and precision $P_{E,k} = E[b_k' R\, b_k] / \sigma^2_E$, and $E[g_k], E[g_k^2]$ from the spike-slab moments. $V_l$ is optimized on the whole pool's marginal likelihood; the on/off decision follows; prior weights $\omega_v$ and $\omega_p q^w_k$ normalize to the categorical posterior $\alpha_l$.
2. **Links.** The factor of $g_k$ is the spike-slab posterior pooling the hurdle term with the selection-weighted eQTL evidence from the on effects: score $\lambda \hat g_k / w_k + \sum_l \alpha_{l,p+k} S_{E,k}$, precision $\lambda^2 / w_k + \sum_l \alpha_{l,p+k} P_{E,k}$, prior $\pi$, $\tau_g^2$; closed form, giving $q_k$, slab mean and variance. The scores $S_{E,k}$ are recomputed against the residuals as they stand after step 1 (the values seen during each effect's own update are stale once later effects have moved), so this step is exact given the current state.
3. **caQTL single effects.** As in Model A: each $b_{kl}$ pools the caQTL residual with the eQTL residual weighted by the peak's expected eQTL coefficient, $\sum_l \alpha_{l,p+k} E[g_k]$ and $\sum_l \alpha_{l,p+k} E[g_k^2]$ over the on effects.
4. **Residual variances and the gene ELBO**, with the link terms added globally.

Then across genes, once per outer iteration: refresh $q^w_k \leftarrow q_k$ and re-evaluate each gene's categorical prior term under the new weights; M-steps for $\pi$, $\tau_g^2$, $\tau_u^2$ (as in Model A) and for $\rho_p$ (a one-dimensional search on $\sum_g [M_g \log \rho_p - n^{\text{on}}_g \log(p_g + \rho_p Q_g)]$ with $M_g$ the mass on peak elements, $Q_g = \sum_k q^w_k$), followed by a re-evaluation of the prior term under the new $\rho_p$ so that step is an ascent.

Within an outer iteration the pool weights are constants, so every gene pass and M-step is a coordinate ascent on a fixed-weight objective. The refresh redefines the objective, and the printed ELBO can step down there by at most the refresh amount; the log reports that amount (`pool_refresh`) and the largest weight change (`max_dqw`). Convergence requires the absolute ELBO change below the per-gene tolerance, every shared hyperparameter and the pool weights to have moved by less than the relative tolerance.

**Stacking.** A peak can be selected by more than one single effect, and in the tied model this is *not* self-limiting the way it is for variants in SuSiE. A variant's second copy would add a free effect against a residual that no longer wants it. A peak's copy adds the same fixed contribution $g_k m_k$; if the hurdle-implied effect $\hat g_k / \lambda$ is smaller than what expression shows, one copy explains only part of the signal and a second effect selects the same peak to supply the rest. Two copies reproduce the right total, so the fit is happy, $P_k$ is unaffected ($P(\text{some effect selects } k)$ is 1 either way), but the bookkeeping, which assumes at most one copy (it uses $E[N_k]$ where the exact terms need $E[N_k^2]$, and products of means where the shared $g_k$ needs $E[g_k^2]$), is then inconsistent with the objective, which is the source of the small ELBO dips on real data. Stacking is diagnosed by the per-peak selection mass $\sum_{l \text{ on}} \alpha_{l,p+k}$, written to the components file as `peak_selection_mass`; values well above 1 mean the implied effect is too small at the current $\lambda$. It is also why $\lambda$ cannot be learned inside the tied model (see limitations).

Cost: Model A's per-gene cost plus $K$ extra Bayes factors per single effect, dot products with $R\,E[b_k]$, which is already maintained.

## Initialization

One start; the multi-start of Models A and D existed only because the direct-vs-mediated split was decided by the starting point.

**Shared hyperparameters, once.** $\lambda$ from the large gene set (never the per-subset moment estimate). $\pi = 0.2$; $\tau_g^2$ by the robust method of moments on the hurdle estimates (links with finite estimates and SE within ten times the median; refuse to start if the result is at the floor). $\tau_u^2 = 0$, then estimated. $\rho_p = 1$: a real-link peak element and a variant element start with the same prior weight, so the first pass is a fair contest and the direction of the first EM step is informative.

**Per gene, on first visit.**

1. caQTL side: SuSiE-RSS on each peak's caQTL alone, to convergence, giving $E[b_k]$, $R\,E[b_k]$ and $E[b_k' R\, b_k]$.
2. Links: the spike-slab posterior of each $g_k$ from the hurdle term alone: $q_k$, slab mean $\approx \hat g_k / \lambda$, slab variance. This sets every peak element's contribution $E[g_k]\, m_k$ and the starting pool weight $\omega_p q^w_k$ with $q^w_k = q_k$.
3. eQTL single effects: all $L$ effects empty, uniform over the pool's prior weights with zero effect. Not an eQTL-only SuSiE fit: copying one in would hand the whole expression signal to the variants before any peak element has been scored, which is the variant-side version of the tilt this model removes. With empty effects, the first SER pass evaluates every variant and every peak on the full expression signal at once, and the split is set by their Bayes factors and the equal prior weights.

**Pass order**, first and thereafter: eQTL single effects, then links, then caQTL effects. The empty start is not "mediated first": a peak element wins the first pass only if $E[g_k]\, m_k$ fits the expression signal better than the best single variant does, with equal prior weight.

**Later visits.** Per-gene state carried in memory; only the LD is reloaded.

## Outputs

- Per single effect: a categorical posterior $\alpha_l$ over variants and peaks, and a credible set that may contain variants, peaks, or both.
- Per variant: direct PIP, mediated PIP, combined eQTL PIP (below).
- Per peak: the selection probability, $P(\text{some on single effect selects } k)$; $P_k$, that probability times $q_k$, the probability the peak mediates; and the posterior of $g_k$ ($q_k$, slab mean and variance).
- Shared: $\rho_p$, $\pi$, $\tau_g^2$, $\tau_u^2$.

### The combined eQTL PIP

Same composition as Models A and D, with the peak posterior coming from the SER instead of a link indicator. For variant $j$ in a gene with $L$ single effects and $K$ peaks:

- **Direct PIP**: $1 - \prod_{l \text{ on}} (1 - \alpha_{lj})$, SuSiE's PIP over the variant elements of the pool.
- **Peak posterior**: $P_k = \big[1 - \prod_{l \text{ on}} (1 - \alpha_{l,p+k})\big] \cdot q_k$.
- **Mediated PIP**: $1 - \prod_k \big(1 - P_k \cdot \text{caQTL PIP}_{kj}\big)$, with the caQTL PIP from the peak's own fine-mapping under the joint fit.
- **Combined eQTL PIP**: $1 - (1 - \text{direct})(1 - \text{mediated})$, the probability the variant affects expression by either route.

What follows from the structure:

- A single effect's mass is shared across variants and peaks. At a colocalized locus where the peak element and its lead variant are collinear, the split might be 0.5 on the variant and 0.5 on the peak, and the peak's 0.5 flows back to the same variant through its caQTL PIP near 1. The combined PIP is then close to 1, which is right: the variant is causal either way and only the route is undetermined.
- Where the caQTL credible set is broader than the eQTL's, the mediated route spreads the peak's mass across the caQTL candidates, so the combined PIP of the eQTL lead can be lower than its eQTL-only PIP. That is the model saying the expression signal may act through any of those chromatin variants.
- Mixed credible sets at the element level need a variant-level rendering: a peak element in a credible set is expanded into that peak's caQTL credible set, so a 95% set reads "these variants directly, or one of those variants through peak $k$".

## Identifiability and limitations

- **Collinearity is the honest failure mode.** When a peak's caQTL has one causal variant with PIP near 1, $m_k \approx x_j$ up to scale and the data cannot separate "direct at $j$" from "through $k$"; the tied effect $g_k$ helps only if the implied size $g_k m_k$ differs from what a free direct effect would fit, which requires $\lambda$ to be right. SuSiE treats the pair as two elements in perfect LD: the credible set contains both and the split follows the prior weights. That is the correct statement of what summary statistics can say. Separation exists where $m_k$ differs from every single variant column (multi-signal caQTLs, LD shapes the panel represents well); $\rho_p$ gets its information from those cases. Report how many real-link peaks are effectively collinear with their lead variant.
- **$\lambda$.** Fixed, from the large gene set. If misspecified, peak elements lose to variants everywhere and $\rho_p \to 0$: visible, but confounded with "peaks do not mediate". A $\lambda$ sensitivity grid and the E-free comparison below are mandatory.
- **$\lambda$ as a random variable (`--estimate_link_scale`) does not recover it.** The runner can give $\lambda$ a Gaussian variational factor, updated from the links weighted by their selection probability (unselected links cannot inform $\lambda$, since the hurdle term fixes only $\lambda g_k$), with $E[\lambda]$ and $E[\lambda^2]$ entering the link posteriors, the link ELBO terms and the $\tau_u^2$ step; $\tau_u^2$ must then be fixed, because a wrong $\lambda$ is otherwise absorbed as link noise. On the synthetic sets it is mechanically sound (holds at the prior when nothing is selected) but does not find the truth: started at twice the true value it stays there, started at half it never selects a peak and so never moves, started at the truth it drifts by 15%. The reason is stacking (see Inference): when $\lambda$ is too large the model selects the peak twice rather than move $\lambda$, the pooled posterior of $g_k$ stays at $\hat g_k / \lambda$ because the hurdle precision dwarfs the expression precision, and the update returns the current value; when $\lambda$ is too small the peak overshoots, loses to the variant, and nothing anchors $\lambda$ at all. No in-model estimator escapes this while the peak effect is tied; the grid over $\lambda$ with E-free as the $\lambda$-free control remains the way to choose it, and a soft tie (a per-effect peak coefficient $\beta_l \sim N(g_k, \tau_\beta^2)$, which removes the reason to stack and makes the fitted coefficients informative about $\lambda$) is the model change that would make it learnable. The flag is kept for experiments; its posterior sd understates the uncertainty (mean-field ignores the $\lambda$-$g_k$ coupling).
- **Feedback into $b_k$.** Expression data reshape the caQTL profile when a peak is selected. In A/D nothing pushed back; here the peak element competes with variants on the same residual, so a profile that must be bent to fit expression competes against a variant that fits it directly. This moderates the chase at colocalized loci but does not remove it.
- **The pool refresh.** The eQTL evidence reaches the pool weight only through the once-per-iteration refresh, so the printed ELBO is a per-iteration diagnostic rather than a global certificate; the fixed point is a state where $q_k$ and the weights built from it agree. The hurdle gate gives a monotone ELBO at the cost of the eQTL evidence never reaching the pool weight; on the synthetic checks both gates reach the same answers.
- **Untestable real links** (implied effect too small for the expression data to see) have peak-element Bayes factors near 1 and receive only their prior share of an on single effect's mass: small, and neutral in the EM. No prior-driven mediated PIPs of the Model D kind.
- **Colocalization vs mediation.** A peak element winning means the expression signal follows the peak's genetic profile, at the hurdle-implied size and sign, better than any single variant does. Stronger than colocalization at a lead variant; still not causal mediation, and the collinear cases are exactly where the two cannot be told apart.

## Comparison variant: E-free

`--free_peak_effects`: a selected peak element gets a free effect $\beta_l \sim N(0, V_l)$ estimated like a variant's, sharing the effect's $V_l$; the link then only gates the pool and does not enter the eQTL model, so $\lambda$ and $\tau_u^2$ play no role in the eQTL side (the link factor is fit from the hurdle alone). Comparing $\rho_p$ between the two variants at a grid of $\lambda$ values tests whether the hurdle magnitude carries usable information. On synthetic data E-free rejects the same non-mediating and direct-only cases but sometimes loses a real mediating peak to its lead variant when the caQTL profile is imperfectly fine-mapped, since a free effect has nothing beyond profile shape to tie the peak to the eQTL, and once admitted a chance fit with $\rho_p$ at its cap; the tied version is the stronger of the two there.

## Relation to the other models

| | A | B | D | E | E-free |
|---|---|---|---|---|---|
| hurdle used as | measurement of effect | prior on inclusion | measurement + inclusion gate | measurement of the peak element's effect; spike gates the pool | spike gates the pool |
| needs $\lambda$ | fixed | no | fixed | fixed | no |
| direct vs mediated decided by | update order | eQTL BF vs prior | eQTL BF vs prior, order-dependent | same SER, symmetric | same SER, symmetric |
| caQTL profile | jointly refit | jointly refit | jointly refit | jointly refit | jointly refit |
| mediation estimand | $\pi$ (hurdle shape) | $a, b$ | $\pi_u$ (boundary-attracted) | $\rho_p$ per real link | $\rho_p$ per real link |
| untestable real links get | link prob $\times$ caQTL PIP | prior $\times$ Occam | $\pi_u \times$ caQTL PIP | prior share of an on effect | prior share of an on effect |
| null links | count toward $\pi$ | prior $\approx 0$ | count toward $\pi$ | weight $\approx 0$ in the pool | weight $\approx 0$ in the pool |

## Evaluation plan

1. Simulations from Model A's generative process with single- and multi-signal caQTLs and a fraction of null links: recovery of mediating peaks where separable, mixed credible sets where not, null links out of the pool, $\rho_p$ without drift; E-free alongside.
2. Real data, E vs E-free at $\lambda \in \{0.5, 1.2, 2\}$: $\rho_p$ as a function of $\lambda$ says whether the hurdle magnitude helps.
3. Distribution of $P_k$ against the caQTL's number of signals; fraction of credible sets that are pure-variant, pure-peak, mixed.
4. The link-support table (implied vs observed eQTL z at caQTL leads) with $P_k$ in place of D's inclusion probabilities.
5. Combined eQTL PIPs vs eQTL-only and vs D on the same screened gene set: threshold counts with gene-bootstrap CIs, resolution, components plot.

## Implementation

**Files.** `ca_qtl_elements_susie.py` (built on the Model A class for data preparation, the caQTL side, the link spike-slab and its M-steps) and `run_ca_qtl_elements_eqtl_finemapping.py`. Runner flags: `--link_scale_init` ($\lambda$; with `--estimate_link_scale` and `--link_scale_prior_sd` it becomes the prior mean of a variational factor, see limitations, which requires `--no_estimate_link_bias_variance`), `--peak_weight_ratio_init` / `--no_estimate_peak_weight_ratio` ($\rho_p$), `--pool_gate running|hurdle`, `--free_peak_effects`, and the shared `--no_estimate_link_bias_variance`, `--link_bias_variance`, `--estimate_residual_variance`, `--max_outer_iter`, `--hyper_tol`, `--elbo_tol`, `--no_component_output`. Outputs use the same formats as Models A/D: the components file's `link_inclusion_probs` column holds $P_k$ and `peak_selection_probs` holds the raw selection probability, so the comparison plots, locus plots and link-support table carry over unchanged. `run_fine_mapping.sh` has a disabled step 2b with the Model E and E-free commands.

**Synthetic checks** (tied version, both gates give the same answers):

| scenario | outcome |
|---|---|
| real mediating peak + strong non-mediating link, per gene | $P_k$ 1.0 and 0.0; mediated PIP 1.0 at the real causal variant, 0.01 at the other |
| direct eQTL at a non-caQTL variant, two strong peaks with strong links | $P_k = 0$ for all peaks, direct PIP 1.0, $\rho_p$ at its floor |
| one real, one spurious, eight untestable links per gene | $P_k$ 1.0, 0.0, and $\approx 0$ for the untestable ones |
| first synthetic set incl. a gene with no peaks | correct; $K = 0$ handled |

With the hurdle gate the ELBO is monotone on every set; with the running gate the only step-downs coincide with pool refreshes and are of their size (at most 0.013 on an ELBO of about 700).
