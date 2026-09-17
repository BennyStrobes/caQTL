# Model L: causal eQTL effects as a spike plus a Gaussian centred on the chromatin-predicted effect (implemented)

Companion to `caqtl_mediated_fine_mapping_model.md`. Section 2 is implemented: `ca_qtl_predicted_mean_susie.py` (class `CAQTL_PREDICTED_MEAN_SUSIE`), `susie_rss_mixture.py` (SuSiE-RSS with a per-element Gaussian-mixture prior), runner `run_ca_qtl_predicted_mean_eqtl_finemapping.py`, pipeline step 2d in `run_fine_mapping.sh`. See "Implementation" at the end for what the code does and where it departs from the text.

Three layers of latent effects, each with a standard spike-and-slab prior: variant-to-peak effects informed by caQTL summary statistics, peak-to-gene links informed by the hurdle estimates, and causal eQTL effects whose slab is centred on the effect predicted through chromatin. Section 1 states the full latent Bayesian model. Section 2 shows that its eQTL posterior can be computed exactly by fitting the caQTL and link layers on their own data and marginalizing them inside the eQTL fit, which is the recommended implementation. Section 3 gives the joint variational scheme for the cases where feedback into the caQTL layer is wanted. SuSiE appears only as the algorithm used to compute spike-and-slab posteriors in LD, never as part of the model.

## 1. Full latent Bayesian model

### Data

For gene $g$ with $p$ cis variants and LD matrix $R$ (shared by both modalities, same population):

- eQTL z-scores $z^e_g \in \mathbb{R}^p$;
- for each peak $k \in \mathcal{K}_g$ (the peaks linked to $g$ that survive the LD screen), caQTL z-scores $z^c_k \in \mathbb{R}^p$ over the same variants;
- for each pair $(k, g)$, the hurdle link estimate $\hat g_{kg}$ with standard error $se_{kg}$.

A peak's caQTL data are one object shared by every gene it is linked to.

### Latent variables and priors

**Variant-to-peak effects** (one per peak $k$ and variant $j$, in caQTL z units, shared across the genes linked to $k$):

$$\delta_{kj} \sim \text{Bernoulli}(\pi_c), \qquad b_{kj} \mid \delta_{kj} = 1 \sim N(0, \tau_c^2), \qquad b_{kj} = 0 \text{ if } \delta_{kj} = 0.$$

**Peak-to-gene links** (one per pair, in hurdle units, expression per unit accessibility):

$$u_{kg} \sim \text{Bernoulli}(\pi_g), \qquad g_{kg} \mid u_{kg} = 1 \sim N(0, \tau_g^2), \qquad g_{kg} = 0 \text{ if } u_{kg} = 0.$$

**Causal eQTL effects** (one per gene and variant, in eQTL z units):

$$\gamma_{gj} \sim \text{Bernoulli}(\pi_e), \qquad \beta_{gj} \mid \gamma_{gj} = 1,\, b, g \sim N\big(\lambda\, m_{gj},\ \sigma^2\big), \qquad \beta_{gj} = 0 \text{ if } \gamma_{gj} = 0,$$

$$m_{gj} = \sum_{k \in \mathcal{K}_g} b_{kj}\, g_{kg}.$$

$m_{gj}$ is the effect of variant $j$ on gene $g$ predicted through chromatin: its effect on each linked peak times that peak's effect on the gene, summed over peaks. It is zero unless $j$ is a caQTL variant for some peak that is linked to $g$, and its sign is the product of the caQTL sign and the link sign, which is where sign concordance lives. $\lambda$ converts the product of a caQTL z-score and a hurdle slope into eQTL z units (sample sizes, standardizations). $\sigma^2$ is the spread of true causal effects around their prediction. A non-causal variant has effect exactly zero whatever its prediction; a causal variant with no prediction has the ordinary zero-mean slab.

### Likelihoods

Summary-statistic likelihoods in z units (the RSS form used throughout this code, residual variance 1):

$$z^c_k \mid b_k \sim N(R\, b_k,\ R), \qquad \hat g_{kg} \mid g_{kg} \sim N(g_{kg},\ se_{kg}^2), \qquad z^e_g \mid \beta_g \sim N(R\, \beta_g,\ R).$$

### Shared hyperparameters

$\theta = (\pi_c, \tau_c^2,\ \pi_g, \tau_g^2,\ \pi_e,\ \lambda,\ \sigma^2)$, shared across all peaks, links and genes. Weak hyperpriors: $\lambda \sim N(0, s_\lambda^2)$ with a large $s_\lambda$ (as implemented, default 10; the moment estimate from `tabulate_link_support.py`, 1.64 on the current Mono set, is a check on the fitted value rather than its prior centre), a floor on $\sigma^2$, $\text{Beta}(1,1)$ on the $\pi$'s. Refinements that keep the same structure: a separate slab variance $\sigma_0^2$ for variants with $m_{gj} = 0$ and $\sigma_1^2$ for variants with a prediction; per-bin $\lambda_b$ by hurdle significance so that $m_{gj} = \sum_b \lambda_b \sum_{k \in b} b_{kj} g_{kg}$.

### Joint posterior

$$p(b, \delta, g, u, \beta, \gamma \mid z^c, \hat g, z^e) \;\propto\; \prod_k \Big[ p(z^c_k \mid b_k) \prod_j p(b_{kj} \mid \delta_{kj})\, p(\delta_{kj}) \Big] \prod_{(k,g)} \Big[ p(\hat g_{kg} \mid g_{kg})\, p(g_{kg} \mid u_{kg})\, p(u_{kg}) \Big] \prod_g \Big[ p(z^e_g \mid \beta_g) \prod_j p(\beta_{gj} \mid \gamma_{gj}, b, g)\, p(\gamma_{gj}) \Big].$$

The structure to notice: the caQTL data touch only $b$, the hurdle only $g$, and the eQTL data touch $\beta$ and, through the prior mean of $\beta$, $b$ and $g$. Given $(b, g)$ the three data types are independent.

### What each piece does

**The mean carries sign and magnitude.** A causal variant in a caQTL for a peak with a positive link is expected to raise expression, by an amount that grows with the caQTL effect and the link. For a variant whose prediction is $\lambda m_j$, its single-effect log Bayes factor against "no effect" is

$$\log BF_j = \tfrac12 \log\frac{s_j^2}{\sigma^2 + s_j^2} - \frac{(\hat z_j - \lambda m_j)^2}{2(\sigma^2 + s_j^2)} + \frac{\hat z_j^2}{2 s_j^2},$$

with $\hat z_j$ the effect of $j$ on the current residual and $s_j^2 = 1/R_{jj} = 1$. Relative to a variant with the same $\hat z_j$ and no prediction, $j$ gains

$$\Delta_j = \frac{2\hat z_j\,\lambda m_j - \lambda^2 m_j^2}{2(\sigma^2 + 1)}$$

log units: positive when the observed effect agrees with the prediction in sign and size, negative when it disagrees in sign or is much smaller than predicted. Between two variants in perfect LD, which the eQTL data cannot separate, this is the whole difference, and it is exactly the information the chromatin data add.

**The null component is the escape hatch.** A variant with a large prediction and no eQTL effect is simply not causal; nothing about the gene's fit is distorted. This is what Model A lacked (a strong link had to be explained somewhere) and why $\pi_u \to 1$ in Model D. Spurious links cost nothing here; they also earn nothing.

**$\sigma^2$ decides how hard the prediction pushes, and it is learned.** The gain is scaled by $1/(\sigma^2 + 1)$. If predictions are good across genes, $\sigma^2$ comes out small and a matching prediction is decisive; if they are poor, $\sigma^2$ is large and the prior mean fades into a zero-mean fit. For a prediction $\lambda m_j = 6.4$ (caQTL z 8, link 0.5, $\lambda$ 1.6) and two perfect LD-mates, one carrying the prediction and one not:

| $\sigma^2$ | $\Delta_j$ if $\hat z_j = +6.4$ | PIP split | $\Delta_j$ if $\hat z_j = -6.4$ (discordant) |
|---|---|---|---|
| 25 | 0.8 | 0.69 / 0.31 | -2.4 |
| 9 | 2.0 | 0.88 / 0.12 | -6.1 |
| 4 | 4.1 | 0.98 / 0.02 | -12.3 |

**$\lambda$ is identifiable.** The scale of $b$ is pinned by the caQTL data and the scale of $g$ by the hurdle measurement, so $\lambda$ is a regression slope of causal eQTL effects on their predictions across genes. There is no free $\lambda^2\tau_g^2$ product as in Model A (where $\lambda$ sat in the hurdle measurement equation and $\tau_g^2$ was learned from the same data) and no stacking as in Model E.

## 2. Computing the eQTL posterior: marginalize the caQTL and link layers

### The identity

Because the three data types are conditionally independent given $(b, g)$,

$$p(\gamma_g \mid z^e, z^c, \hat g) \;\propto\; \int p(z^e_g \mid \gamma_g, b, g)\, p(\gamma_g)\; p(b \mid z^c)\; p(g \mid \hat g)\; db\, dg.$$

The exact posterior over eQTL configurations is obtained by taking the caQTL-only posterior of $b$ and the hurdle-only posterior of $g$ as the prior inside the eQTL model and integrating them out. Within one gene, the eQTL data informing $b_{kj}$ and $b_{kj}$ informing $\gamma_{gj}$ is the same evidence going round in a circle; the exact posterior counts it once, and this integral is that posterior. Letting the eQTL data flow back into $q(b)$ and $q(g)$ changes the reported caQTL and link posteriors, not $\gamma_g$ (the one exception, peaks shared by several genes, is in Section 3).

### The caQTL and link posteriors

- $p(b_k \mid z^c_k)$: the posterior of a spike-and-slab in LD. It is computed with SuSiE (sum of single effects) because that is the algorithm that handles LD; the exact single-causal-variant posterior is SuSiE with $L = 1$, $\alpha_j \propto \pi_j BF_j$; $L > 1$ is a cap with unused effects switching off. Mean-field VI over independent per-variant indicators fits the same prior but concentrates mass on one LD-mate arbitrarily and is overconfident, and here that choice would be copied straight into which eQTL variant receives the prediction, so it is not used. What the eQTL side consumes is per variant: $\text{PIP}^c_{kj} = P(\delta_{kj} = 1 \mid z^c_k)$ and the conditional mean and variance of $b_{kj}$ given $\delta_{kj} = 1$.
- $p(g_{kg} \mid \hat g_{kg})$: closed form. $P(u_{kg} = 1 \mid \hat g) = \pi_g BF^h_{kg} / (1 - \pi_g + \pi_g BF^h_{kg})$ with $BF^h_{kg} = N(\hat g; 0, \tau_g^2 + se^2) / N(\hat g; 0, se^2)$, and $g_{kg} \mid u = 1 \sim N\big(\hat g\, \tau_g^2 / (\tau_g^2 + se^2),\ \tau_g^2 se^2 / (\tau_g^2 + se^2)\big)$. Untestable links get a small $P(u = 1)$ automatically; no per-link inclusion variable is needed.

### The eQTL fit: a spike plus a mixture-of-means slab

Integrating $b$ and $g$ out of $N(\beta_j; \lambda m_j, \sigma^2)$ makes the slab a mixture over which links are active. For a variant in one linked peak $k$:

$$\beta_j \mid \gamma_j = 1 \;\sim\; w_j\, N\big(\lambda\, \bar b_{kj} \bar g_{kg},\ \sigma^2 + \lambda^2 v_{jk}\big) + (1 - w_j)\, N(0, \sigma^2), \qquad w_j = \text{PIP}^c_{kj}\; P(u_{kg} = 1 \mid \hat g),$$

with $\bar b, \bar g$ the conditional posterior means and $v_{jk} = E[b^2]E[g^2] - \bar b^2 \bar g^2$ the posterior variance of the product (the product of two Gaussians is approximated by a Gaussian with matched moments). A variant in two peaks has one component per subset of active links. This replaces the point-estimate prediction $\lambda\, E[b]E[g]$, which would give a variant with caQTL PIP 0.3 a mean of 0.3 times the prediction, a compromise that matches neither "it is the caQTL variant" nor "it is not".

Per element the single-effect Bayes factor is $\sum_c w_{jc} BF_{jc}$ with each $BF_{jc}$ the Gaussian Bayes factor above; the posterior of $\beta$ given the element is selected is a Gaussian mixture with weights $\propto w_{jc} BF_{jc}$, whose first two moments are closed form. SuSiE's KL identity still holds because the single-effect posterior is exact. Switching effects off uses SuSiE's own rule (keep effect $l$ if its single-effect log-likelihood exceeds that of no effect), with the slab variance $\sigma^2$ global rather than optimized per effect, so the Model K degeneracy (per-effect variance driven to zero with the effect pinned at its mean) cannot occur.

### Hyperparameter estimation

Empirical Bayes by EM across genes, one step per round of gene fits, using the posterior moments of active effects ($\alpha_{lj}$ the inclusion weight, $E[\cdot]$ the conditional moments under the mixture):

$$\lambda = \frac{\sum_{g,l,j} \alpha_{lj}\, E[m_j\, \beta_{lj}]}{\sum_{g,l,j} \alpha_{lj}\, E[m_j^2]}, \qquad \sigma^2 = \frac{\sum_{g,l,j} \alpha_{lj}\, E\big[(\beta_{lj} - \lambda m_j)^2\big]}{\sum_{g,l,j} \alpha_{lj}},$$

a weighted least-squares fit of posterior causal effects on their predictions and its residual variance. $\pi_g, \tau_g^2$ from the hurdle estimates across all links (standard spike-and-slab EM: $\pi_g$ the mean of $P(u = 1 \mid \hat g)$, $\tau_g^2$ the $P(u=1)$-weighted mean of $E[g^2 \mid u = 1]$). $\tau_c^2$ shared across peaks or per effect as SuSiE estimates it; $\pi_c$ is implicit in the cap $L$. $\pi_e$ uniform, or Model J-style group weights on the variant's strongest link. Rounds as in Model J until $\lambda, \sigma^2$ stop moving; $\sigma^2$ cannot collapse because it pools every causal effect, including the unlinked ones.

### Where a single $\sigma^2$ bites

On the z scale, causal eQTL effects range from about 4 to well over 50 across genes. A single $\sigma^2$ pooled over all causal effects is dominated by the large direct eQTLs, will come out large (tens), and then the prediction moves PIPs only for the strongest predictions (first row of the table). In order:

1. **Two shared variances.** $\sigma_0^2$ for variants with no prediction, $\sigma_1^2$ for variants with one. $\sigma_0^2$ plays the ordinary prior-variance role for the bulk of the genome; $\sigma_1^2$ is the residual around the prediction, which is the quantity that should govern trust.
2. **Per-bin $\lambda_b$** by hurdle significance, so predictions through weak links get their own slope (near zero if uninformative) instead of polluting $\sigma_1^2$. A $B \times B$ solve replaces the scalar step.
3. **Heteroscedastic residual** $\sigma_1^2 + \kappa^2 (\lambda m_j)^2$, only if the diagnostic plot below calls for it.

A gene-specific effect scale (per-effect $V_l$) is the natural fourth refinement but reintroduces the Model K degeneracy for linked variants unless the mean is tied to $\sqrt{V_l}$; not a starting point.

## 3. Joint variational inference (only if feedback into the caQTL layer is wanted)

### What feedback can and cannot add

Within one gene, nothing for $\gamma_g$ (Section 2). Across genes, a real channel: if gene 1's eQTL data pin down the caQTL variant of peak $k$, that sharpened $b_k$ is legitimately informative for gene 2, fit on different data. Also real: refined caQTL and link posteriors as outputs (colocalization-style), and the eQTL informing the link layer, positively when a causal variant matches its prediction and negatively when a causal variant carries the wrong effect, silent when the caQTL variant is not eQTL-causal.

### Factorization and updates

$q = \prod_k q(b_k)\, \prod_{(k,g)} q(g_{kg})\, \prod_g q(\beta_g, \gamma_g)$, with $q(b_k)$ and $q(\beta_g, \gamma_g)$ SuSiE-structured. Coordinate ascent on the ELBO:

- $q(\beta_g, \gamma_g)$: the eQTL fit of Section 2 with the mixture-of-means slab built from the current $q(b)$, $q(g)$ (per-element marginalization, not the mean-field expected mean, which is what overcounts).
- $q(b_k)$: the caQTL single-effect regression on $z^c_k$ in which element $j$'s Bayes factor is multiplied by the eQTL-side factor $\prod_{g \ni k} \exp E_q\big[\gamma_{gj} \log N(\beta_{gj};\ \lambda b_{kj} g_{kg} + \lambda \sum_{k' \ne k} b_{k'j} g_{k'g},\ \sigma^2)\big]$, a Gaussian in $b_{kj}$ with precision $\lambda^2 \sum_g \text{PIP}^e_{gj} E[g_{kg}^2] / \sigma^2$. Exact per element.
- $q(g_{kg})$: spike-and-slab posterior combining the hurdle likelihood with the analogous Gaussian factor from the eQTL, precision $\lambda^2 \sum_j \text{PIP}^e_{gj} E[b_{kj}^2] / \sigma^2$. Closed form.
- Hyperparameters: the M-steps above.

**Cavity discipline.** When updating $q(\beta_g, \gamma_g)$, the $q(b_k)$ it consumes must exclude gene $g$'s own eQTL factor (leave-one-gene-out), otherwise gene $g$'s evidence is handed back to itself as if independent. Plain mean-field VI has no such rule and self-reinforces; that is the mechanism behind Model A's dominating link term. With the cavity rule the multi-gene channel is captured in one or two rounds.

### Cost

A variational scheme over products of latent variables, outer iterations of the size Model A had, and a $\lambda$–$\tau_g^2$ trade-off when both are learned jointly from links with weak hurdle evidence. For fine-mapping the eQTL, Section 2 gives the same posterior at Model J-scale compute with no feedback loop to go wrong.

## Relation to the earlier models

- **A / D**: the link was latent with a measurement likelihood and the mediated effect was deterministic, so the link term dominated, $\lambda$ was not identified and untestable links drove $\pi_u \to 1$. Here the chromatin layers only centre the slab, the null component exists at the variant level, and $\lambda$ is a slope.
- **E / F**: peak genetic components were added to the design, giving stacking. Here nothing is added to the design.
- **J**: the hurdle acts through group inclusion weights only. Here sign and magnitude act through the mean; J's group weights can still set $\pi_e$ per variant.
- **K**: a signed prior mean at the peak level with per-effect variances, which is what made it degenerate. Here the variance is global.

## What it does not do

- No penalty for spurious links, only no reward; $\sigma_1^2$ absorbs their cost across genes, and per-bin $\lambda_b$ localizes it to weak bins.
- Gains are limited to loci where the caQTL posterior localizes the caQTL variant within the eQTL credible set and the predicted effect is comparable to $\sigma_1$. Elsewhere it reduces to eQTL-only fine-mapping with a fixed prior variance.

## Outputs

- Per gene: PIPs; per variant the prediction weight $w_j$, the prior mean $\lambda \bar b \bar g$, the posterior mean effect, and for variants with PIP above 0.5 the standardized residual $(E[\beta_j] - \lambda m_j)/\sigma_1$; a per-variant mediated share (fraction of the posterior mean explained by the prediction).
- Shared: $\lambda$ (or $\lambda_b$), $\sigma_0^2$, $\sigma_1^2$, $\pi_g$, $\tau_g^2$, rounds, summed ELBO.
- Optional (Section 3 or a post-hoc reweighting): caQTL and link posteriors updated by the eQTL evidence.
- Diagnostic plot: posterior mean effect against $\lambda m_j$ for variants with PIP above 0.5, across genes; slope near 1, scatter near $\sigma_1$.

## Recommendation

Implement Section 2: spike-and-slab posteriors for $b$ (SuSiE on the caQTL z-scores) and $g$ (closed form from the hurdle), marginalized per element inside a variant-only SuSiE-RSS eQTL fit with the mixture-of-means slab, two shared variances, per-bin $\lambda_b$ initialized at the moment estimate, and a Model J-style round loop. Read $\sigma_1^2$ against the typical size of $\lambda m_j$ on the first real run before looking at PIPs. Add the leave-one-gene-out feedback of Section 3 afterwards if peaks shared across genes turn out to matter.

## Implementation (as coded)

Runner `run_ca_qtl_predicted_mean_eqtl_finemapping.py`, same inputs and output files as the other runners (PIPs, shared hyperparameters, gene convergence, components with per-gene `direct_pip`, `mediated_pip`, `caqtl_pip`, `eqtl_posterior_mean`, `eqtl_predicted_mean` and `prediction_weight` arrays). Works with `visualize_fine_mapping_comparison.py`, `plot_fine_mapping_example_loci.py` and `tabulate_link_support.py` (which reads `link_scale`, the $\lambda$ of the strongest hurdle bin).

- **Link layer.** `fit_link_spike_slab`: EM for $\pi_g, \tau_g^2$ over all links from the hurdle estimates alone; per link $P(u = 1 \mid \hat g)$ and the conditional mean and variance of $g$. Printed at startup with the number of links above 0.5 and 0.9 and the counts per hurdle bin.
- **caQTL layer.** One SuSiE-RSS fit per peak on the caQTL z-scores (`--L_caqtl`, default 5), frozen. Conditional moments of $b_{kj}$ given non-zero are taken as $E[b_{kj}] / \text{PIP}_{kj}$ and $E[b_{kj}^2] / \text{PIP}_{kj}$ over the active effects (exact for a single effect on the variant).
- **Mixture per variant.** A (peak, variant) pair enters a variant's mixture when caQTL PIP $\times P(u = 1)$ is at least `--min_caqtl_pip` (default 1e-3); at most `--max_peaks_per_variant` pairs (default 3, strongest kept), one component per subset of them, component 0 the empty subset. The product $b_{kj} g_{kg}$ is summarized by its posterior mean and variance and the components of a subset add over its peaks. Stored compactly (only variants with a prediction); the full $p \times C$ arrays are assembled at fit time.
- **eQTL fit.** `SUSIE_RSS_MIXTURE`: the base class with `single_effect_regression` replaced by the mixture version (per-element log Bayes factor as a log-sum-exp over components, posterior mean and second moment as mixture moments), no per-effect prior-variance optimization; an effect is on when its single-effect log likelihood ratio against no effect is positive (`--all_effects_on` keeps every effect on). Verified to reproduce the base class exactly with one zero-mean component. After the fit the component posteriors (`omega`, `comp_mean`, `comp_var`) are recomputed from each effect's final residual for the hyperparameter updates.
- **Hyperparameters.** $\lambda_b$ per hurdle bin (`--t_bin_edges`, default `2,4`; `none` for a single $\lambda$) by the weighted least-squares step of Section 2 with the component variances held at the current $\lambda$ and a Gaussian prior $N(0, s_\lambda^2)$, `--lambda_prior_sd` default 10, initialized at `--lambda_init` (default 0, so the first round is a plain fixed-variance fit and the first $\lambda$ is the calibration regression of posterior causal effects on their predictions). $\sigma_0^2$ (unlinked) and $\sigma_1^2$ (linked) as the pooled second moments of their classes, the linked one net of $\lambda^2$ times the prediction variance, floored at `--sigma2_floor` (default 0.1), initialized at `--sigma2_init` (default 25), updated only when the class carries at least 0.1 expected causal effects; `--single_variance` ties them. Rounds of gene fits until the maximum relative change of $\lambda$ and the variances is below `--tol` (default 1e-3), at most `--max_rounds` (default 30). Each round prints $\lambda$ per bin, both variances, the expected number of causal effects with and without a prediction, and the summed ELBO.
- **Summaries.** `eqtl_pip` from the active effects; `direct_pip` and `mediated_pip` split each variant's mass by whether the selected component is the empty subset or a chromatin prediction; `peak_selection_prob` and `peak_selection_mass` are the posterior mass of causal effects acting through each peak (written as `link_inclusion_probs` for the plots); `link_probs` and `link_means` are the hurdle-only posterior of each link.
- **Not implemented:** the leave-one-gene-out feedback of Section 3, the heteroscedastic residual, and the moment-estimate initialization of $\lambda$ (replaced by the zero-mean prior at the user's request).

**Synthetic checks** (scratch sets used for the earlier models; mean PIP at the true causal variant, eQTL-only SuSiE versus this model): spurious-link set 0.925 to 0.969 at the mediating variant and 0.021 to 0.000 at the spurious peak's caQTL variant, $\lambda_{t>4}$ 0.93 with $\sigma_1^2$ at the floor (the synthetic predictions are exact); uninformative-links set 0.845 to 0.944 and 0.021 to 0.000; direct-effect set unchanged with $\lambda = 0$; discordant set unchanged, $\lambda_{t>4} = -0.10$ with $\sigma_1^2 = 35$ (half the predictions have the wrong sign, so the prediction is learned to be untrustworthy). No false positives at PIP above 0.5 on any set.
