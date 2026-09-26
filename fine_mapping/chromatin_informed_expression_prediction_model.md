# Chromatin-informed expression prediction: proposed two-stage model

Ben Strober, 2026-09-21

## 1. Summary

Implemented on 2026-09-21 as steps 3 and 4 of `run_fine_mapping.sh`: stage 1 is `chromatin_only_expression_prediction.py` (run by `run_chromatin_only_expression_prediction.py`, and used as a prediction method in its own right) and stage 2 is `chromatin_informed_expression_prediction.py` (run by `run_chromatin_informed_expression_prediction.py`). One departure from this document: the implementation keeps the RSS likelihood on the standard errors, with no sample size and no per-gene residual variance (the likelihood in Section 5 is not used). It is an alternative to the fully joint model in `joint_eqtl_caqtl_expression_prediction.py` for the purpose of predicting expression.

- **Stage 1** fits chromatin on its own data, once: caQTL effects for each peak (SuSiE) and peak-to-gene links (spike and slab). From these it computes, for every gene and variant, the probability that the variant has a chromatin path to the gene and the mean and variance of the effect predicted through that path.
- **Stage 2** fits eQTL effects with a spike-and-slab prior that uses the stage 1 quantities in two ways: a variant with a chromatin path gets its own prior inclusion probability, and its slab is centred on the chromatin-predicted effect with its own variance.

Nothing flows back from the eQTL data into the caQTL effects or the links. The eQTL-only model in `eqtl_only_expression_prediction.py` is the special case in which no variant has a path.

## 2. Why change the joint model

On GTEx whole blood the joint model and the eQTL-only model gave the same accuracy (mean per-gene correlation 0.0623 for both; correlation of 0.996 between their per-gene correlations). Three properties of the joint model explain this.

1. Chromatin only moves the slab mean, and the slab is diffuse. One variance $\sigma_e^2$ covers all eQTLs, most of which have no prediction, so it is as wide as a typical effect. In the test run the slab SD was about 0.035 and a typical prediction about 0.026. Shifting the slab centre by less than one slab SD changes almost nothing.
2. Chromatin never changes the prior probability that a variant is an eQTL. Weak but real eQTLs stay at inclusion probabilities near the prior and contribute nothing to the prediction. For prediction this is the lever that matters most.
3. Where the slab mean does act, it mostly decides which member of an LD block carries the signal. Within one population, LD partners predict equally well.

The joint model's extra ingredient, feedback from eQTL data into the caQTL and link layers, moved those layers very little in simulation, accounted for most of the computing time, and was the source of the unstable mediation indicator.

## 3. Data and scale

For gene $g$ with $p$ cis variants and LD matrix $R$:

- eQTL marginal effects $\hat\beta \in \mathbb{R}^p$ with standard errors $s$, and the eQTL sample size $n$;
- for each linked peak $k = 1, \dots, K$: caQTL marginal effects $\hat b_k$ with standard errors $s_k$, and the caQTL sample size;
- for each peak: the estimated link effect $\hat g_k$ with standard error $se_k$.

Genotypes are standardized and expression is not. eQTL effects are per standard deviation of genotype in raw expression units. The scale $\lambda$ converts (caQTL units $\times$ link units) into eQTL units.

## 4. Stage 1: chromatin, fit once

### caQTL effects

For each peak, SuSiE with $L_2$ single effects (`SUSIE_RSS`), giving $\alpha^c_{klj}$, $\mu^c_{klj}$ and $v^c_{klj}$. SuSiE spreads inclusion probability correctly over LD partners, which the per-variant factorized fit does not. Per variant:

$$PIP^c_{kj} = 1 - \prod_l (1 - \alpha^c_{klj}), \qquad E[b_{kj}] = \sum_l \alpha^c_{klj}\mu^c_{klj},$$

$$E[b_{kj}^2] = E[b_{kj}]^2 + \sum_l \Big[ \alpha^c_{klj}\big((\mu^c_{klj})^2 + v^c_{klj}\big) - \big(\alpha^c_{klj}\mu^c_{klj}\big)^2 \Big].$$

### Links

Spike and slab on each link from $\hat g_k$ and $se_k$ alone, with $(\pi_g, \tau_g^2)$ shared across genes and estimated by iterating the closed-form updates (the current `update_peak_gene_effects` with the eQTL terms removed):

$$\frac{1}{t_k^2} = \frac{1}{se_k^2} + \frac{1}{\tau_g^2}, \qquad \nu_k = t_k^2\,\frac{\hat g_k}{se_k^2}, \qquad \text{logit}(a_k) = \text{logit}(\pi_g) + \frac12\log\frac{t_k^2}{\tau_g^2} + \frac{\nu_k^2}{2 t_k^2}.$$

$E[g_k] = a_k\nu_k$ and $E[g_k^2] = a_k(\nu_k^2 + t_k^2)$.

### Per-variant quantities passed to stage 2

The predicted effect is $m_j = \sum_k g_k b_{kj}$. It is non-zero only if variant $j$ is a caQTL variant for a peak whose link is on. Call that event a chromatin path, $c_j = 1$.

$$\rho_j = P(c_j = 1) = 1 - \prod_k \big(1 - a_k\, PIP^c_{kj}\big),$$

$$E[m_j] = \sum_k E[g_k]\,E[b_{kj}], \qquad E[m_j^2] = E[m_j]^2 + \sum_k \Big( E[g_k^2]E[b_{kj}^2] - E[g_k]^2E[b_{kj}]^2 \Big).$$

Because $m_j = 0$ whenever $c_j = 0$, the moments given a path are exact:

$$\mu_j = E[m_j \mid c_j = 1] = \frac{E[m_j]}{\rho_j}, \qquad \omega_j = \text{Var}[m_j \mid c_j = 1] = \frac{E[m_j^2]}{\rho_j} - \mu_j^2.$$

Stage 2 receives $(\rho_j, \mu_j, \omega_j)$ for every variant. Set all three to zero where $\rho_j$ is below a small threshold.

## 5. Stage 2: eQTL effects with a chromatin-informed prior

### Prior

$$c_j \sim \text{Bernoulli}(\rho_j), \qquad \gamma_j \mid c_j \sim \text{Bernoulli}(\pi_{c_j}),$$

$$\beta_j = 0 \ \text{ if } \gamma_j = 0, \qquad \beta_j \mid \gamma_j = 1, c_j = 0 \sim N(0, \sigma_0^2), \qquad \beta_j \mid \gamma_j = 1, c_j = 1, m_j \sim N(\lambda m_j, \sigma_1^2).$$

Integrating over the stage 1 uncertainty in $m_j$ (treated as Gaussian given a path), the third component is

$$\beta_j \mid \gamma_j = 1, c_j = 1 \sim N\big(\lambda\mu_j,\; \sigma_1^2 + \lambda^2\omega_j\big).$$

An uncertain prediction therefore gives a wide slab, and a confident one gives a tight slab. This replaces the $E[m_j^2]$ penalty in the joint model, which over-penalizes uncertain predictions.

Shared across genes and estimated: $\pi_0$, $\pi_1$, $\sigma_0^2$, $\sigma_1^2$ and $\lambda$, with a weak prior $\lambda \sim N(0, 10^2)$. The ratio $\pi_1/\pi_0$ is the enrichment of eQTLs among variants with a chromatin path.

### Likelihood

Use the eQTL sample size and a residual variance per gene, as `SUSIE_RSS` does:

$$X'X = (n-1)R, \qquad X'y = (n-1)\hat\beta, \qquad y'y = \text{median}_j\; (n-1)\big(s_j^2(n-2) + \hat\beta_j^2\big),$$

$$y \mid \beta \sim N(X\beta, \sigma_{e}^2 I), \qquad \sigma_{e}^2 \text{ estimated for each gene}.$$

$\beta$ stays on the same scale as now. This removes the failure seen in simulation, where one gene whose cis variants explained most of the expression variance gained dozens of spurious effects and inflated the shared slab variance. It also sets the amount of shrinkage correctly, which matters more for prediction than it did for fine-mapping.

### Coordinate update for variant $j$

Let $\bar\beta_i = E[\beta_i]$, $d = n - 1$, and remove the other variants from the data:

$$r_j = (n-1)\Big(\hat\beta_j - \sum_{i \ne j} R_{ji}\bar\beta_i\Big).$$

For the two slab components, with prior means and variances $(M_0, T_0) = (0, \sigma_0^2)$ and $(M_1, T_1) = (\lambda\mu_j,\ \sigma_1^2 + \lambda^2\omega_j)$:

$$V_c = \Big(\frac{d}{\sigma_e^2} + \frac{1}{T_c}\Big)^{-1}, \qquad A_c = V_c\Big(\frac{r_j}{\sigma_e^2} + \frac{M_c}{T_c}\Big), \qquad \log BF_c = \frac12\log\frac{V_c}{T_c} + \frac{A_c^2}{2V_c} - \frac{M_c^2}{2T_c}.$$

The posterior over the three outcomes (no effect; effect without a path; effect with a path) is exact given the other variants:

$$q^{null}_j \propto (1-\rho_j)(1-\pi_0) + \rho_j(1-\pi_1), \qquad q^0_j \propto (1-\rho_j)\,\pi_0\,e^{\log BF_0}, \qquad q^1_j \propto \rho_j\,\pi_1\,e^{\log BF_1}.$$

Then

$$\alpha_j = q^0_j + q^1_j, \qquad \bar\beta_j = q^0_j A_0 + q^1_j A_1, \qquad E[\beta_j^2] = q^0_j(A_0^2 + V_0) + q^1_j(A_1^2 + V_1).$$

$\bar\beta_j$ is the prediction weight. Updates are sequential over variants, as in the current numba kernel, which would need a second slab component.

### Residual variance of each gene

$$\sigma_e^2 = \frac{1}{n}\Big[\, y'y - 2\bar\beta' X'y + \bar\beta' X'X \bar\beta + \sum_j d\,\big(E[\beta_j^2] - \bar\beta_j^2\big) \Big].$$

### Shared hyperparameters

Let $q(c_j = 1) = q^1_j + q^{null}_j\,\rho_j(1-\pi_1)\big/\big[(1-\rho_j)(1-\pi_0) + \rho_j(1-\pi_1)\big]$ and $q(c_j = 0) = 1 - q(c_j = 1)$. Summing over all genes and variants:

$$\pi_1 = \frac{\sum q^1_j}{\sum q(c_j = 1)}, \qquad \pi_0 = \frac{\sum q^0_j}{\sum q(c_j = 0)}, \qquad \sigma_0^2 = \frac{\sum q^0_j(A_{0j}^2 + V_{0j})}{\sum q^0_j}.$$

$\lambda$ and $\sigma_1^2$ maximize

$$Q(\lambda, \sigma_1^2) = -\frac{\lambda^2}{200} + \sum_{g,j} q^1_j\Big[ -\frac12\log\big(\sigma_1^2 + \lambda^2\omega_j\big) - \frac{(A_{1j} - \lambda\mu_j)^2 + V_{1j}}{2(\sigma_1^2 + \lambda^2\omega_j)} \Big],$$

a smooth two-parameter problem, solved numerically each iteration from the previous values. Only variants with non-negligible $q^1_j$ contribute. When the $\omega_j$ are ignored the maximizer is closed-form, which gives the starting values:

$$\lambda = \frac{\sum q^1_j A_{1j}\mu_j}{\sum q^1_j\mu_j^2}, \qquad \sigma_1^2 = \frac{\sum q^1_j\big[(A_{1j} - \lambda\mu_j)^2 + V_{1j}\big]}{\sum q^1_j}.$$

Put a floor on each variance.

## 6. Algorithm

1. Stage 1 for every peak and link. Compute $(\rho_j, \mu_j, \omega_j)$ for every gene.
2. Initialize stage 2 from the eQTL-only fit ($\pi_1 = \pi_0$, $\sigma_1^2 = \sigma_0^2$, $\lambda$ from a moment estimate).
3. Iterate: for each gene, one sweep of the variant updates and the residual variance; then the shared hyperparameters.
4. Stop when the relative change in every shared hyperparameter is below `tol`.

Stage 2 costs the same per iteration as the eQTL-only model. The per-peak sweeps, which dominate the joint model's run time, are gone.

## 7. Outputs

- Prediction weights $\bar\beta_j$ and inclusion probabilities $\alpha_j$ (the same first four columns as the current results files).
- $q^1_j$: the probability that variant $j$ is an eQTL whose effect follows the chromatin prediction.
- Shared: the enrichment $\pi_1/\pi_0$, $\lambda$, and $\sigma_1^2$ against $\sigma_0^2$. These say how much chromatin matters before any held-out evaluation. If chromatin is uninformative, $\pi_1 \approx \pi_0$, $\lambda \approx 0$ and $\sigma_1^2 \approx \sigma_0^2$, and the fit reduces to the eQTL-only model.

The inclusion probabilities are still not fine-mapping PIPs. For fine-mapping, replace the stage 2 variant update with the single-effect-with-null form in `joint_eqtl_caqtl_susie_model.md`; stage 1 and the prior are unchanged.

## 8. Known issues and open decisions

1. **High $\pi_1$ is correlation, not mediation.** caQTLs and eQTLs both concentrate in regulatory DNA. Any real enrichment helps prediction, but $\pi_1/\pi_0$ is not evidence that the peaks mediate the genes.
2. **No feedback.** One gene's eQTL can no longer sharpen a caQTL signal that then helps a neighbouring gene sharing the peak. This was the original motivation for a joint fit. No simulation so far has tested it.
3. **Co-accessible peaks.** $\rho_j$ uses "at least one path" and is not inflated by correlated peaks, but $m_j$ sums over peaks and is. Options: keep the sum; use the peak with the largest path probability; cluster peaks first.
4. **Where stage 1 runs.** The input currently gives each gene its own copy of a peak's caQTL statistics over the gene's variants. The simplest implementation runs SuSiE per (gene, peak). Running once per peak over the peak's own window is cleaner and removes the duplication, but needs variant alignment between windows.
5. **Sample size identity.** The $y'y$ identity assumes OLS on standardized genotypes. The FinnGen effects were standardized with $\sqrt{2f(1-f)}$ and may come from a mixed model, so the identity will hold approximately and $n$ is an effective sample size. `SUSIE_RSS` warns when the implied $y'y$ varies by more than 5% across variants.
6. **Small baseline $\rho_j$.** Many small caQTL inclusion probabilities give every variant a small $\rho_j$. The threshold in Section 4 keeps the second slab from being evaluated everywhere.
7. **LD in stage 2.** The per-variant factorized update still places a signal on one member of a tight LD block. This is harmless for prediction within a population and is a limitation for transfer across populations.

## 9. Checks before building

1. For the current results, the correlation between the joint and eQTL-only weight vectors for each gene, and the fraction of genes with any variant having $\rho_j > 0.5$ and $\alpha_j > 0.1$. If that fraction is a few percent, the average gain is bounded by it in any model.
2. The GTEx evaluation stratified by whether the gene has a linked peak, and by eQTL strength. Weak-eQTL genes with a chromatin path are where a gain can appear.
3. The disattenuated correlation between eQTL effects and chromatin-predicted effects, which is the ceiling.

## 10. Validation plan

1. Simulate genes with realistic LD, a known enrichment, a known $\lambda$, and both mediated and unmediated eQTLs. Check recovery of $\pi_1/\pi_0$, $\lambda$, $\sigma_1^2$ and $\sigma_0^2$, and out-of-sample accuracy against the eQTL-only model, separately for weak and strong eQTLs.
2. With $\rho_j = 0$ everywhere, confirm the fit equals the eQTL-only model with the same likelihood.
3. GTEx evaluation: paired difference in per-gene correlation against the eQTL-only model, overall and within the strata of Section 9.
