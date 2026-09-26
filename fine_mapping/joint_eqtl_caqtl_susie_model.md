# Joint fine-mapping of eQTLs and caQTLs: SuSiE-based model (revised)

Ben Strober, first version 2026-09-18, revised 2026-09-21

## 1. Summary

This is a proposal, not yet implemented. It fine-maps a gene's eQTLs and the caQTLs of its linked peaks with SuSiE, puts a spike and slab on each peak-gene link, and lets the chromatin data shape the prior of every eQTL effect.

Three things changed from the first version, because of what the expression prediction experiments showed (GTEx, 2026-09-21).

1. **Chromatin now changes where eQTL effects are expected, not only their prior mean.** A variant with a chromatin path to the gene is $e$ times more likely to carry an eQTL effect, with $e$ learned across genes. In the first version chromatin only shifted the centre of one wide slab, and that design had no measurable effect.
2. **Path variants get their own slab.** It is centred on the chromatin-predicted effect and has its own variance, which is inflated where the prediction is uncertain. The $E[m_j^2]$ penalty of the first version, which over-penalized uncertain predictions, is gone.
3. **The caQTL and link fits are frozen by default.** They are fit on their own data first. Feedback from the eQTL data into them is an option (Section 7), not the default: in simulation it moved those fits very little, and it is the source of the instabilities seen in Models A, D and E.

With the chromatin fits frozen, this is the fine-mapping counterpart of the two-stage prediction model in `chromatin_informed_expression_prediction_model.md`: the same prior, with a SuSiE eQTL layer in place of the per-variant factorized one. Stage 1 already exists (`chromatin_only_expression_prediction.py`), and the two-slab logic exists in `chromatin_informed_expression_prediction.py`.

## 2. Why the eQTL layer must be SuSiE

Simulation: 500 samples, 60 variants, one causal variant (variant 22) inside a block of four near-identical variants (20 to 23). A model with one independent inclusion probability per variant had converged after one sweep and was unchanged after 2,000.

| $r$ within block | one probability per variant, variants 20 to 23 | SuSiE PIPs, variants 20 to 23 |
|---|---|---|
| 1.00 | 1.00, 0.01, 0.01, 0.01 | 0.29, 0.29, 0.29, 0.29 |
| 0.99 | 1.00, 0.02, 0.01, 0.02 | 0.60, 0.21, 0.28, 0.09 |
| 0.95 | 1.00, 0.03, 0.03, 0.03 | 0.02, 0.42, 0.44, 0.13 |

Independent indicators cannot represent "one of these variants is causal, and I do not know which". A single effect with one categorical distribution over variants can.

## 3. Data and scale

For gene $g$ with $p$ cis variants and LD matrix $R$ (the same for both modalities):

- eQTL marginal effects $\hat\beta$ with standard errors $s$;
- for each linked peak $k = 1, \dots, K$: caQTL marginal effects $\hat b_k$ with standard errors $s_k$;
- for each peak: the estimated link effect $\hat g_k$ with standard error $se_k$.

Genotypes are standardized and expression is not. eQTL effects are per standard deviation of genotype in raw expression units. The scale $\lambda$ converts (caQTL units $\times$ link units) into eQTL units. Write $z = \hat\beta / s$.

## 4. Model

### Likelihoods

$$\hat\beta \mid \beta \sim N\big(S R S^{-1}\beta,\; S R S\big), \qquad \hat b_k \mid b_k \sim N\big(S_k R S_k^{-1} b_k,\; S_k R S_k\big), \qquad \hat g_k \mid g_k \sim N\big(g_k,\; se_k^2\big),$$

with $S = \text{diag}(s)$ and $S_k = \text{diag}(s_k)$. No sample size and no $R^{-1}$ are needed.

### caQTL layer

Standard SuSiE for each peak, with $L_2$ single effects and a prior variance for each effect. The priors are centred at zero, so the usual pruning of effects by $V \to 0$ is safe here.

### Link layer

$$u_k \sim \text{Bernoulli}(\pi_g), \qquad g_k \mid u_k = 1 \sim N(0, \tau_g^2), \qquad g_k = 0 \text{ if } u_k = 0,$$

with $(\pi_g, \tau_g^2)$ shared across genes.

### Chromatin path of a variant

The predicted effect of variant $j$ on the gene is $m_j = \sum_k g_k b_{kj}$. It is non-zero only if $j$ is a caQTL variant for a peak whose link is on. Call that event a chromatin path, $c_j = 1$. From the caQTL and link posteriors:

$$\rho_j = P(c_j = 1) = 1 - \prod_k\big(1 - a_k\,PIP^c_{kj}\big), \qquad \mu_j = \frac{E[m_j]}{\rho_j}, \qquad \omega_j = \frac{E[m_j^2]}{\rho_j} - \mu_j^2,$$

where $a_k$ is the posterior probability that link $k$ is on, $PIP^c_{kj}$ is the caQTL PIP, and $\mu_j$ and $\omega_j$ are the mean and variance of $m_j$ given a path (exact, because $m_j = 0$ without one). The moments are

$$E[m_j] = \sum_k E[g_k]E[b_{kj}], \qquad E[m_j^2] = E[m_j]^2 + \sum_k\Big(E[g_k^2]E[b_{kj}^2] - E[g_k]^2E[b_{kj}]^2\Big).$$

Variants with $\rho_j$ below a small threshold are treated as having no path.

### eQTL layer

$\beta = \sum_{l=1}^{L} \theta_l \mathbf{1}_{\gamma_l}$. Each single effect has three kinds of outcome:

| outcome | prior probability | effect size |
|---|---|---|
| absent | $1 - \rho_e$ | contributes nothing |
| at variant $j$, no path | $\rho_e\,(1 - \rho_j)/Z$ | $\theta \sim N(0, \sigma_0^2)$ |
| at variant $j$, path | $\rho_e\, e\,\rho_j / Z$ | $\theta \sim N\big(\lambda\mu_j,\; \sigma_1^2 + \lambda^2\omega_j\big)$ |

with $Z = \sum_i \big[1 + (e - 1)\rho_i\big]$.

- $\rho_e$ is the prior probability that a single effect exists. The expected number of eQTL signals per gene is $L\rho_e$.
- $e$ is the enrichment: how much more likely a variant with a path is to carry an effect.
- $\sigma_1^2$ measures how closely real effects follow the prediction. The term $\lambda^2\omega_j$ widens the slab where the prediction itself is uncertain.
- The explicit "absent" outcome and the shared variances are what make a non-zero prior mean safe. With a variance for each effect, $V_l \to 0$ would mean "the effect equals the prediction exactly" and not "no effect", which is the degenerate mode Model K hit.

**Shared across genes and estimated:** $\rho_e$, $e$, $\sigma_0^2$, $\sigma_1^2$, $\lambda$ (with a weak prior $\lambda \sim N(0, s_\lambda^2)$), $\pi_g$ and $\tau_g^2$.

## 5. Inference with frozen chromatin fits (default)

**Step 1.** Fit each peak's caQTL with SuSiE and the links with their spike and slab, each on its own data. Compute $(\rho_j, \mu_j, \omega_j)$. This is what `chromatin_only_expression_prediction.py` does.

**Step 2.** Fit the eQTL layer. For single effect $l$, remove the other effects:

$$\tilde z_j = z_j - \sum_i R_{ji}\,\frac{\bar\beta_i - \bar\beta_{li}}{s_i}.$$

For each variant and each slab $c \in \{0, 1\}$, with $(M_0, T_0) = (0, \sigma_0^2)$ and $(M_{1j}, T_{1j}) = (\lambda\mu_j,\ \sigma_1^2 + \lambda^2\omega_j)$:

$$V_{cj} = \Big(\frac{1}{s_j^2} + \frac{1}{T_{cj}}\Big)^{-1}, \qquad A_{cj} = V_{cj}\Big(\frac{\tilde z_j}{s_j} + \frac{M_{cj}}{T_{cj}}\Big), \qquad \log BF_{cj} = \frac12\log\frac{V_{cj}}{T_{cj}} + \frac{A_{cj}^2}{2V_{cj}} - \frac{M_{cj}^2}{2T_{cj}}.$$

The posterior over outcomes is one softmax:

$$\alpha_{l0} \propto 1 - \rho_e, \qquad \alpha^0_{lj} \propto \frac{\rho_e(1 - \rho_j)}{Z}\,e^{\log BF_{0j}}, \qquad \alpha^1_{lj} \propto \frac{\rho_e\,e\,\rho_j}{Z}\,e^{\log BF_{1j}},$$

and the moments passed to the other effects are

$$\bar\beta_{lj} = \alpha^0_{lj}A_{0j} + \alpha^1_{lj}A_{1j}, \qquad E[\theta_l^2 1(\gamma_l = j)] = \alpha^0_{lj}(A_{0j}^2 + V_{0j}) + \alpha^1_{lj}(A_{1j}^2 + V_{1j}).$$

Given the other effects, this update is exact: nothing is factorized between an effect's location, its path status and its size.

**Step 3.** Shared parameters, pooled over genes:

$$\rho_e = \frac{\sum_{g,l}(1 - \alpha_{l0})}{\sum_g L}, \qquad \sigma_0^2 = \frac{\sum \alpha^0_{lj}(A_{0j}^2 + V_{0j})}{\sum\alpha^0_{lj}}.$$

$e$ maximizes $\sum_g\big[\, r_g\log e - n_g\log(p_g - P_g + e P_g)\big]$, with $r_g = \sum_{l,j}\alpha^1_{lj}$, $n_g = \sum_l(1 - \alpha_{l0})$, $p_g$ the number of variants and $P_g = \sum_j\rho_j$ (one-dimensional).

$\lambda$ and $\sigma_1^2$ maximize

$$-\frac{\lambda^2}{2s_\lambda^2} + \sum_{g,l,j}\alpha^1_{lj}\Big[-\frac12\log\big(\sigma_1^2 + \lambda^2\omega_j\big) - \frac{(A_{1j} - \lambda\mu_j)^2 + V_{1j}}{2(\sigma_1^2 + \lambda^2\omega_j)}\Big],$$

a smooth two-parameter problem solved numerically from the previous values, as in `chromatin_informed_expression_prediction.py`.

**Loop.** Start from $e = 1$, $\lambda = 0$, $\sigma_1^2 = \sigma_0^2$. Alternate step 2 over all genes with step 3 until the shared parameters change by less than a tolerance.

## 6. Outputs

- eQTL PIP of variant $j$: $1 - \prod_l\big(1 - \alpha^0_{lj} - \alpha^1_{lj}\big)$. Credible sets from each effect's non-null mass, with the usual purity filter.
- Probability that variant $j$ carries an eQTL effect that follows the chromatin prediction: $1 - \prod_l(1 - \alpha^1_{lj})$.
- caQTL PIPs and credible sets for each peak, and the link posteriors.
- $e$, $\lambda$, $\sigma_1^2$ against $\sigma_0^2$, and $\rho_e$. If chromatin is uninformative, $e \approx 1$, $\lambda \approx 0$ and $\sigma_1^2 \approx \sigma_0^2$, and the eQTL layer is SuSiE with a null outcome on the eQTL data alone.

## 7. Optional: feedback into the caQTL and link fits

With feedback on, the path slab is written $\theta \sim N(\lambda m_j, \sigma_1^2)$ with $m_j$ latent, and its expected log density adds terms to the other two layers. Let $w^1_j = \sum_l\alpha^1_{lj}$ and $\bar\beta^1_j = \sum_l\alpha^1_{lj}A_{1j}$, and let $\bar m_{-k,j}$ be the predicted effect from the other peaks.

**Link $k$** gains a precision and a linear term:

$$A_k = \frac{\lambda^2}{\sigma_1^2}\sum_j w^1_j\,E[b_{kj}^2], \qquad B_k = \frac{1}{\sigma_1^2}\sum_j E[b_{kj}]\big(\lambda\bar\beta^1_j - \lambda^2 w^1_j\,\bar m_{-k,j}\big),$$

$$\frac{1}{t_k^2} = \frac{1}{se_k^2} + \frac{1}{\tau_g^2} + A_k, \qquad \nu_k = t_k^2\Big(\frac{\hat g_k}{se_k^2} + B_k\Big).$$

**caQTL single effect $l'$ of peak $k$** gains one Gaussian pseudo-observation per variant:

$$C_{kj} = \frac{\lambda^2 E[g_k^2]\,w^1_j}{\sigma_1^2}, \qquad D_{kl'j} = \frac{1}{\sigma_1^2}\Big[\lambda E[g_k]\bar\beta^1_j - \lambda^2 w^1_j\Big(E[g_k^2]\,\big(E[b_{kj}] - \bar b_{kl'j}\big) + E[g_k]\,\bar m_{-k,j}\Big)\Big],$$

added to the precision and to the linear term of that variant in the caQTL single-effect update. After each sweep, $(\rho_j, \mu_j, \omega_j)$ are recomputed from the updated posteriors.

Both sets of terms vanish where no eQTL effect sits on a path variant ($w^1_j \approx 0$), so feedback acts only where an eQTL signal, a caQTL signal and a supported link coincide. With feedback on, the updates no longer ascend one exact objective, because step 2 integrates over $m_j$ and these terms do not. Turn it on only after the frozen version works, and check that it changes the caQTL credible sets for the better.

## 8. Alternative for the path slab: a scale-free mean with a variance for each effect

Shared slab variances in raw expression units are a compromise (issue 1 below). The alternative keeps standard SuSiE's variance for each effect and ties the prior mean to it:

$$\theta_l \mid \gamma_l = j \sim N\big(c\,\sqrt{V_l}\;d_j,\; V_l\big), \qquad P(\gamma_l = j) \propto 1 + (e - 1)\rho_j,$$

with $d_j \in [-1, 1]$ the confidence-weighted predicted direction, $P(m_j > 0) - P(m_j < 0)$, and $c$ learned across genes. The mean shrinks with $V_l$, so $V_l \to 0$ is a null effect and no explicit null outcome is needed. This uses the direction of the prediction and not its size. It is the better choice if effect sizes turn out not to line up across genes under the current standardization.

## 9. Known issues and open decisions

1. **Shared slab variances.** eQTL signals differ widely in size within and across genes, and expression is unstandardized. SuSiE's PIPs are fairly insensitive to the prior variance when allocating a strong signal among LD partners, but a shared variance can cost power and calibration for weak secondary signals. Options: a small grid of scale factors for each effect; working in z-score units; or the alternative of Section 8.
2. **Two effects on one variant.** If two eQTL single effects choose the same path variant, the prior mean is counted twice. Rare; check the fitted $\alpha$ for it.
3. **Shared donors.** Expression and accessibility come from the same people, so noise in the eQTL and caQTL estimates at a variant is correlated, with the sign of the link. This can inflate $e$ and the apparent agreement with the prediction at weak caQTLs. Stage 1's SuSiE passes only strong caQTL signals, which limits it. Check $e$ and $\sigma_1^2$ by caQTL strength, and simulate with correlated noise. The principled remedy is a joint likelihood with a residual correlation.
4. **Where the gain can come from.** Only loci where the caQTL is better resolved than the eQTL. First check: at loci where an eQTL credible set overlaps a caQTL credible set of a linked peak, compare credible-set sizes and lead z-scores.
5. **Links are not mediation.** A high $e$ partly reflects that caQTLs and eQTLs both concentrate in regulatory DNA. No whole-profile test of mediation is made: each path variant's effect is tied to the prediction separately. The element models (E, I, J) do test the whole caQTL profile of a peak against the eQTL signal.
6. **Co-accessible peaks.** $\rho_j$ uses "at least one path" and is not inflated by correlated peaks, but $m_j$ sums over peaks and is.
7. **Peaks shared between genes.** Each gene currently gets its own copy of a peak's caQTL statistics. With frozen chromatin fits this costs only computing time. With feedback it would let different genes pull the same peak's caQTL fit in different directions.
8. **Choice of $L$ and $L_2$.** With $\rho_e$ learned, results are stable once $L$ is comfortably above the number of signals (tested from $L = 5$ to $40$ with $L\rho_e$ fixed). Use 10 and 5, and rerun a subset at double.

## 10. Validation plan

1. With $e = 1$, $\lambda = 0$ and $\sigma_1^2 = \sigma_0^2$, confirm the eQTL layer equals a SuSiE-with-null fit of the eQTL data alone, and compare its PIPs with `SUSIE_RSS`.
2. Simulate eQTL summary statistics from the real LD matrices and the real $(\rho_j, \mu_j, \omega_j)$, with causal variants placed at a known $e$, $\lambda$ and $\sigma_1^2$, including the uninformative case. Check PIP calibration, credible-set coverage and size, and recovery of the shared parameters. PIPs must stay calibrated when chromatin is uninformative.
3. The same with noise correlated between expression and accessibility, to bound issue 3.
4. Real data against eQTL-only SuSiE, within genes that have a chromatin path: fraction of credible sets that shrink, number of variants with PIP above 0.9, enrichment of high-PIP variants in functional readouts that do not come from open chromatin, and replication in independently fine-mapped eQTLs.
5. Feedback on against feedback off, on the same genes.
