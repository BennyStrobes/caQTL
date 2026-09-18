# Joint fine-mapping of eQTLs and caQTLs: proposed SuSiE-based model

Ben Strober, 2026-09-18

## 1. Summary

This is a proposal, not yet implemented. It is the SuSiE-based version of the model in `joint_eqtl_caqtl_expression_prediction.py` (that file is now used for expression prediction).

The model has three layers of latent effects for each gene:

1. variant-to-peak effects, informed by caQTL summary statistics;
2. peak-to-gene link effects, informed by the estimated link effects;
3. causal eQTL effects, whose prior is centred on the effect predicted through chromatin (caQTL effect times link effect, summed over peaks, times a scale $\lambda$).

All three layers are fit jointly by variational inference, once across all genes, with shared hyperparameters.

The change from the current code is the variational family for the eQTL and caQTL layers. The current code gives every variant its own independent inclusion probability. Under tight LD that family cannot represent "one of these variants is causal, and I do not know which", so it puts probability 1 on an arbitrary member of an LD block (Section 2). Here each layer is a sum of single effects, and each single effect has one categorical distribution over variants plus an explicit "no effect" outcome.

## 2. Why not one inclusion probability per variant

Simulation: 500 samples, 60 variants, one causal variant (variant 22) inside a block of four near-identical variants (20 to 23), no chromatin information. The fully factorized update had converged after one sweep and was unchanged after 2,000 sweeps.

| $r$ within block | factorized, variants 20 to 23 | SuSiE PIPs, variants 20 to 23 |
|---|---|---|
| 1.00 | 1.00, 0.01, 0.01, 0.01 | 0.29, 0.29, 0.29, 0.29 |
| 0.99 | 1.00, 0.02, 0.01, 0.02 | 0.60, 0.21, 0.28, 0.09 |
| 0.95 | 1.00, 0.03, 0.03, 0.03 | 0.02, 0.42, 0.44, 0.13 |

The winner is whichever variant is updated first. Starting from the even split (0.25 each) collapses to a single variant within 10 sweeps. Learning the prior inclusion probability makes the collapse sharper. This is a property of the factorized family, not of the optimizer: four independent indicators at 0.25 each put 32% of their mass on "no causal variant in the block" and about 26% on "two or more", and both fit the data badly.

## 3. Data and scale

For gene $g$ with $p$ cis variants and LD matrix $R$ ($p \times p$, the same matrix for both modalities):

- eQTL marginal effects $\hat\beta \in \mathbb{R}^p$ with standard errors $s \in \mathbb{R}^p$;
- for each peak $k = 1, \dots, K$ linked to the gene, caQTL marginal effects $\hat b_k \in \mathbb{R}^p$ with standard errors $s_k \in \mathbb{R}^p$, over the same variants;
- for each peak $k$, the estimated peak-to-gene link effect $\hat g_k$ with standard error $se_k$.

Genotypes are standardized and expression is not. All eQTL effects below, observed and latent, are per standard deviation of genotype in raw expression units. caQTL effects are per standard deviation of genotype in the units of the caQTL summary statistics. The scale parameter $\lambda$ converts (caQTL units $\times$ link units) into eQTL units, so it must be free.

Write $S = \text{diag}(s)$, $S_k = \text{diag}(s_k)$, $z = S^{-1}\hat\beta$ and $z_k = S_k^{-1}\hat b_k$.

## 4. Model

### Likelihoods

Summary-statistic (RSS) likelihoods, which need no sample size:

$$\hat\beta \mid \beta \sim N\big(S R S^{-1} \beta,\; S R S\big), \qquad \hat b_k \mid b_k \sim N\big(S_k R S_k^{-1} b_k,\; S_k R S_k\big), \qquad \hat g_k \mid g_k \sim N\big(g_k,\; se_k^2\big).$$

Up to a constant, the eQTL log likelihood is

$$\log p(\hat\beta \mid \beta) = \sum_j \frac{\hat\beta_j \beta_j}{s_j^2} - \frac12 \sum_{i,j} \frac{R_{ij}}{s_i s_j}\beta_i\beta_j ,$$

so $R^{-1}$ is never needed. The caQTL log likelihood has the same form.

### Single effect with a null outcome

Both the eQTL and the caQTL layers are built from the same object. A single effect is a pair $(\gamma, \theta)$:

$$P(\gamma = 0) = 1 - \rho, \qquad P(\gamma = j) = \rho / p \quad (j = 1, \dots, p), \qquad \theta \mid \gamma = j \sim N(M_j, \sigma^2).$$

$\gamma = 0$ means the effect is absent and contributes nothing. $\gamma = j$ means the effect sits on variant $j$ with size $\theta$. $\rho$ is the prior probability that the effect exists, and $M_j$ is the prior mean at variant $j$.

The null outcome replaces the usual SuSiE device of estimating a prior variance $V_l$ for each effect and letting $V_l \to 0$ switch the effect off. With a non-zero prior mean, $V_l \to 0$ does not mean "no effect". It means "the effect equals $M_j$ exactly", which is the degenerate mode that Model K hit. So: an explicit null outcome, one slab variance shared across effects and genes, and no per-effect variances.

### eQTL layer

$$\beta_j = \sum_{l=1}^{L} \theta_l \, 1[\gamma_l = j], \qquad (\gamma_l, \theta_l) \text{ single effects with parameters } (\rho_e,\; M_j = \lambda m_j,\; \sigma_e^2),$$

$$m_j = \sum_{k=1}^{K} g_k \, b_{kj}.$$

$m_j$ is the effect of variant $j$ on the gene predicted through chromatin. It is zero unless $j$ is a caQTL variant for a linked peak. A gene with no peaks has $m_j = 0$ everywhere, and the layer is ordinary SuSiE with a null outcome.

### caQTL layer

For each peak $k$:

$$b_{kj} = \sum_{l'=1}^{L_2} \phi_{kl'} \, 1[\delta_{kl'} = j], \qquad (\delta_{kl'}, \phi_{kl'}) \text{ single effects with parameters } (\rho_c,\; M_j = 0,\; \sigma_c^2).$$

### Link layer

$$u_k \sim \text{Bernoulli}(\pi_g), \qquad g_k \mid u_k = 1 \sim N(0, \tau_g^2), \qquad g_k = 0 \text{ if } u_k = 0.$$

There is no LD between peaks in the link likelihood, so an ordinary spike and slab is adequate here.

### Shared parameters

$\lambda \sim N(0, s_\lambda^2)$ with large $s_\lambda$, shared across all genes. The hyperparameters $(\rho_e, \sigma_e^2, \rho_c, \sigma_c^2, \pi_g, \tau_g^2)$ are shared across all genes and estimated by maximizing the ELBO. The implied prior inclusion probability per variant is about $L \rho_e / p$.

## 5. Variational family

$$q = q(\lambda) \prod_{l} q(\gamma_l, \theta_l) \prod_{k} \Big[ q(u_k, g_k) \prod_{l'} q(\delta_{kl'}, \phi_{kl'}) \Big],$$

with

- $q(\gamma_l = j) = \alpha_{lj}$ for $j = 0, \dots, p$, and $q(\theta_l \mid \gamma_l = j) = N(\mu_{lj}, v_{lj})$;
- $q(\delta_{kl'} = j) = \alpha^c_{kl'j}$, and $q(\phi_{kl'} \mid \delta_{kl'} = j) = N(\mu^c_{kl'j}, v^c_{kl'j})$;
- $q(u_k = 1) = a_k$, and $q(g_k \mid u_k = 1) = N(\nu_k, t_k^2)$;
- $q(\lambda) = N(\mu_\lambda, v_\lambda)$.

Moments used below (independence across single effects, peaks and $\lambda$):

$$\bar\beta_{lj} = \alpha_{lj}\mu_{lj}, \qquad \bar\beta_j = \sum_l \bar\beta_{lj}, \qquad w_j = \sum_l \alpha_{lj},$$

$$\bar b_{kj} = \sum_{l'} \alpha^c_{kl'j}\mu^c_{kl'j}, \qquad E[b_{kj}^2] = \bar b_{kj}^2 + \sum_{l'} \Big[ \alpha^c_{kl'j}\big((\mu^c_{kl'j})^2 + v^c_{kl'j}\big) - \big(\alpha^c_{kl'j}\mu^c_{kl'j}\big)^2 \Big],$$

$$\bar g_k = a_k \nu_k, \qquad E[g_k^2] = a_k(\nu_k^2 + t_k^2), \qquad E[\lambda^2] = \mu_\lambda^2 + v_\lambda,$$

$$\bar m_j = \sum_k \bar g_k \bar b_{kj}, \qquad E[m_j^2] = \bar m_j^2 + \sum_k \Big( E[g_k^2]\,E[b_{kj}^2] - \bar g_k^2 \bar b_{kj}^2 \Big).$$

$w_j$ is the expected number of eQTL effects on variant $j$. It is the weight with which the eQTL layer feeds back into the other two layers.

## 6. Coordinate updates

### eQTL single effect $l$

Remove every other eQTL effect from the data:

$$\tilde z_j = z_j - \sum_i R_{ji} \frac{\bar\beta_i - \bar\beta_{li}}{s_i}.$$

Then for each variant $j = 1, \dots, p$ (vectorized over $j$):

$$\frac{1}{v_{lj}} = \frac{1}{s_j^2} + \frac{1}{\sigma_e^2}, \qquad \mu_{lj} = v_{lj}\left( \frac{\tilde z_j}{s_j} + \frac{\mu_\lambda \bar m_j}{\sigma_e^2} \right),$$

$$\log BF_{lj} = \frac12 \log\frac{v_{lj}}{\sigma_e^2} + \frac{\mu_{lj}^2}{2 v_{lj}} - \frac{E[\lambda^2]\,E[m_j^2]}{2\sigma_e^2},$$

$$\alpha_{l0} \propto 1 - \rho_e, \qquad \alpha_{lj} \propto \frac{\rho_e}{p}\exp(\log BF_{lj}).$$

The posterior mean, the variance and the log Bayes factor are what `update_eqtl_effects` already computes. The differences are the residual (other effects removed, not other variants) and the last line (one softmax over $p + 1$ outcomes, not $p$ independent sigmoids).

The last term of the Bayes factor uses $E[M_j^2]$ and not $E[M_j]^2$. It is how uncertainty in the caQTL effects, the links and $\lambda$ discounts the prediction. Between two variants in perfect LD, the eQTL data contribute identically, and the entire difference in $\alpha_{lj}$ comes from the chromatin terms.

### Link $k$

Let $\bar m_{-k,j} = \bar m_j - \bar g_k \bar b_{kj}$. The eQTL prior contributes a quadratic and a linear term in $g_k$:

$$A_k = \frac{E[\lambda^2]}{\sigma_e^2}\sum_j w_j\, E[b_{kj}^2], \qquad B_k = \frac{1}{\sigma_e^2}\sum_j \bar b_{kj}\Big( \mu_\lambda \bar\beta_j - E[\lambda^2]\, w_j\, \bar m_{-k,j} \Big).$$

$$\frac{1}{t_k^2} = \frac{1}{se_k^2} + \frac{1}{\tau_g^2} + A_k, \qquad \nu_k = t_k^2\left( \frac{\hat g_k}{se_k^2} + B_k \right), \qquad \text{logit}(a_k) = \text{logit}(\pi_g) + \frac12\log\frac{t_k^2}{\tau_g^2} + \frac{\nu_k^2}{2 t_k^2}.$$

With no eQTL signal near the peak's caQTL variants ($w_j \approx 0$ there), $A_k = B_k = 0$ and the link posterior is the one implied by $\hat g_k$ alone.

### caQTL single effect $l'$ of peak $k$

Remove the peak's other effects from its caQTL data:

$$\tilde z_{kj} = z_{kj} - \sum_i R_{ji}\frac{\bar b_{ki} - \bar b_{kl'i}}{s_{ki}}, \qquad \bar b_{kl'j} = \alpha^c_{kl'j}\mu^c_{kl'j}.$$

The eQTL prior acts as one extra Gaussian pseudo-observation per variant:

$$C_{kj} = \frac{E[\lambda^2]\, E[g_k^2]\, w_j}{\sigma_e^2}, \qquad D_{kl'j} = \frac{1}{\sigma_e^2}\Big[ \mu_\lambda \bar g_k \bar\beta_j - E[\lambda^2]\, w_j \Big( E[g_k^2]\,(\bar b_{kj} - \bar b_{kl'j}) + \bar g_k\, \bar m_{-k,j} \Big) \Big].$$

$$\frac{1}{v^c_{kl'j}} = \frac{1}{s_{kj}^2} + \frac{1}{\sigma_c^2} + C_{kj}, \qquad \mu^c_{kl'j} = v^c_{kl'j}\left( \frac{\tilde z_{kj}}{s_{kj}} + D_{kl'j} \right),$$

$$\log BF^c_{kl'j} = \frac12\log\frac{v^c_{kl'j}}{\sigma_c^2} + \frac{(\mu^c_{kl'j})^2}{2 v^c_{kl'j}}, \qquad \alpha^c_{kl'0} \propto 1 - \rho_c, \qquad \alpha^c_{kl'j} \propto \frac{\rho_c}{p}\exp(\log BF^c_{kl'j}).$$

When the link is off ($a_k \approx 0$) or no eQTL effect is near ($w_j \approx 0$), $C$ and $D$ vanish and this is ordinary SuSiE on the caQTL data. The feedback matters only where an eQTL signal, a caQTL signal and a supported link coincide. There it pulls the caQTL signal and the eQTL signal toward the same member of an LD block.

### Scale $\lambda$

Summing over all genes $g$ and variants $j$:

$$\frac{1}{v_\lambda} = \frac{1}{s_\lambda^2} + \frac{1}{\sigma_e^2}\sum_{g,j} w_{gj}\,E[m_{gj}^2], \qquad \mu_\lambda = \frac{v_\lambda}{\sigma_e^2}\sum_{g,j}\bar m_{gj}\,\bar\beta_{gj}.$$

### Hyperparameters

Closed-form maximizers of the ELBO, pooled over all genes:

$$\sigma_e^2 = \frac{\sum_{g,l}\sum_{j \ge 1}\alpha_{lj}\Big(\mu_{lj}^2 + v_{lj} - 2\mu_{lj}\,\mu_\lambda\bar m_j + E[\lambda^2]E[m_j^2]\Big)}{\sum_{g,l}(1 - \alpha_{l0})}, \qquad \rho_e = \frac{\sum_{g,l}(1 - \alpha_{l0})}{\sum_g L},$$

$$\sigma_c^2 = \frac{\sum_{k,l'}\sum_{j\ge1}\alpha^c_{kl'j}\big((\mu^c_{kl'j})^2 + v^c_{kl'j}\big)}{\sum_{k,l'}(1-\alpha^c_{kl'0})}, \qquad \rho_c = \frac{\sum_{k,l'}(1-\alpha^c_{kl'0})}{\sum_k L_2}, \qquad \tau_g^2 = \frac{\sum_k a_k(\nu_k^2 + t_k^2)}{\sum_k a_k}, \qquad \pi_g = \frac{\sum_k a_k}{\sum_g K_g}.$$

Put a floor on each variance.

## 7. Algorithm

1. **Warm start (no feedback).** Fit the caQTL layer of every peak on its own data. Fit the links from $\hat g_k$ alone. Set $\mu_\lambda$ from a moment estimate and $v_\lambda$ small. Fit the eQTL layer given these. This first pass is Model L without the mixture-of-means slab.
2. **Joint sweeps.** For each gene: update the $L_2$ caQTL effects of each peak, then the links, then the $L$ eQTL effects. After all genes: update $\lambda$, then the hyperparameters.
3. **Stop** when the ELBO increases by less than `tol`.
4. **Multi-start per gene.** Coordinate ascent on the earlier joint model (`CAQTL_MEDIATED_SUSIE`) got stuck in poor local optima. Run each gene from two starts (eQTL effects at null; eQTL effects from an eQTL-only SuSiE fit) and keep the higher ELBO.

The warm start matters for a second reason. If the caQTL and link variances start large, $E[m_j^2]$ is large and the last term of the eQTL Bayes factor pushes every $\alpha_{lj}$ to zero on the first sweep.

## 8. Outputs

- Posterior inclusion probability of variant $j$ for the gene: $PIP_j = 1 - \prod_l (1 - \alpha_{lj})$. Likewise per peak for the caQTL layer.
- Credible sets: for each effect with $1 - \alpha_{l0}$ above a threshold, the smallest set of variants holding 95% of the non-null mass, with the usual purity filter.
- Link posteriors $a_k$ and $\bar g_k$.
- Posterior mean eQTL effects $\bar\beta_j$ in raw expression units.
- $\mu_\lambda$, $v_\lambda$ and the hyperparameters.

## 9. Known issues and open decisions

1. **Two effects on one variant.** If two eQTL single effects choose the same variant, the prior mean is counted twice. This is the usual SuSiE overlap and should be rare. Check the fitted $\alpha$ for it.
2. **Mean-field shrinks the prediction.** $\bar b_{kj} = \alpha\mu$ is shrunk toward zero when the caQTL effect is uncertain, and the $E[m_j^2]$ term penalizes the variant further. The exact prior is a mixture: shifted mean if the caQTL is causal there, zero mean otherwise. Model L handled this with a mixture-of-means slab inside the single-effect update (`susie_rss_mixture.py`). The same device fits here, at the cost of a messier feedback term.
3. **Peaks shared between genes.** The current input gives each gene its own copy of a peak's caQTL statistics. A peak near three genes is then fine-mapped three times, and its data are counted three times in $\rho_c$ and $\sigma_c^2$. Options: key the caQTL layer by peak; or keep per-gene copies and weight each by one over the number of genes.
4. **Residual variance.** The RSS likelihood takes the standard errors at face value, so the implied residual variance is the marginal one. For genes whose cis variants explain 20 to 30% of expression this is conservative, and unequal standard errors slightly distort the expected marginal effects of LD partners. Passing the eQTL sample size and estimating a residual variance per gene, as `SUSIE_RSS` does, removes both problems and keeps the same effect scale.
5. **One slab variance or two.** Model L used separate $\sigma_e^2$ for variants with and without a prediction. With a single $\sigma_e^2$, ordinary eQTLs with no chromatin support set its size, and the prediction may carry little weight.
6. **Choice of $L$ and $L_2$.** The null outcome makes unused effects cheap, so moderately generous values (10 and 5) should be safe.
7. **Memory.** The caQTL layer stores $K \times L_2 \times (p + 1)$ values three times per gene. With all genes in memory this can reach tens of GB. Store caQTL effects sparsely, or stream genes.
8. **Imputed caQTL statistics.** The input has an imputation mask and an $r^2$ of imputation for each caQTL cell. They are not used here.

## 10. Validation plan

1. Simulate genes with realistic LD, known mediated and unmediated effects, and known $\lambda$. Check PIP calibration, credible-set coverage and bias in $\lambda$.
2. With uninformative links, confirm that the fit reduces to separate SuSiE fits.
3. On a few dozen real genes, compare against a Gibbs sampler with block moves, several chains, and the hyperparameters fixed at the variational estimates.
