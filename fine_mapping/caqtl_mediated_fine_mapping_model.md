# caQTL-mediated eQTL fine-mapping model

Joint fine-mapping of a gene's eQTL together with the caQTLs of its linked peaks, fit across genes. Implemented in `ca_qtl_mediated_susie.py` (class `CAQTL_MEDIATED_SUSIE`).

## Notation (one gene)

| Symbol | Meaning |
|---|---|
| $n_E$, $n_A$ | eQTL and caQTL sample sizes |
| $p$ | number of cis SNPs |
| $K$ | number of peaks linked to the gene |
| $X$ | $n \times p$ standardized genotype matrix |
| $E$ | gene expression ($n_E$) |
| $A_k$ | chromatin accessibility of peak $k$ ($n_A$) |
| $L_{ca}$, $L_e$ | number of SuSiE single effects for each caQTL and for the eQTL |
| $\hat g_k$, $se_k$ | peak-gene link estimate and its standard error (hurdle model) |

## Per-gene generative model

caQTL of each peak $k$:

$$A_k = X b_k + e_k, \qquad b_k = \sum_{l=1}^{L_{ca}} b_{kl}, \qquad e_k \sim N(0,\ \sigma^2_{A_k} I)$$

eQTL, with a direct component and a component mediated through the peaks:

$$E = X\Big(\sum_{k=1}^{K} g_k\, b_k + d\Big) + e, \qquad d = \sum_{m=1}^{L_{e}} d_m, \qquad e \sim N(0,\ \sigma^2_E I)$$

Every $b_{kl}$ and $d_m$ is a SuSiE single effect: exactly one SNP $j$ is chosen with prior weight $1/p$ and carries an effect $\sim N(0, V)$, with $V$ estimated per single effect. The total eQTL effect of SNP $j$ is therefore

$$\beta_j = d_j + \sum_k g_k\, b_{kj},$$

a direct effect plus a mediated effect: SNP $j$ changes accessibility of peak $k$ by $b_{kj}$, and peak $k$ changes expression by $g_k$.

## Peak-gene link term

The hurdle estimate from the peak-gene linking analysis is a noisy, rescaled measurement of $g_k$:

$$\hat g_k \sim N(\lambda\, g_k,\ se_k^2 + \tau_u^2)$$

- $\lambda$ converts the model's $g$ scale to the hurdle scale. In z-score mode it also absorbs $\sqrt{n_E / n_A}$ and the phenotype SD ratio.
- $\tau_u^2$ is extra variance allowing the hurdle estimate to be biased relative to the true mediating effect ($\tau_u^2 = 0$ means the hurdle estimate is an unbiased noisy measurement of $\lambda g_k$).

## Spike-and-slab prior on links

$$g_k = z_k h_k, \qquad z_k \sim \text{Bern}(\pi), \qquad h_k \sim N(0,\ \tau_g^2)$$

Each link is either null ($z_k = 0$) or real, with effect size governed by $\tau_g^2$.

## Shared vs. gene-specific parameters

| Shared across genes (EM, or fixed by flag) | Gene specific |
|---|---|
| $\pi$ (link prior probability) | single-effect prior variances $V$ for each $b_{kl}$, $d_m$ |
| $\tau_g^2$ (link prior variance) | residual variances $\sigma^2_E$, $\sigma^2_{A_k}$ |
| $\lambda$ (link scale) | |
| $\tau_u^2$ (link bias variance) | |

The runner flags `--no_estimate_link_scale` and `--no_estimate_link_bias_variance` fix $\lambda$ at `--link_scale_init` and $\tau_u^2$ at 0.

## Summary-statistics form

The likelihoods depend on the data only through $X'X$, $X'E$ and $X'A_k$.

- With sample sizes given: reconstructed from marginal effects, SEs and LD as in SuSiE-RSS.
- Z-score mode (sample sizes `None`, what the real-data pipeline uses): $X'X = R$ (LD), $X'y = z = \hat\beta / se$ for each trait, residual variances fixed at 1 (Zou et al. 2022). Effects are then on the z scale (standardized effect $\times \sqrt{n}$), which is why $\lambda$ is estimated from the data rather than set to a value from the effect-size scale.
- Z-score mode with residual variances (`--estimate_residual_variance` in the runner): $z \sim N(Rb,\ \sigma^2 R)$ per trait per gene, with $y'y = z'R^{-1}z$ and $n = p$ so that the Gaussian log likelihood is exact up to a constant and $\sigma^2 = E[\mathrm{RSS}]/p$ is the maximum likelihood update. $\sigma^2 > 1$ means the z-scores are overdispersed relative to the LD null (polygenic background, LD mismatch). Per-gene estimates are written to the gene convergence file.

## Inference

Mean-field variational inference on the ELBO. Every single effect ($b_{kl}$, $d_m$) and every link $g_k$ has a closed-form conditional posterior, so the fit is coordinate ascent:

1. For each gene: a generalized IBSS pass over the caQTL single effects and the eQTL single effects, then closed-form updates of the link posteriors $q(g_k)$ with $q_k = P(z_k = 1)$, mean $m_k$, variance $v_k$.
2. Global EM updates of $\pi$, $\tau_g^2$ (and optionally $\lambda$, $\tau_u^2$) from the pooled link posteriors.
3. Repeat until the ELBO change falls below tolerance.

The posterior on $g_k$ pools two sources of evidence:

- the link term, with precision $\lambda^2 / (se_k^2 + \tau_u^2)$;
- the eQTL residual regressed on the posterior caQTL effect $X b_k$.

## Outputs

- eQTL PIP per SNP: the posterior probability that it affects expression either directly (some $d_m$) or through a linked peak (some $b_{kl}$ with $q_k > 0$).
- With component output on: direct and mediated PIPs separately, and the per-link posteriors $q_k, m_k, v_k$.
- Shared hyperparameters $\pi, \tau_g^2, \lambda, \tau_u^2$, plus per-gene convergence diagnostics.

Genes with $K = 0$ reduce exactly to eQTL-only SuSiE.

## Reading the log line

`outer iter t: ELBO ..., pi ..., tau2_g ..., lambda ..., tau2_u ...` reports the total ELBO across genes and the four shared hyperparameters after outer iteration $t$. Only ELBO differences between iterations are meaningful. $\tau_g^2$ at its floor ($10^{-8}$) with $\pi$ stuck at its initial value indicates the collapsed solution in which links carry no information into the eQTL model.

## Variants

- `caqtl_mediated_fine_mapping_model_proposal_spurious_links.md` (Model C, implemented then replaced by Model D): the link term becomes a mixture in which, with probability $\rho$, the hurdle estimate is unrelated to $g_k$; bounds the evidence a hurdle link can supply.
- `caqtl_mediated_fine_mapping_model_proposal_link_prior.md` (Model B, proposal): hurdle z-scores set only the prior inclusion probability of each link; no $\lambda$.
- `caqtl_mediated_fine_mapping_model_proposal_inclusion.md` (Model D, implemented, runner flag `--link_inclusion`): Model A plus a per-link indicator $u_k$ for whether the link enters the eQTL model; the hurdle sets the implied effect, the eQTL decides inclusion.
- `caqtl_mediated_fine_mapping_model_proposal_elements.md` (Model E, proposal): one SuSiE over variants plus each peak's genetic component $X \hat b_k$ as competing elements, with learned per-class prior weights; hurdle gives sign (and optionally prior weight), no $\lambda$. cTWAS at the peak-gene level.
- `caqtl_mediated_fine_mapping_model_proposal_soft_tie.md` (Model F, proposal): Model E with a selected peak's coefficient drawn around the link value, $\beta \sim N(g_k, \tau_\beta^2)$; removes peak stacking and makes $\lambda$ learnable from the selected peaks.
- `caqtl_mediated_fine_mapping_model_proposal_sign.md` (Model G, proposal): E-free with a sign-weighted two-sided slab on the peak coefficient, $\Phi(t_k) N^+ + (1 - \Phi(t_k)) N^-$; the hurdle contributes direction and gating only, no $\lambda$.
- `caqtl_mediated_fine_mapping_model_proposal_sign_gate.md` (Model H, proposal): E-free with a per-peak sign gate on the pool, comparing the hurdle's direction with the marginal eQTL/caQTL direction at the peak's caQTL lead; the hurdle contributes link existence and direction only.
- `caqtl_mediated_fine_mapping_model_proposal_ctwas_style.md` (Model I, proposal): cTWAS-style, peak profiles fixed from caQTL-only fine-mapping, free coefficients, hurdle-strength bins with learned prior weights, and the hurdle sign either as an oriented predictor with a half-normal coefficient (I-a) or as a pool gate (I-b). Cheap; no $\lambda$; no feedback into the caQTL.
- `caqtl_mediated_fine_mapping_model_proposal_ctwas_groups.md` (Model J, implemented, runner `run_ca_qtl_ctwas_eqtl_finemapping.py`): cTWAS on peaks with standardized fixed peak predictors, free effects, and group priors by hurdle significance and sign concordance learned by EM.
- `caqtl_mediated_fine_mapping_model_proposal_signed_mean.md` (Model K, proposal): Model J with a per-bin signed prior mean $\mu_b\,\text{sign}(\hat g_k)$ (optionally a per-bin variance) on peak effects; the sign becomes a learned soft prior on the effect's direction, the size a learned per-bin effect, no $\lambda$.
- `caqtl_mediated_fine_mapping_model_proposal_predicted_mean.md` (Model L, implemented, runner `run_ca_qtl_predicted_mean_eqtl_finemapping.py`): three spike-and-slab layers, variant-to-peak effects $b_{kj}$ (caQTL data), peak-to-gene links $g_{kg}$ (hurdle data) and causal eQTL effects with slab $N(\lambda \sum_k b_{kj} g_{kg}, \sigma^2)$, $\sigma^2$ shared across genes; the eQTL posterior is computed exactly by marginalizing the caQTL and link posteriors inside the eQTL fit (mixture-of-means slab), with an optional leave-one-gene-out feedback scheme; no peak elements, $\lambda$ a regression slope.
