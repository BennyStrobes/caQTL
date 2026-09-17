# Model I: cTWAS-style fine-mapping with fixed peak profiles, sign concordance and hurdle-strength priors (proposal)

Companion to the Model E family (`..._elements.md`, `..._soft_tie.md`, `..._sign.md`, `..._sign_gate.md`). Model I moves closer to cTWAS (Zhao et al. 2024): the peak genetic components are **fixed** from the caQTL-only fine-mapping, exactly as cTWAS fixes imputed expression from its weights, and the eQTL is one SuSiE over variants and those fixed predictors. The hurdle contributes its **sign** and its **significance**, in two variants below. No $\lambda$, no joint refit of the caQTL, no tie between coefficients and hurdle sizes. Not implemented.

## Premise

Everything learned so far points the same way: the hurdle's effect size cannot be lined up with eQTL effects across genes (gene- and peak-specific scales, partial mediation), its sign and its significance can. And the joint refit of the caQTL profiles, the one channel through which expression can bend a peak toward itself, buys eQTL-side sharpening of caQTL credible sets at the price of a feedback loop that has to be watched. cTWAS keeps its exposure predictors fixed for exactly that reason. Model I does the same, and spends the hurdle where it is reliable.

## Model

**Fixed peak profiles.** For each linked peak $k$, fine-map its caQTL alone with SuSiE-RSS (already done as the initialization of Model E). Take the posterior mean effect vector $\hat b_k$ and the caQTL PIPs, and freeze them. The peak's genetic component is the fixed predictor

$$m_k = X \hat b_k,$$

with sufficient statistics $\hat b_k' z_E$ and $\hat b_k' R\, \hat b_k$ computed once per gene. This is cTWAS's imputed expression, with peaks in place of genes and caQTL fine-mapping weights in place of expression-prediction weights.

**eQTL regression over variants and peaks.**

$$E = \sum_{l=1}^{L} \beta_l\, w_{\gamma_l} + e, \qquad w_{\gamma_l} \in \{x_1, \dots, x_p\} \cup \{m_k\}, \qquad \beta_l \sim N(0, V_l) \text{ (variants)}$$

One SuSiE over the augmented pool. Variants have free effects with $V_l$ estimated per single effect. What a peak element's coefficient looks like, and how the hurdle sets its prior weight, define the two variants.

**Hurdle-strength priors (both variants).** cTWAS learns a separate prior inclusion weight per group of elements. Here the groups are the variants and bins of hurdle strength: peaks with $|t_k| = |\hat g_k / se_k|$ in, say, $[0, 2)$, $[2, 4)$ and $[4, \infty)$, each with its own weight $\omega_{p,b}$ learned by EM across genes (posterior mass on the group's elements over the number of elements in the group). $\omega_{p,b} / \omega_v$ is the enrichment of peaks in bin $b$ over variants. This replaces the spike-slab gating of Model E: a null-looking link is not removed from the pool, it sits in the weakest bin, and the data decide how much that bin is worth. It also uses the hurdle's significance in the way cTWAS uses group membership, with the enrichment of strong links over weak ones as a direct readout of whether the hurdle carries information.

### Variant I-a: hard sign through the predictor

Orient each peak's predictor by the hurdle sign, $\tilde m_k = \text{sign}(\hat g_k)\, m_k$, and give the peak coefficient a half-normal prior on the positive side, $\beta_l \sim N^{+}(0, V_l)$. "Opening the peak changes expression in the direction the hurdle says" is then a hard constraint: a discordant peak has a coefficient posterior pushed against zero and a Bayes factor that collapses, so it loses to the variant. The single-effect update for a peak is a truncated normal (closed form via the Mills ratio; the Bayes factor is the ordinary SER Bayes factor times $2\Phi(\mu/\sqrt{\nu})$). Weak-sign links are handled by the bins rather than by softening the constraint: a link in the $[0, 2)$ bin has a small learned weight, so a wrong sign there costs little.

### Variant I-b: sign gate on the pool

Keep free, symmetric coefficients for peaks (plain cTWAS), and apply Model H's gate: a peak whose hurdle sign disagrees with the marginal direction at its caQTL lead, when both directions are clear, gets pool weight zero for that gene. No truncated normals; the sign acts once, before fitting, on marginal statistics. Coarser than I-a (one variant, one threshold, a mild double use of the eQTL z at the lead) but nothing new in the SER.

**Effect sizes** are free in both variants and never compared with $\hat g_k$.

## What is given up relative to Model E

- **No eQTL-side sharpening of caQTL credible sets.** The mediated PIP of a variant is $P_k \times$ (caQTL-only PIP), composed from the frozen caQTL fine-mapping. This is the cTWAS trade: the exposure model is fit once from its own data.
- **No hurdle magnitude.** By design.
- **Diffuse caQTLs are weaker predictors** only through $\hat b_k$ being spread; the posterior uncertainty of $b_k$ no longer enters. (It could, by using $E[b_k' R\, b_k]$ from the caQTL fit as the precision, at no cost; worth doing.)

## What is gained

- **Cost.** No caQTL refit inside the loop: per gene, one SuSiE with $K$ extra columns whose sufficient statistics are precomputed. Roughly the cost of eQTL-only fine-mapping, far below Model E.
- **No feedback loop, no stacking, exact bookkeeping, no calibration parameters.** The only learned quantities beyond SuSiE's are the group weights.
- **A clean readout of what the hurdle is worth.** The enrichment of the strong-link bin over the weak-link bin, and the fraction of discordant peaks that would otherwise have been selected, are the two numbers that say whether the hurdle links carry information about mediation. Neither depends on a scale.
- **Sign concordance used where it matters** with its uncertainty handled by binning (I-a) or by requiring clear marginal directions (I-b).

## Inference

Standard SuSiE-RSS coordinate ascent over the augmented pool, z-score mode, with per-group prior weights; the on/off rule of Model E for single effects; EM for $\omega_v$ and the $\omega_{p,b}$ across genes (posterior mass per group over element count per group, over on effects). I-a adds the truncated-normal peak posterior; I-b adds the gate at initialization. Nothing else iterates: no link factor, no pool refresh, no $\rho_p$ search. Convergence is SuSiE's ELBO test plus the group weights settling.

## Outputs

Per variant: direct, mediated and combined PIPs as in E. Per peak: $P_k$ = P(some on effect selects $k$), the fitted coefficient, its bin, and (I-a) the posterior sign-agreement probability or (I-b) the gate value. Shared: the group weights and their enrichments over variants.

## Relation to the other models

| | E (tied) | E-free | G | H | I-a | I-b |
|---|---|---|---|---|---|---|
| caQTL profile | jointly refit | jointly refit | jointly refit | jointly refit | fixed | fixed |
| hurdle size | hard | unused | unused | unused | unused | unused |
| hurdle sign | via size | unused | prior mixture, $\Phi(t_k)$ | pool gate | oriented predictor + half-normal | pool gate |
| hurdle significance | spike gates pool | spike gates pool | spike gates pool | spike gates pool | learned weight per $\lvert t_k \rvert$ bin | learned weight per $\lvert t_k \rvert$ bin |
| needs $\lambda$ | fixed | no | no | no | no | no |
| eQTL sharpens caQTL sets | yes | yes | yes | yes | no | no |
| per-gene cost | high | high | high | high | eQTL-only + $K$ columns | eQTL-only + $K$ columns |

## Evaluation plan

1. Synthetic: the E sets plus discordant colocalizations; I-a and I-b should reject discordant peaks, match E-free on concordant ones, and show the strong-link bin enriched over the weak one when links are informative and not otherwise.
2. Real data, same screened gene set: I-b vs E-free (fixed vs refit profiles, gate on vs off), I-a vs I-b (hard sign in the coefficient vs gate), and both against tied E at the calibrated $\lambda$. Compare the bin enrichments, selected peaks, the fraction of discordant peaks removed, PIP-threshold counts with gene-bootstrap CIs, and the resolution plot; the last is where fixed profiles should show their cost, if any.
3. Runtime, since the point of fixing the profiles is to make the full gene set cheap.

## Implementation notes

I-b is the smaller change from the existing E code: run the per-peak caQTL SuSiE fits once, freeze $E[b_k]$, $R\,E[b_k]$, $E[b_k' R\, b_k]$ and the caQTL PIPs, skip the caQTL and link updates in the gene pass, replace $\omega_p q^w_k$ by $\omega_{p, b(k)} c_k$, and add the per-bin EM. I-a additionally orients the peak predictors by the hurdle sign and uses a truncated-normal posterior in the peak branch of the SER. Runner, screen, plots and link-support table carry over.
