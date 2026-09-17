# Model G: elements with sign-concordant peak effects (proposal)

Companion to `caqtl_mediated_fine_mapping_model_proposal_elements.md` (Model E, implemented) and `caqtl_mediated_fine_mapping_model_proposal_soft_tie.md` (Model F, proposal). Model G is Model E-free with one change: a selected peak's coefficient carries a prior whose sign follows the hurdle link. The hurdle contributes its **direction** and its **gating** of the pool, and nothing else: no effect size, no $\lambda$, no $\tau_u^2$. Not implemented.

## Why

Across Models A, D, E and F the hurdle's effect size has been the source of every calibration problem ($\lambda$ unidentified or drifting, $\tau_u^2$ absorbing mis-specification, peaks stacked under a wrong $\lambda$). Its sign has never caused trouble and has a clear use. At a colocalized locus, a peak whose hurdle says "more accessible, more expressed" can explain the eQTL only if the variant that opens the peak also raises expression. A variant element can take either sign; a sign-constrained peak element cannot. A peak whose direction disagrees with the expression signal therefore gets a Bayes factor near zero and loses to the variant outright. Half of chance colocalizations have the wrong sign, so this halves that false-positive route, and it needs only a dimensionless quantity, the hurdle z-score, which requires no calibration.

## Model

Per gene, as in Model E:

$$A_k = X b_k + e_k, \qquad g_k = z_k h_k, \quad z_k \sim \text{Bern}(\pi), \quad h_k \sim N(0, \tau_g^2), \qquad \hat g_k \sim N(\lambda_0 g_k,\ se_k^2)$$

The link block is used only to gate the pool: $q_k = P(z_k = 1)$ from the hurdle alone, with $\lambda_0$ any fixed scale (it cancels in $q_k$'s dependence on the z-score up to the slab width, and never enters the eQTL side). It can equally be replaced by any monotone map from the hurdle z-score to a weight.

$$E = \sum_{l=1}^{L} \beta_l\, w_{\gamma_l} + e, \qquad \gamma_l \in \{1, \dots, p\} \cup \{p + k : z_k = 1\}, \qquad w_{\gamma_l} = \begin{cases} x_j & \gamma_l = j \le p \\ m_k = X b_k & \gamma_l = p + k \end{cases}$$

For single effect $l$:

- variant element $j$: $\beta_l \sim N(0, V_l)$, $V_l$ estimated per effect (SuSiE);
- peak element $k$: **sign-weighted two-sided slab**

$$\beta_l \sim \Phi(t_k)\, N^{+}(0, V_l) + \big(1 - \Phi(t_k)\big)\, N^{-}(0, V_l), \qquad t_k = \hat g_k / se_k,$$

with $N^{\pm}$ the half-normals on the positive and negative half-lines and $\Phi$ the standard normal CDF. For a strong link $\Phi(t_k)$ is essentially 0 or 1 and the peak is one-sided; for a weak link the prior is nearly symmetric, as it should be when the sign is uncertain. The magnitude is free and shares the effect's $V_l$ with the variants, so there is nothing to calibrate and no reason to stack.

Prior on the selected element ($\omega_v$ per variant, $\omega_p q^w_k$ per peak, $\rho_p = \omega_p / \omega_v$ learned), pool gating and its refresh, and the on/off rule as in E.

**Where $m_k$ enters.** As in E: a selected peak contributes $\beta_l\, X b_k$, with $E[b_k]$ and $E[b_k' R\, b_k]$ used as the predictor's score and precision, $b_k$ refit jointly (the expression residual reaches $b_k$ with weight equal to the peak's expected coefficient), and the mediated PIP of a variant equal to $P_k$ times its caQTL PIP within $b_k$.

**Direction, stated precisely.** $m_k$ is oriented so that a positive caQTL effect means the alternate allele opens the peak. A positive $\beta_l$ then means opening the peak raises expression, which is what a positive hurdle link asserts. Sign concordance is the requirement $\text{sign}(\beta_l) = \text{sign}(\hat g_k)$, softened by the hurdle's own uncertainty through $\Phi(t_k)$.

**Limits.** $\Phi \equiv \tfrac12$ recovers E-free. Replacing the half-normals by $N(g_k, \tau_\beta^2)$ recovers F. Tied E is F with $\tau_\beta^2 \to 0$.

## Inference

Mean-field variational, coordinate ascent, z-score mode, as in E. What changes is confined to the peak elements of the SER:

1. **Peak element Bayes factor and posterior.** With score $S = S_{E,k}$ and precision $P = P_{E,k}$ of the peak's genetic component on the residual, and prior variance $V$, the untruncated conditional posterior is $N(\mu, \nu)$ with $\nu = 1/(1/V + P)$, $\mu = \nu S$. Under a half-normal prior the posterior is the same Gaussian truncated to the half-line, and the marginal likelihood picks up a normal-CDF factor. Writing $a = \mu / \sqrt{\nu}$,
   - $\text{BF}^{+} = \text{BF}_{\text{sym}} \cdot 2\,\Phi(a)$, $\text{BF}^{-} = \text{BF}_{\text{sym}} \cdot 2\,\Phi(-a)$, where $\text{BF}_{\text{sym}}$ is the ordinary SER Bayes factor;
   - the element's Bayes factor is $\Phi(t_k)\,\text{BF}^{+} + (1 - \Phi(t_k))\,\text{BF}^{-}$;
   - the posterior of $\beta$ is a mixture of the two truncated normals with weights proportional to $\Phi(t_k)\text{BF}^{+}$ and $(1 - \Phi(t_k))\text{BF}^{-}$; the truncated-normal mean and variance are closed form (Mills ratio), and the mixture moments $E[\beta]$, $E[\beta^2]$ follow.
   A peak whose sign disagrees with the residual has $\Phi(a)$ near zero on the favored side and a Bayes factor near $\text{BF}_{\text{sym}} \cdot 2\Phi(-|a|) \cdot (1 - \Phi(t_k))$, which for a strong link is tiny: it loses to the variant.
2. **Prior variance $V_l$.** Optimized on the whole pool's marginal likelihood as in E, with the peak elements' Bayes factors computed as above.
3. **On/off rule** as in E.
4. **Links.** Hurdle only: $q_k$, slab moments from the hurdle alone, as in E-free (the eQTL does not inform $g_k$; $g_k$ exists only to gate the pool).
5. **caQTL single effects.** As in E-free: the peak's expected eQTL coefficient and second moment are $\sum_l \alpha_{l,p+k} E[\beta_{lk}]$ and $\sum_l \alpha_{l,p+k} E[\beta_{lk}^2]$ over on effects, per-effect moments, so the bookkeeping is exact.
6. **Hyperparameters.** $\pi, \tau_g^2$ from the hurdle marginal (as now); $\rho_p$ as in E. No $\lambda$, no $\tau_u^2$.

Stacking is self-limiting as for variants in SuSiE: a second copy of a peak is a free coefficient against a residual that no longer wants it.

## Initialization

As in E: caQTL-only fits per peak; link posteriors from the hurdle alone, which set $q^w_k$ and $\Phi(t_k)$; all eQTL single effects empty; $\rho_p = 1$; pass order eQTL effects, links, caQTL effects. Nothing needs a starting $\lambda$.

## Outputs

As in E ($P_k$, direct and mediated PIPs, combined PIP, $\rho_p$, per-peak selection probability and mass), plus per selected peak the fitted coefficient $E[\beta]$ and the posterior probability that its sign agrees with the hurdle, which is the mixture weight of the concordant truncated component. A peak selected with a discordant sign should be rare and is worth inspecting.

## Identifiability and limitations

- **No calibration parameters.** $\rho_p$ is a symmetric split of posterior mass; $\pi, \tau_g^2$ describe the hurdle marginal; $V_l$ is per effect. Nothing about the hurdle's size enters, so nothing about it can drift.
- **Collinearity remains.** A peak with a single-signal caQTL is still indistinguishable from its lead variant in the concordant case; the split follows $\rho_p$ and the credible set contains both. What the sign adds is that the *discordant* half of such ties is now resolved in favor of the variant, which the symmetric models could not do.
- **The sign is only as good as the hurdle's.** $\Phi(t_k)$ carries the hurdle's own uncertainty, so a weak link imposes almost nothing. A hurdle link whose sign is wrong for a systematic reason (a repressive peak scored as activating) would push its peak out of the model at loci where it does mediate; the sign-agreement output makes such cases visible.
- **Magnitude information is discarded.** If the hurdle sizes are in fact calibrated once $\lambda$ is right, F uses information G ignores. The comparison of G against F and tied E at a fixed $\lambda$ is how to find out; G is the version that cannot be wrong about size because it says nothing about it.

## Relation to the other models

| | E (tied) | E-free | F (soft tie) | G (sign) |
|---|---|---|---|---|
| hurdle size | hard constraint | unused | prior center | unused |
| hurdle sign | via the tied size | unused | via the prior center | prior on the coefficient's sign, weighted by $\Phi(t_k)$ |
| needs $\lambda$ | fixed | no | learnable | no |
| discordant colocalization | loses on size mismatch, if $\lambda$ right | can be selected | penalized by $\tau_\beta^2$ | loses outright for strong links |
| stacking | possible | self-limiting | self-limiting | self-limiting |
| bookkeeping | approximate | exact | exact | exact |

## Evaluation plan

1. Synthetic: the E sets plus a set with discordant colocalizations (a caQTL and an eQTL at the same variant with opposite-sign hurdle link). G should reject the discordant peaks that E-free accepts, match E-free elsewhere, and show no stacking at any $\lambda$ (there is none to set).
2. Real data, same screened gene set: G vs E-free (isolates the value of the sign), G vs tied E and F at the calibrated $\lambda$ (isolates the value of the size). Compare $\rho_p$, the number of selected peaks, the sign-agreement distribution, PIP-threshold counts with gene-bootstrap CIs, and the link-support table.
3. Expect G's $\rho_p$ above tied E's, since concordant peaks are no longer penalized for a mis-sized implied effect, and below E-free's if E-free was admitting discordant peaks.

## Implementation notes

Starting from `ca_qtl_elements_susie.py` with `free_peak_effects=True`: the peak-element branch of the SER computes $\text{BF}^{\pm}$ from the ordinary SER quantities and $\Phi(\pm a)$, mixes them with $\Phi(t_k)$, and stores truncated-normal mixture moments in place of the Gaussian ones (`pk_mu`, `pk_mu2`); $t_k$ is available from the link inputs at initialization. The link factor, caQTL updates, $\rho_p$ M-step, runner, plots and link-support table carry over unchanged; the components output gains the sign-agreement probability.
