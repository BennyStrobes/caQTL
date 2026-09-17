# Model F: elements with a soft-tied peak effect (proposal)

Companion to `caqtl_mediated_fine_mapping_model_proposal_elements.md` (Model E, implemented). Model F is Model E with one change: a selected peak's coefficient in the eQTL is not fixed at the link value $g_k$ but drawn around it, $\beta \sim N(g_k, \tau_\beta^2)$, with $\tau_\beta^2$ shared across genes and estimated. Everything else (caQTL side, spike-slab link with hurdle measurement, one SuSiE over variants and real-link peaks, learned class weight $\rho_p$, pool gating, on/off rule, single start) is as in E. Not implemented.

## Why

Two findings about the tied version of E (peak coefficient fixed at $g_k$):

1. **Stacking.** A peak can be selected by more than one single effect, and with a tied effect that is not self-limiting: if the hurdle-implied effect $\hat g_k / \lambda$ is smaller than what expression shows, one copy explains part of the signal and a second single effect selects the same peak to supply the rest. Two copies at half size reproduce the right total. On a synthetic set with $\lambda$ set to twice the truth, the selection mass on every real peak was 1.9. $P_k$ is unaffected, but the bookkeeping assumes one copy, which is where the small ELBO dips on real data come from, and the fit silently absorbs a wrong $\lambda$.
2. **$\lambda$ cannot be learned inside the tied model.** Because stacking (or, when $\lambda$ is too small, losing the peak to a variant) absorbs any mis-specification, the posterior of $g_k$ never leaves $\hat g_k / \lambda$ and no in-model update of $\lambda$ has a signal. The tests confirmed this: a variational $\lambda$ factor stays wherever it starts.

The soft tie removes both. A second copy of a peak would carry its own free coefficient against a residual that no longer wants it, exactly as a variant's second copy does in SuSiE, so stacking has no reason to happen. And the fitted coefficients of selected peaks now sit where expression puts them, so their relationship to the hurdle estimates identifies $\lambda$.

## Model

Per gene, as in Model E:

$$A_k = X b_k + e_k, \qquad g_k = z_k h_k, \quad z_k \sim \text{Bern}(\pi), \quad h_k \sim N(0, \tau_g^2), \qquad \hat g_k \sim N(\lambda g_k,\ se_k^2 + \tau_u^2)$$

$$E = \sum_{l=1}^{L} \beta_l\, w_{\gamma_l} + e, \qquad \gamma_l \in \{1, \dots, p\} \cup \{p + k : z_k = 1\}, \qquad w_{\gamma_l} = \begin{cases} x_j & \gamma_l = j \le p \\ m_k = X b_k & \gamma_l = p + k \end{cases}$$

**Where $m_k$ enters.** Each single effect selects one element and multiplies its predictor by the effect's coefficient. For a variant the predictor is the genotype column $x_j$; for a peak it is the peak's genetic component $m_k = X b_k$, the accessibility of peak $k$ predicted from genotype through its fine-mapped caQTL. A selected peak therefore contributes $\beta_l\, X b_k$ to expression (in the tied Model E, $g_k\, X b_k$, the same mediated term as Model A). $m_k$ is random: the updates use $E[m_k] = X E[b_k]$ for the score and $E[b_k' X'X\, b_k]$ for the precision (in z-score mode $E[b_k]' z_E$ and $E[b_k' R\, b_k]$, so $m_k$ is never formed explicitly), and a diffusely fine-mapped peak is automatically a weaker, more uncertain predictor. It is also refit jointly: the expression residual feeds back into $b_k$ with weight equal to the peak's expected eQTL coefficient, which is how the eQTL sharpens a caQTL credible set, and the one place a peak's profile can be bent toward expression. At the variant level this is where the mediated PIP comes from: a variant's mediated PIP is $P_k$ times its caQTL PIP within $b_k$, so it inherits the variant composition of $m_k$.

For single effect $l$:

- variant element $j$: $\beta_l \sim N(0, V_l)$, $V_l$ estimated per effect (SuSiE);
- peak element $k$: $\beta_l \sim N(g_k,\ \tau_\beta^2)$ **(new)**: the coefficient of the peak's genetic component $m_k = X b_k$ is centered on the link value and may deviate from it.

Prior on the selected element, pool weights, and the on/off rule as in E. Shared hyperparameters: $\pi, \tau_g^2, \tau_u^2$ (as in A/E), $\rho_p$ (as in E), $\tau_\beta^2$ (new), and $\lambda$ (now estimable, see below).

**Limits.** $\tau_\beta^2 \to 0$ recovers the tied Model E; $\tau_\beta^2 \to \infty$ recovers E-free (a free peak coefficient with the link only gating the pool). $\tau_\beta^2$ estimated from data says where between those the hurdle links actually sit.

**Interpretation of the pieces.** $g_k$ is the peak-gene link as the hurdle measures it; $\beta_l$ is what this gene's expression says the peak's effect is; $\tau_\beta^2$ is how much the two disagree across mediating peaks after the scale $\lambda$ is accounted for. A per-peak deviation $E[\beta] - E[g_k]$ is a diagnostic of $\lambda$ misfit at that peak.

## Inference

Mean-field variational, coordinate ascent, z-score mode, as in E. What changes:

1. **Peak element in the SER.** Under $q(g_k)$ the prior on $\beta$ for peak $k$ is a Gaussian kernel with mean $E[g_k]$ and variance $\tau_\beta^2$ (the expected log prior is $-(\beta^2 - 2\beta E[g_k] + E[g_k^2]) / 2\tau_\beta^2$ up to constants). With the eQTL score $S_{E,k}$ and precision $P_{E,k}$ of the peak's genetic component on the residual, the coefficient's conditional posterior is Gaussian with precision $1/\tau_\beta^2 + P_{E,k}$ and mean $(E[g_k] / \tau_\beta^2 + S_{E,k})$ divided by that precision; the element's Bayes factor is the corresponding closed-form Gaussian integral (the SER Bayes factor with a non-zero prior mean). Variant elements are unchanged. The single effect's prior variance $V_l$ is optimized over the variant elements only; the peak elements use $\tau_\beta^2$.
2. **Links.** The factor of $g_k$ pools the hurdle term with the selection-weighted evidence from the fitted coefficients: score $\lambda \hat g_k / w_k + \sum_{l \text{ on}} \alpha_{l,p+k} E[\beta_{lk}] / \tau_\beta^2$, precision $\lambda^2 / w_k + \sum_{l \text{ on}} \alpha_{l,p+k} / \tau_\beta^2$, prior $\pi, \tau_g^2$; closed form. The expression data now reach $g_k$ through the coefficients, with weight $1/\tau_\beta^2$: a tight tie lets them move $g_k$, a loose one does not.
3. **caQTL single effects.** As in E, with the peak's expected eQTL coefficient $\sum_l \alpha_{l,p+k} E[\beta_{lk}]$ and second moment $\sum_l \alpha_{l,p+k} E[\beta_{lk}^2]$ in place of the shared $E[g_k]$, $E[g_k^2]$. Because each effect has its own $\beta$, the cross-effect terms are products of expectations exactly as in SuSiE: the bookkeeping is exact and no $E[N_k^2]$ correction is needed.
4. **Hyperparameters, across genes.** $\pi, \tau_g^2, \tau_u^2, \rho_p$ as in E. New: $\tau_\beta^2 = \sum \alpha_{l,p+k} E[(\beta_{lk} - g_k)^2] / \sum \alpha_{l,p+k}$ over on effects, with $E[(\beta - g)^2] = E[\beta^2] - 2E[\beta]E[g] + E[g^2]$ under the factorization. $\lambda$: the selection-weighted regression of $\hat g_k$ on $E[g_k]$, $\lambda = \sum_k s_k \hat g_k E[g_k] / w_k \big/ \sum_k s_k E[g_k^2] / w_k$ with $s_k$ the selection probability; this is the update that had no signal in the tied model and has one here, because $E[g_k]$ for a selected peak now reflects the fitted coefficient. Keep $\tau_u^2$ fixed while $\lambda$ is estimated (they are not jointly identified).

**Stacking is self-limiting again.** A second effect selecting peak $k$ faces a residual from which the first copy's $E[\beta] m_k$ has been removed; its own coefficient's posterior is pulled toward 0 by the data and toward $g_k$ by the prior, and the Bayes factor pays $E[g_k]^2 / 2\tau_\beta^2$ for the disagreement. With $\tau_\beta^2$ small relative to $g_k^2$ that is a large penalty; with $\tau_\beta^2$ large the second copy is a free effect against an empty residual, which SuSiE already handles.

**Identifiability of $\lambda$ and $\tau_\beta^2$.** $\tau_\beta^2$ is the empirical spread of fitted coefficients around $E[g_k]$ over selected peaks, an ordinary variance estimate. $\lambda$ is the ratio of hurdle estimates to fitted link values over the same peaks. They are separately identified as long as some peaks are selected and the tie is not so loose that $g_k$ decouples from $\beta$ (the E-free limit, where $\lambda$ is unidentified by construction). The EM for $\tau_\beta^2$ has no reason to run to infinity: a larger $\tau_\beta^2$ makes deviations cheap but lowers the prior density of every fitted coefficient, the usual variance trade-off.

## Initialization

As in E, plus: each peak element's coefficient starts at $E[g_k]$ from the hurdle-only link posterior; $\tau_\beta^2$ starts at a fraction of $\tau_g^2$, e.g. $0.1\,\tau_g^2$ (a fairly tight tie, so the first passes behave like the tied model), and is then estimated; $\lambda$ starts at the moment estimate or the external calibration and, if estimated, updates only after $\rho_p$ and the pool have settled (a few outer iterations), since its update needs selected peaks.

## Outputs

As in E ($P_k$, direct and mediated PIPs, combined PIP, $\rho_p$), plus per selected peak the fitted coefficient $E[\beta]$ and its deviation from $E[g_k]$, and the shared $\tau_\beta^2$ and, if estimated, $\lambda$ with its update trajectory. The per-peak selection mass stays in the output as the check that stacking has gone.

## Relation to the other models

| | E (tied) | E-free | F (soft tie) |
|---|---|---|---|
| peak coefficient | $= g_k$ | free $N(0, V_l)$ | $N(g_k, \tau_\beta^2)$ |
| hurdle size and sign | hard constraint | unused | prior center |
| stacking | possible when $\lambda$ too large | self-limiting | self-limiting |
| bookkeeping with shared $g_k$ | approximate ($E[N_k]$ for $E[N_k^2]$) | exact | exact |
| $\lambda$ learnable in-model | no | not applicable | yes, from selected peaks |
| new shared parameter | | | $\tau_\beta^2$ |

## Evaluation plan

1. The stacking test: Model A's generative process with $\lambda$ set to twice and half the truth; the selection mass on real peaks should be $\approx 1$, $P_k$ correct, and the in-model $\lambda$ should move toward the truth from both sides (the tied model fails all three).
2. The synthetic sets used for E (real vs spurious links, direct-only, untestable links): F should match E where E was right.
3. Real data: F vs E at the same fixed $\lambda$, comparing $P_k$, the number of selected peaks, and the ELBO's monotonicity (F's should be clean apart from refresh steps); then F with $\lambda$ estimated, compared against the external calibration from the link-support table.
4. $\tau_\beta^2$ on real data: near zero says the hurdle sizes are well calibrated once $\lambda$ is right; large says the hurdle carries direction but not size, and E-free is the honest model.

## Implementation notes

Starting from `ca_qtl_elements_susie.py`: the peak entries of the SER get a Gaussian coefficient with prior mean $E[g_k]$ and variance $\tau_\beta^2$ (a non-zero-mean SER Bayes factor and posterior, a few lines), stored per effect like the E-free coefficients already are; the link update takes its eQTL evidence from the fitted coefficients rather than from the residual scores; the caQTL update and the ELBO use per-effect coefficient moments, which removes the shared-$g_k$ approximations; two M-steps are added ($\tau_\beta^2$, and $\lambda$ made live). The runner, screen, plots and link-support table carry over unchanged.
