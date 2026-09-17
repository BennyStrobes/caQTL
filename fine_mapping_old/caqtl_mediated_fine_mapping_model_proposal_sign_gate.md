# Model H: free peak effects with a sign gate (proposal)

Companion to `caqtl_mediated_fine_mapping_model_proposal_elements.md` (Model E, implemented) and `caqtl_mediated_fine_mapping_model_proposal_sign.md` (Model G, proposal). Model H is the simplest model that uses what the hurdle links can be trusted for and nothing else. Not implemented.

## Premise

The hurdle estimate $\hat g_k$ is a per-cell regression coefficient of expression counts on peak accessibility, on scales that differ by gene and by peak. The eQTL and caQTL effects are z-scores whose scales depend on each gene's expression variance, each peak's accessibility variance, sample size and cell composition. The conversion between the two is gene- and peak-specific, and mediation through any one peak is usually partial. So the assumption behind Models A, D and tied E, that (hurdle link) $\times$ (caQTL effect) lines up with the eQTL effect up to one global scale $\lambda$, is unrealistic across genes, and every calibration problem seen so far ($\lambda$ unidentified or drifting, $\tau_u^2$ absorbing mis-specification, peaks stacked or lost) follows from it.

What does not depend on any scale is the **direction**: if a peak's accessibility is positively associated with the gene's expression, a variant that opens the peak should raise expression. Model H uses the hurdle for two things only, whether a link is real and which direction it points, and lets the expression data determine every effect size.

## Model

Per gene, as in Model E-free:

$$A_k = X b_k + e_k \qquad \text{(caQTL, SuSiE single effects, jointly refit)}$$

$$E = \sum_{l=1}^{L} \beta_l\, w_{\gamma_l} + e, \qquad w_{\gamma_l} = \begin{cases} x_j & \text{variant } j \\ m_k = X b_k & \text{peak } k \end{cases}, \qquad \beta_l \sim N(0, V_l) \text{ for every element}$$

One SuSiE over the $p$ variants and the linked peaks; a selected peak contributes $\beta_l\, X b_k$ with a free coefficient; $V_l$ estimated per single effect; class weights $\omega_v$, $\omega_p$ with $\rho_p = \omega_p / \omega_v$ learned; the on/off rule and the pool refresh as in E. No $\lambda$, no $\tau_u^2$, no tie between $\beta_l$ and $\hat g_k$.

**The hurdle enters through the pool weight of each peak, in two factors:**

$$P(\gamma_l = p + k) \propto \omega_p \cdot q_k \cdot c_k$$

- $q_k = P(z_k = 1 \mid \hat g_k)$: the spike-slab probability that the link is non-zero, from the hurdle alone, as in E-free. A null-looking link barely enters the pool.
- $c_k \in [0, 1]$ **(new): the sign gate.** Let $j^\ast$ be the peak's caQTL lead variant and $d_k = \text{sign}(z_{E, j^\ast})\, \text{sign}(z_{A_k, j^\ast})$ the direction the marginal statistics already show for "opening the peak changes expression". Concordance means $d_k = \text{sign}(\hat g_k)$. Then

$$c_k = \begin{cases} 1 & \text{concordant, or direction undetermined} \\ 0 & \text{discordant and both directions clear} \end{cases}$$

with "clear" meaning $|z_{E, j^\ast}| > z_{\min}$ and $|t_k| = |\hat g_k / se_k| > z_{\min}$ (default $z_{\min} = 2$; the caQTL lead's $|z|$ is large by construction). A soft version replaces the indicator by the probability of concordance, $c_k = \Phi\big(d_k\, t_k\big)$ when the eQTL direction at the lead is clear, which lets a weak hurdle sign impose almost nothing; the hard gate is the default for transparency.

A discordant peak therefore never competes for a single effect, and at a collinear locus the variant wins by default. A concordant peak competes exactly as in E-free.

**Direction, stated precisely.** $m_k$ is oriented so that a positive caQTL effect means the alternate allele opens the peak; a positive fitted $\beta_l$ means opening the peak raises expression; a positive $\hat g_k$ asserts the same. The gate compares the hurdle's assertion with the sign of the eQTL effect at the variant that most strongly opens the peak.

## What this buys and what it costs

- **Buys.** Half of chance colocalizations between an eQTL and a caQTL have the wrong direction; the gate removes them before they can be selected. Everything about effect sizes is learned from expression, so nothing can be mis-calibrated, nothing can stack, and the bookkeeping is exact (each effect has its own coefficient, as in SuSiE).
- **Costs.** The gate uses one variant per peak and a threshold, so it is coarse; a peak whose lead has a weak eQTL is not gated at all (which is right: no direction to compare). It uses the eQTL z at the lead both to gate and to fit, a mild double use of the kind functional priors also make. And the hurdle's magnitude is discarded entirely, by design; if the sizes are in fact informative once heterogeneity is allowed for, Model F is the version that would use them.
- **Relation to G.** G does the same thing inside the coefficient's prior, per single effect and with the hurdle's sign uncertainty carried through $\Phi(t_k)$, at the cost of truncated-normal mixtures in the SER. H is the one-line approximation: gate at the pool, on the marginal signs. If H proves useful, G is the version to grow into.

## Inference

Identical to Model E with `free_peak_effects`, plus the computation of $c_k$ once per gene at initialization (from the input z-scores) and its multiplication into the pool weights wherever $q^w_k$ is used, including the normalizer of the $\rho_p$ M-step ($Q_g = \sum_k q^w_k c_k$). No new updates, no new hyperparameters.

## Initialization, outputs

As in E-free. Outputs gain, per peak, the gate value $c_k$ and the direction $d_k$ used, so a gated-out peak is visible in the components file; the selection mass column stays as the check that nothing stacks.

## Relation to the other models

| | E (tied) | E-free | F (soft tie) | G (sign in prior) | H (sign gate) |
|---|---|---|---|---|---|
| hurdle size | hard constraint | unused | prior center | unused | unused |
| hurdle sign | via the tied size | unused | via the prior center | in the coefficient prior, $\Phi(t_k)$-weighted | pool gate against the marginal direction at the caQTL lead |
| needs $\lambda$ | fixed | no | learnable | no | no |
| new machinery | | | $\tau_\beta^2$, non-zero-mean SER | truncated-normal mixtures | one gate per peak |
| stacking | possible | self-limiting | self-limiting | self-limiting | self-limiting |

## Evaluation plan

1. Synthetic: the E sets plus a set with discordant colocalizations (eQTL and caQTL at the same variant, hurdle sign opposite). H should reject the discordant peaks E-free accepts and match E-free elsewhere.
2. Real data, same screened gene set: H vs E-free isolates the gate; H vs tied E at the calibrated $\lambda$ isolates the value of the size. Compare $\rho_p$, selected peaks, the fraction of links gated out, PIP-threshold counts with gene-bootstrap CIs, and the link-support table.
3. Report how many links were gated out and, among them, how many would have been selected by E-free; that number is the gate's direct effect.

## Implementation notes

In `ca_qtl_elements_susie.py`, with `free_peak_effects=True`: compute $d_k$ and $c_k$ in `initialize_gene` from `data['XtE']` and `data['XtA']` at the caQTL lead (excluding imputed cells) and the link inputs; store `state['gate']`; use `state['qw'] * state['gate']` in `prior_weights` and in the $\rho_p$ normalizer. Runner flags: `--sign_gate`, `--sign_gate_min_z` (default 2), `--sign_gate_soft`. Components output gains `peak_sign_gate` and `peak_marginal_direction`. Everything else carries over unchanged.
