# Model J: cTWAS on peaks with standardized peak predictors and group priors by hurdle significance and sign concordance

Implemented in `ca_qtl_ctwas_susie.py` (class `CAQTL_CTWAS_SUSIE`) with the runner `run_ca_qtl_ctwas_eqtl_finemapping.py`; see *Implementation* at the end. Companion to the Model E family and to `..._ctwas_style.md` (Model I), of which J is the plainest member.

## Premise

The hurdle's effect size cannot be lined up with eQTL effects across genes; its sign and its significance can. The joint refit of caQTL profiles buys eQTL-side sharpening at the price of a feedback loop. Model J keeps nothing that needs calibration or watching: peak profiles are fixed from their own data, every element has a free effect, and the hurdle enters only through learned prior weights on groups of peaks defined by how significant the link is and whether its direction agrees with the marginal data. Nothing is constrained; whether significance and sign matter is measured, as the enrichment of one group over another.

## Model

**Fixed, standardized peak predictors.** For each linked peak $k$, fine-map its caQTL alone with SuSiE-RSS and freeze the posterior mean $E[b_k]$, the second moment $s_k^2 = E[b_k' R\, b_k]$, and the caQTL PIPs. The peak's genetic component, standardized to unit variance in the LD-scaled units of a variant column, is

$$\tilde m_k = X E[b_k] / s_k .$$

Its sufficient statistics are a variant's: score $\tilde m_k' z_E$-equivalent $= E[b_k]' z_E / s_k$ (a TWAS-style z-score of imputed accessibility against expression) and self-precision 1; its correlation with variant $j$ is $(R\, E[b_k])_j / s_k$ and with peak $k'$ is $E[b_k]' R\, E[b_{k'}] / (s_k s_{k'})$. The augmented LD matrix over $p + K$ elements is built from these and the eQTL is fit on it.

**eQTL regression.**

$$E = \sum_{l=1}^{L} \beta_l\, w_{\gamma_l} + e, \qquad w_{\gamma_l} \in \{x_1, \dots, x_p\} \cup \{\tilde m_1, \dots, \tilde m_K\}, \qquad \beta_l \sim N(0, V_l),$$

one SuSiE over the augmented pool, $V_l$ estimated per single effect and shared across element types, which is meaningful because every predictor has unit variance. No sign constraint, no tie to $\hat g_k$, no $\lambda$.

**Group priors.** Each element belongs to a group: variants form one group; peaks are grouped by the hurdle z-score $t_k = \hat g_k / se_k$ in bins (default edges 2 and 4) and, for bins where the sign is meaningful, by concordance:

- $d_k = \text{sign}(z_{E, j^\ast})\, \text{sign}(z_{A_k, j^\ast})$ at the peak's caQTL lead $j^\ast$ is the direction the marginal data show for "opening the peak changes expression";
- concordant if $d_k = \text{sign}(\hat g_k)$, discordant if opposite, undetermined if the eQTL at the lead is not clear ($|z_{E, j^\ast}|$ below a threshold, default 2).

Default groups: `variant`, `peak_t<2`, and for each of `t2-4` and `t>4` the three states `concordant`, `discordant`, `undetermined` (seven peak groups). Each group $g$ has a prior weight $\pi_g$, shared across genes and learned by EM as the expected number of causal elements in the group per element in the group:

$$\pi_g = \frac{\sum_{\text{genes}} \sum_{l \text{ active}} \sum_{e \in g} \alpha_{le}}{\sum_{\text{genes}} |g|}, $$

shrunk toward the variant rate by pooling each peak group with a small number of pseudo-elements at that rate (default 10; a group with one or two elements would otherwise swing between rounds) and floored so an empty group cannot vanish. Within a gene the prior over elements is $\propto \pi_{g(e)}$. The ratios $\pi_g / \pi_{\text{variant}}$ are the enrichments; the two that matter are strong-concordant over weak (does significance carry information?) and concordant over discordant at the same strength (does the sign?). This is cTWAS's group prior with hurdle strength and sign as the grouping.

**How the priors act, with numbers.** Within a gene the prior probability that a single effect selects element $e$ is $\pi_{g(e)}$ divided by the summed weight of the gene's pool. Take 1,000 variants and three peaks: a strong concordant one with enrichment 30, a weak one with enrichment 1, a strong discordant one with enrichment 0.6. The pool's weight is $1000 + 30 + 1 + 0.6$, so each variant holds about 0.097% of the prior, the strong concordant peak 2.9%, the weak peak 0.1%, the discordant one 0.06%. To reach a posterior of 0.5 on its own, a variant needs a Bayes factor near 1,000 and the strong concordant peak near 33. At a collinear locus, where a peak and its lead variant have the same likelihood, the split is the prior ratio alone: 97:3 in favor of the strong concordant peak, 37:63 against the discordant one. The effect-size prior is the same for every element, $\beta_l \sim N(0, V_l)$ with $V_l$ estimated per single effect and switched off when the null does at least as well; because peak predictors are standardized, sharing $V_l$ between a variant and a peak is meaningful (both are effects per unit of a unit-variance predictor), and nothing about the hurdle's magnitude touches it. So the hurdle enters in one place only, the group membership of each peak, and what the model learns about the hurdle is the handful of group enrichments.

**Measuring the sign vs imposing it.** With learned group weights, J measures whether discordant peaks are selected as often as concordant ones; it does not assume they are not. That is the honest default, and its limit is the usual one: a discordant peak that sits on a direct causal variant is collinear with it and will be selected as a colocalized element, because from summary statistics nothing distinguishes colocalization from mediation. If the sign is to be treated as a constraint rather than a hypothesis, `--sign_gate` sets the prior weight of every discordant peak (both directions clear) to zero, which is Model H's gate inside J; the discordant group's learned weight is then reported for information only. On the discordant synthetic set (direct eQTL at a peak's caQTL lead, hurdle sign opposite), J without the gate selects the discordant peaks exactly as E-free does and learns a large weight for them, since in that set every discordant peak is perfectly colocalized; with the gate they are excluded and the direct variant keeps its PIP.

**Direction, stated precisely.** $\tilde m_k$ is oriented so that a positive caQTL effect means the alternate allele opens the peak; a positive coefficient means opening the peak raises expression; a positive $\hat g_k$ asserts the same. Concordance compares the hurdle's assertion with the marginal eQTL sign at the variant that most strongly opens the peak.

## Inference

Alternate two steps until the group weights settle:

1. **Per gene:** load the LD, build the augmented system, and run SuSiE-RSS to convergence with prior weights $\pi_{g(e)}$. This is the existing IBSS with prior-variance estimation and the $V = 0$ switch-off; nothing new.
2. **Across genes:** update every $\pi_g$ from the posterior masses of the active effects.

A handful of rounds suffices; each round refits every gene from scratch, which is cheap because the caQTL side is not touched again. Z-score mode only.

## Outputs

- Per variant: direct PIP (mass on variant elements), mediated PIP $= 1 - \prod_k (1 - P_k \cdot \text{caQTL PIP}_{kj})$ with the frozen caQTL PIPs, combined PIP.
- Per peak: $P_k = P(\text{some active effect selects } k)$, its group, its fitted coefficient on the original $m_k$ scale ($E[\beta] / s_k$), the marginal direction $d_k$.
- Shared: the group weights, their enrichments over variants, the number of elements per group.
- The components file keeps the columns the plots and the link-support table read (`link_probs` is filled with the two-sided significance of the hurdle link, `link_inclusion_probs` with $P_k$), and adds `peak_groups`.

## What it gives up, what it gains

Gives up: eQTL-side sharpening of caQTL credible sets (the mediated PIP composes with the caQTL-only fine-mapping), the hurdle magnitude, and the residual-variance option. Gains: cost of about eQTL-only fine-mapping plus $K$ columns per gene, no calibration parameters, no feedback loop, no stacking, exact SuSiE bookkeeping, and two scale-free readouts of what the hurdle is worth. Collinearity between a peak and its lead variant is still a mixed credible set split by the prior weights, and a discordant peak at such a locus is now split by a weight the data have set for discordant peaks rather than by fiat.

## Relation to the other models

| | E (tied) | E-free | H (gate) | I-a | J |
|---|---|---|---|---|---|
| caQTL profile | jointly refit | jointly refit | jointly refit | fixed | fixed, standardized |
| hurdle size | hard | unused | unused | unused | unused |
| hurdle sign | via size | unused | pool gate | half-normal coefficient | learned group weight |
| hurdle significance | spike gates pool | spike gates pool | spike gates pool | learned bin weight | learned bin weight |
| constraints imposed | tie | none | gate | sign | none |
| needs $\lambda$ | fixed | no | no | no | no |
| per-gene cost | high | high | high | low | low |

## Evaluation plan

1. Synthetic: the E sets, plus a discordant set (eQTL direct at a variant that is also a peak's caQTL lead, hurdle sign opposite). J should put the discordant group's weight at or below the variants', leave the direct PIP on the variant there, and match E-free on concordant mediation.
2. Real data, same screened gene set: the seven group weights and their enrichments are the first thing to read; then $P_k$, threshold counts with gene-bootstrap CIs, the resolution plot (where fixed profiles should show their cost, if any), and the link-support table.

## Implementation

**Files.** `ca_qtl_ctwas_susie.py` and `run_ca_qtl_ctwas_eqtl_finemapping.py`. The runner takes the same input summary as the other runners and writes the same output formats. Flags: `--t_bin_edges` (default `2,4`), `--t_bin_quantiles N` (N equal-count bins of $|t|$ instead of fixed edges; the edges used are printed), `--no_sign_groups` (significance bins only), `--sign_min_z` (default 2, the clarity threshold for the eQTL direction at the lead), `--sign_gate` (impose the sign), `--group_pseudo_elements` (default 10), `--max_rounds`, `--group_tol`, `--L_eqtl`, `--L_caqtl`, `--no_component_output`. The components file adds `peak_groups`, `peak_marginal_direction` and `peak_sign_gate`. Before any fitting the runner prints the bin edges, the number of variants, and the number of peaks in each group, so an empty or thin group is visible up front. A first pass on real data is best run with `--no_sign_groups --t_bin_quantiles 5`: five equal-count significance bins, whose enrichments say whether and how strongly the hurdle carries information before any sign question is asked. The hyperparameter file has one row per group with its weight, element count and expected causal count, plus a `link_scale` column fixed at 1 so the link-support table's calibration reports the empirical hurdle-to-eQTL scale among selected peaks as a by-product. `run_fine_mapping.sh` has a disabled step 2c with the command.
