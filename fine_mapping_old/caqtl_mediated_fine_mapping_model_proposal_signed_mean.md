# Model K: cTWAS on peaks with a per-bin signed prior mean on peak effects (proposal)

Companion to `caqtl_mediated_fine_mapping_model_proposal_ctwas_groups.md` (Model J, implemented). Model K is Model J with one change: a peak element's effect prior has a non-zero mean, oriented by the hurdle's sign and sized by a per-bin parameter learned across genes. Not implemented.

## Why

Model J uses the hurdle through group weights only: how often peaks of each significance bin, and each sign-concordance state, are selected. On the first real run the concordant bins came out enriched about 20-fold over variants and the discordant bins 4 to 12-fold, so the sign carries information but J can only express that as "discordant peaks win less often". K turns the sign into what it should be, a prior on the direction of the peak's effect, and adds a learned typical effect size per bin, without any of the machinery that failed before: no per-link tie to the hurdle magnitude ($\lambda$, stacking), no truncation (I-a, G), no gate threshold (H). Everything stays Gaussian and closed form.

## Model

Everything as in Model J: fixed, standardized peak predictors $\tilde m_k = X E[b_k] / s_k$; one SuSiE over variants and peaks; group weights $\pi_g$ by hurdle significance bin (sign states optional, see below); single-effect prior variance $V_l$ estimated per effect. The change is the effect prior of a peak element in bin $b$:

$$\beta_l \mid \gamma_l = \text{peak } k \in b \;\sim\; N\big(\mu_b \cdot \text{sign}(\hat g_k),\ V_l\big) \qquad \text{(variants: } \beta_l \sim N(0, V_l)\text{)}$$

- $\mu_b \ge 0$: the typical standardized effect of a mediating peak in bin $b$, shared across genes, learned by EM. It is the size of the effect a peak has on expression when it is selected, on the unit-variance predictor scale, pooled over the bin. It involves nothing from the hurdle except which bin the peak is in.
- $\text{sign}(\hat g_k)$: the hurdle's direction, applied to the mean. Concordance means the fitted $\beta_l$ has the sign of $\hat g_k$.

**Optional per-bin variance.** As in cTWAS, each peak bin can carry its own prior variance $V_b$ in place of the effect's $V_l$, so a bin is described by a mean and a spread: $\beta_l \sim N(\mu_b\, \text{sign}(\hat g_k),\ V_b)$. This is the version to prefer once bins have enough selected peaks to estimate a spread (a few dozen); with fewer, share $V_l$.

**What the signed mean does.** A coefficient of the wrong sign is not forbidden. Relative to the right sign it pays $\big[(\beta + \mu_b)^2 - (\beta - \mu_b)^2\big] / 2V = 2\mu_b |\beta| / V$ in log prior density. The penalty scales with the bin's typical effect and with how confident the fit is about $\beta$; a weak bin with $\mu_b$ near zero imposes almost nothing, a strong bin with a well-established $\mu_b$ pushes hard against discordant fits. The weight of the sign is therefore learned, like the group weights, not set.

**What the per-bin mean does.** $\mu_b$ is one number per bin, identified from the fitted coefficients of the peaks selected in that bin. It replaces the role $\lambda$ played in Models A, E and F (converting a hurdle size into an expected eQTL effect) with something that never touches the hurdle size: if the hurdle's magnitude tracks mediation strength, $\mu_b$ rises across bins; if not, the bins share a $\mu$; either way nothing drifts. It also gives a selected peak a small head start proportional to $\mu_b^2 / 2V$ in its Bayes factor when the fit agrees in sign, which is the correct Bayesian reward for a prior that predicted the direction.

**Sign groups.** With the sign in the effect prior, the concordant / discordant split of the group weights becomes redundant; the recommended configuration is significance bins only for the weights (`--no_sign_groups`), with the sign carried by $\mu_b\, \text{sign}(\hat g_k)$. Keeping both is possible and harmless.

## Inference

Standard SuSiE-RSS coordinate ascent over the augmented pool, with a non-zero-mean single-effect regression for peak elements. For element $e$ with OLS estimate $\hat\beta_e = S_e / P_e$ and sampling variance $s_e^2 = 1 / P_e$, prior $N(m_e, V)$:

- log Bayes factor: $\log N(\hat\beta_e;\ m_e,\ V + s_e^2) - \log N(\hat\beta_e;\ 0,\ s_e^2)$ (the ordinary SER Bayes factor when $m_e = 0$);
- posterior: $N\big(\nu_e (S_e + m_e / V),\ \nu_e\big)$ with $\nu_e = 1 / (P_e + 1/V)$;
- KL term for the effect: the Gaussian KL against $N(m_e, V)$, i.e. the usual expression with $(\mu - m_e)^2$ in place of $\mu^2$.

$V_l$ is optimized on the whole pool's marginal likelihood as now (the peak Bayes factors depend on $V$ through both terms). Rounds alternate gene refits with EM updates of, across genes and over active effects: the group weights $\pi_g$ (as in J); $\mu_b = \sum \alpha_{le}\, \text{sign}(\hat g_{k(e)})\, E[\beta_{le}] / \sum \alpha_{le}$ over peak elements in bin $b$, floored at 0; and, if per-bin variances are used, $V_b = \sum \alpha_{le} E[(\beta_{le} - \mu_b \text{sign}(\hat g_{k(e)}))^2] / \sum \alpha_{le}$. A pseudo-count shrinks $\mu_b$ toward 0 (or toward the pooled mean over bins) so a bin with few selected peaks cannot swing. Convergence: group weights, $\mu_b$ (and $V_b$) settling.

Cost: Model J's, plus the mean shift in the peak branch of the SER.

## Initialization

As in J. $\mu_b = 0$ for every bin at the start, so the first round is exactly Model J; the means are then learned from the peaks the first round selected. This makes K's departure from J visible round by round.

## Outputs

As in J, plus per bin the learned $\mu_b$ (and $V_b$), and per selected peak its fitted coefficient, its bin's mean, and the posterior probability that its sign agrees with the hurdle, $P(\beta_l \cdot \text{sign}(\hat g_k) > 0)$ from the Gaussian posterior. The last is the per-peak concordance readout that J's group weights only gave in aggregate.

## Identifiability and limitations

- $\mu_b$ and $\pi_g$ are separately identified: $\pi_g$ from how often a bin's peaks are selected, $\mu_b$ from how large their fitted effects are once selected. Neither involves the hurdle's magnitude.
- The head start a signed mean gives to concordant fits, and the penalty to discordant ones, both scale with $\mu_b^2 / V$; with a shared $V_l$ that is set by the strongest effects in the gene, so the sign's leverage is modest at genes with strong variant effects and larger at genes with weak ones. Per-bin variances make it uniform.
- Collinearity remains: a concordant peak on a direct causal variant is still a mixed credible set decided by prior weight, now with the extra $\mu_b^2 / 2V$ in the peak's favor. A discordant peak on a direct causal variant is pushed toward the variant in proportion to $\mu_b$, which is the soft version of what the gate did by fiat.
- A systematically wrong hurdle sign for a class of peaks (e.g. repressive elements scored as activating) would be penalized rather than excluded, and would show up as a bin with a low posterior sign agreement; that is the diagnostic to watch.

## Relation to the other models

| | J | H (gate) | I-a (half-normal) | F (soft tie) | K |
|---|---|---|---|---|---|
| sign enters as | group weight | pool gate | hard sign on coefficient | center at $g_k / \lambda$ | signed prior mean $\mu_b$ |
| strength of sign | learned (how often) | fixed | fixed | via $\tau_\beta^2$ | learned ($\mu_b$) |
| effect size from hurdle | none | none | none | per link, needs $\lambda$ | per bin, no hurdle size |
| closed form | yes | yes | truncated normal | yes | yes |
| new parameters | | | | $\tau_\beta^2$, $\lambda$ | $\mu_b$ (and $V_b$) |

## Evaluation plan

1. Synthetic: the J sets plus the discordant set. K should reject discordant peaks in proportion to the learned $\mu_b$ (strongly in the strong bin), match J on concordant mediation, and recover $\mu_b$ near the simulated standardized effect.
2. Real data, same screened gene set, starting from J's run: the trajectory of $\mu_b$ across bins (rising with hurdle strength or flat), the per-peak sign-agreement distribution, the change in discordant selections relative to J, and the threshold counts with gene-bootstrap CIs.

## Implementation notes

On the J code: the SuSiE-RSS class needs a per-element prior mean vector in its single-effect regression (Bayes factor, posterior mean, KL), passed from the runner as $\mu_{b(k)}\, \text{sign}(\hat g_k)$ for peak elements and 0 for variants; the J model adds the $\mu_b$ (and $V_b$) M-steps to its round loop and the extra outputs. Everything else carries over.
