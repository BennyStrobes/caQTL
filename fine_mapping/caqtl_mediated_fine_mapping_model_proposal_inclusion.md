# Proposed additional model: link inclusion indicator (Model D)

**Implemented** in `ca_qtl_mediated_susie.py` (constructor flag `link_inclusion`) and exposed in `run_ca_qtl_mediated_eqtl_finemapping.py` as `--link_inclusion` (with `--link_inclusion_prob_init`, `--no_estimate_link_inclusion_prob`). As implemented, $u_k$ is defined conditional on $z_k = 1$, so $\pi_u$ is the fraction of non-zero links that mediate and its M-step is $\sum_k P(z_k u_k = 1) / \sum_k P(z_k = 1)$. The shared-hyperparameter file gains `link_inclusion` and `link_inclusion_prob`; the components file gains `link_inclusion_probs`, which is what the mediated PIP uses.

Companion to `caqtl_mediated_fine_mapping_model.md` (**Model A**, the current model), `caqtl_mediated_fine_mapping_model_proposal_link_prior.md` (**Model B**, links as a prior only) and `caqtl_mediated_fine_mapping_model_proposal_spurious_links.md` (**Model C**, spurious-link mixture, implemented). Model D keeps Model A's link term unchanged and adds a per-link indicator for whether the link actually enters the eQTL model.

## Why

In Model A the link effect $g_k$ has to satisfy the hurdle estimate and the expression data at the same time. The hurdle term has far more precision, so it wins: a significant hurdle link becomes a mediating link whatever expression says. Model C addressed this by letting the hurdle estimate be spurious, which introduced a second density for $\hat g_k$ (with $\rho$ and $\tau_s^2$) that competes with the first and is fit mostly by the null bulk of small links.

Model D separates the two roles of $g_k$. The hurdle measures the peak-gene link; a separate binary variable says whether that link mediates *this* eQTL. A strong hurdle link with no eQTL support is then simply not included, at no cost to either likelihood.

## Model D

Per gene:

$$A_k = X b_k + e_k \qquad \text{(caQTL of peak } k\text{, as in A)}$$

$$E = X\Big(\sum_k u_k\, g_k\, b_k + d\Big) + e \qquad \text{(eQTL: mediated terms enter only when } u_k = 1\text{)}$$

$$\hat g_k \sim N(\lambda g_k,\ se_k^2) \qquad \text{(link term, as in A; } \tau_u^2 \text{ optional as in A)}$$

$$g_k = z_k h_k, \quad z_k \sim \text{Bern}(\pi), \quad h_k \sim N(0, \tau_g^2) \qquad \text{(as in A)}$$

$$u_k \sim \text{Bern}(\pi_u) \qquad \text{(new: does link } k \text{ mediate the eQTL)}$$

- $g_k$: the peak-gene link as the hurdle regression measures it.
- $u_k$: whether that link mediates expression through the peak's caQTL. A peak mediates when $z_k u_k = 1$.
- $\pi_u$: shared across genes, estimated by EM. It is the fraction of hurdle links that mediate expression, identified from the eQTL side only.
- $\pi, \tau_g^2, \lambda$: as in A. $\lambda$ can stay fixed from data; see the identifiability note for why it may also be estimable here.

## What it does to the evidence

The posterior of $u_k$ is decided entirely by the expression data: it is the Bayes factor for adding the specific vector $g_k b_k$ (with $g_k \approx \hat g_k / \lambda$ from the hurdle) to the eQTL regression, against prior odds $\pi_u / (1 - \pi_u)$. The hurdle supplies the effect size and its sign; the eQTL decides inclusion. A link with a strong hurdle estimate and no eQTL support takes $u_k = 0$: the hurdle is fit perfectly by $g_k = \hat g_k / \lambda$, the eQTL sees nothing, and nothing is penalized.

Compared with Model C: C's spurious cell ($z_k = 1, o_k = 1$) means "the link is real for expression but the hurdle estimate is uninformative"; D's excluded cell ($z_k = 1, u_k = 0$) means "the hurdle estimate is informative about the link, but the link does not mediate this eQTL". D's is the intended semantics, and it needs no second density for $\hat g_k$.

## Inference changes

Only the link update, the link ELBO terms, and the M-steps change. The variational factor for link $k$ is over $(z_k, u_k, h_k)$ with three distinct cells ($u_k$ is irrelevant when $z_k = 0$):

| cell | evidence on $h_k$ | posterior of $h_k$ |
|---|---|---|
| $z_k = 0$ | none ($g_k = 0$) | — |
| $z_k = 1,\ u_k = 0$ | hurdle only: $S_h = \lambda \hat g_k / se_k^2$, $P_h = \lambda^2 / se_k^2$ | $N(m_h, v_h)$ |
| $z_k = 1,\ u_k = 1$ | hurdle + eQTL: $S_1 = S_E + S_h$, $P_1 = P_E + P_h$ | $N(m_1, v_1)$ |

with $S_E, P_E$ the eQTL-side score and precision already computed in Model A, $v = 1 / (1/\tau_g^2 + P)$ and $m = vS$. Cell log-weights are the prior log-probabilities plus $\tfrac12 \log(v / \tau_g^2) + \tfrac12 v S^2$ for the two $z_k = 1$ cells. Closed form, as in C.

Bookkeeping: the eQTL updates use $E[u_k g_k] = r_{11} m_1$ and $E[u_k g_k^2] = r_{11}(m_1^2 + v_1)$ (the included cell only); the hurdle likelihood and the $\tau_g^2$ M-step use $E[g_k]$ and $E[g_k^2]$ over both $z_k = 1$ cells. The mediated PIP becomes $P(z_k u_k = 1) \times$ caQTL PIP. M-steps: $\pi$ and $\tau_g^2$ as now (over $z_k = 1$ cells); $\pi_u$ = mean of $E[u_k]$ across links, where $E[u_k] = r_{11} + (1 - q_k)\pi_u$ (prior for the $z_k = 0$ cell).

Cost: similar to Model C. The caQTL and direct-eQTL updates, the runner, and the output formats are unchanged; outputs gain a per-link inclusion probability.

## Identifiability

- $\pi_u$ is identified purely from the eQTL side, as the fraction of links whose implied effect the expression data accept. This is a real estimand, unlike $\pi$ in A and C, which described the shape of the hurdle distribution.
- $\lambda$ becomes identifiable in principle. Excluded links carry no information about $\lambda$: $g_k = \hat g_k / \lambda$ is self-consistent for any value. Included links have $g_k$ pinned by expression, and $\hat g_k = \lambda g_k$ then calibrates $\lambda$. This is the "calibrate $\lambda$ on colocalized links" idea done inside the model. The EM update $\lambda = \sum_k \hat g_k E[g_k] / \sum_k E[g_k^2]$ is a mixture of "current $\lambda$" votes from excluded links and calibrated votes from included links, so it moves slowly when few links are included. Keep $\lambda$ fixed for the first runs and estimate it as a second step.
- The degenerate direction of A ($\lambda \to \infty$, $g \to 0$ to satisfy hurdle and eQTL-null together) is gone: eQTL-null links take $u_k = 0$ instead of pushing $\lambda$.

## Risks

- **$\lambda$ must start roughly right.** The implied effect $(\hat g_k / \lambda)\, b_k$ is what the eQTL tests. If $\lambda$ is off by a factor of two, every implied effect is off by that factor, inclusion is rejected, $\pi_u \to 0$, and with nothing included $\lambda$ never updates. Initialize from the moment estimate (as now) or from a colocalized subset, and watch $\pi_u$ over the first iterations: collapse to zero means the calibration is wrong, not the biology.
- **Inclusion is strict by design.** For a strong caQTL the penalty for including an effect the eQTL does not show is large ($\approx P_E m^2 / 2$, tens of nats), so D will call far fewer mediating links than A and the PIP-threshold counts will drop. That is the intended behaviour.
- **No slack in the implied effect.** D says the mediated effect is exactly $\hat g_k / \lambda$ up to hurdle noise. If real mediation is only proportional to the hurdle link on average, with link-specific deviations, D under-includes. Adding $\tau_u^2$ (as in A) is the corresponding slack; estimating it in D is safer than in A because excluded links no longer pull on it.
- The direct-versus-mediated competition within a gene is unchanged from A.

## Relation to the other models

| | A | B | C | D |
|---|---|---|---|---|
| hurdle effect size and sign inform the mediated effect | yes | no | when not spurious | yes |
| what decides mediation | hurdle (dominates) | eQTL, with hurdle as prior | eQTL vs a spurious density | eQTL, with hurdle-implied effect |
| link evidence bounded | no | yes | yes | yes (excluded at no cost) |
| needs $\lambda$ | fixed | no | fixed | fixed, estimable from included links |
| new shared parameters | | $a, b$ | $\rho, \tau_s^2$ | $\pi_u$ |
| interpretable "fraction mediating" | no | yes ($\sigma(a + b\lvert t\rvert)$) | partly ($1 - \rho$) | yes ($\pi_u$) |

D is the natural next step from A: same likelihoods, one extra Bernoulli, and the decision the model was getting wrong is handed to the data source that should make it.

## Evaluation plan

Run A (fixed $\lambda$) and D on the same genes and compare:

1. $\pi_u$ trajectory: stable and away from zero is the check that the hurdle-implied effects are compatible with expression at the chosen $\lambda$.
2. Inclusion probabilities against hurdle $|t_k|$ and against the eQTL Bayes factor: inclusion should track the eQTL evidence, not the hurdle strength.
3. Mediated PIP against eQTL-only PIP: D should have very few variants with mediated PIP near 1 and eQTL-only PIP near 0.
4. Counts of variants above PIP 0.9 in genes with peaks, with gene-bootstrap CIs.
5. Second step: estimate $\lambda$ from the included links and compare with the moment estimate.
6. Simulations from A's generative process with a fraction of links given $u_k = 0$, checking that D recovers the mediating peaks and A does not; and with $\lambda$ mis-specified by a factor of two, to measure how sensitive inclusion is to the calibration.
