# Proposed additional model: spurious-link contamination mixture

**Status: implemented, tried on real data, and replaced by Model D** (`caqtl_mediated_fine_mapping_model_proposal_inclusion.md`). On real data the EM for $\rho$ was driven by the null bulk of small links (the spurious branch is a wide component, and most links are small), so $\rho \to 0$ and the fit converged back to Model A. The code has been removed. Original note on the implementation: it was in `ca_qtl_mediated_susie.py` (constructor flag `spurious_links`) and exposed in `run_ca_qtl_mediated_eqtl_finemapping.py` as `--spurious_links` (with `--spurious_link_prob_init`, `--no_estimate_spurious_link_prob`, `--spurious_link_variance`). Requires `--no_estimate_link_scale --no_estimate_link_bias_variance`. The shared-hyperparameter file gains `spurious_link_prob` and `spurious_link_variance`; the components file gains `link_spurious_probs`.

Companion to `caqtl_mediated_fine_mapping_model.md` (**Model A**, the current model) and `caqtl_mediated_fine_mapping_model_proposal_link_prior.md` (**Model B**, links as a prior only). This document proposes **Model C**, which keeps Model A's measurement of the link effect but lets a hurdle estimate be large and significant while the true link is zero.

## Why

In Model A the link term is $\hat g_k \sim N(\lambda g_k,\ se_k^2 + \tau_u^2)$. Under "no link" ($g_k = 0$) this leaves $\hat g_k \sim N(0, se_k^2)$, so a hurdle z-score of 5 is essentially impossible without a link and $q_k$ is forced to 1 before the eQTL data are consulted. Every significant hurdle link becomes a mediating link, whatever expression says.

Model C adds a second way to get a large $\hat g_k$: with some probability the hurdle estimate is spurious and says nothing about $g_k$. The evidence a hurdle link can supply for mediation is then bounded, and the eQTL data decide the rest.

## Model C

Per gene, unchanged from Model A:

$$A_k = X b_k + e_k, \qquad E = X\Big(\sum_k g_k b_k + d\Big) + e, \qquad g_k = z_k h_k, \quad z_k \sim \text{Bern}(\pi), \quad h_k \sim N(0, \tau_g^2)$$

with $b_k$ and $d$ sums of SuSiE single effects. The link term becomes a two-branch mixture with a per-link indicator $o_k$ ("spurious"):

$$o_k \sim \text{Bern}(\rho)$$

$$\hat g_k \mid o_k = 0 \sim N(\lambda g_k,\ se_k^2) \qquad \text{(a real measurement of the mediated effect, as in Model A)}$$

$$\hat g_k \mid o_k = 1 \sim N(0,\ se_k^2 + \tau_s^2) \qquad \text{(spurious: unrelated to } g_k\text{)}$$

- $\rho$: fraction of hurdle links that are spurious with respect to mediation, shared across genes, estimated by EM.
- $\tau_s^2$: spread of spurious-but-significant hurdle estimates. Fixed, at the empirical variance of the significant hurdle estimates (or at $\lambda^2 \tau_g^2$, see below).
- $\lambda$: fixed from data as now (Model A's EM estimate of $\lambda$ is degenerate; nothing here changes that).
- $\tau_u^2$: dropped (the spurious branch is a heavy-tailed replacement for it; a uniform variance inflation is no longer needed).
- $\pi, \tau_g^2$: shared, EM, as now.

## What it does to the evidence

Under Model A the log Bayes factor for "link" vs "no link" from the hurdle estimate grows like $\hat g_k^2 / se_k^2$ without bound. Under Model C it saturates: for large $|\hat g_k|$ both hypotheses explain it, one through $\lambda g_k$ and one through the spurious branch, so

$$\log \text{BF}_k \to \log\frac{1-\rho}{\rho} + \text{a density ratio that stops growing in } |\hat g_k|.$$

A strong hurdle link supplies a bounded amount of evidence. If the eQTL side says $g_k \approx 0$, the model can attribute $\hat g_k$ to the spurious branch at a bounded cost and $q_k$ comes down. A mediated PIP near 1 at a variant with no eQTL support requires the eQTL side to be genuinely uninformative rather than contradictory.

## Inference changes

Only the link update and the link part of the ELBO change.

- **Link posterior.** The variational factor for link $k$ is over $(z_k, o_k, h_k)$: four $(z_k, o_k)$ combinations, each with a Gaussian conditional for $h_k$ (score and precision from the eQTL side plus, for $o_k = 0$, the link term $\lambda \hat g_k / se_k^2$ and $\lambda^2 / se_k^2$). Closed form; $q_k = P(z_k = 1)$ marginalizes over $o_k$.
- **M-steps.** $\rho$ = mean of the spurious responsibilities across all links. $\pi$, $\tau_g^2$ as now, from the $z_k = 1$ responsibilities.
- **ELBO.** The link log-likelihood becomes the expected log of the two-branch mixture under the variational factor; the KL gains a Bernoulli term for $o_k$.

Cost is small and confined to `spike_slab_posterior`, `link_terms`, and `update_link_hyperparameters` in the model file. The caQTL and direct eQTL updates, the runner, and the output formats are unchanged. Outputs gain the per-link spurious probability, which is itself useful: "this hurdle link is significant but has no mediation support."

## Identifiability

From the hurdle data alone, a real link and a spurious one are both "a wide component around zero," so $\rho$ against $\pi$ is not identified by the link likelihood. It is identified only through the eQTL side: real links carry mediated effects consistent with expression, spurious ones do not. This is the intended design, but it means $\rho$ is learned from the same weak signal that made $\lambda$ drift. Hence:

- fix $\tau_s^2$ rather than estimate it (estimating both $\rho$ and $\tau_s^2$ adds a flat direction);
- keep $\lambda$ fixed;
- $\rho$ is then the only new free parameter, a proportion with a well-behaved EM update. Its convergence should be checked on the 300-gene set; if it drifts, fix it too (a sensitivity grid of 0.3, 0.5, 0.7).

## Relation to Model B

With $\tau_s^2 = \lambda^2 \tau_g^2$ the two branches are indistinguishable on the hurdle scale, and Model C reduces to Model B with a two-level prior: significant links get prior probability $\pi(1-\rho) / (\pi(1-\rho) + \rho)$-ish of being real, non-significant links a lower one. Model C is therefore the middle ground:

| | Model A | Model C | Model B |
|---|---|---|---|
| hurdle effect size informs $g_k$ | yes | yes, when not spurious | no |
| hurdle sign informs $g_k$ | yes | yes, when not spurious | no (extension) |
| link evidence bounded | no | yes | yes |
| needs $\lambda$ | yes (fixed) | yes (fixed) | no |
| new shared parameters | | $\rho$ | $a, b$ |

Choose C over B if the hurdle effect sizes are believed calibrated enough to be worth using; choose B if not. Running both on the same genes settles it.

## Evaluation plan

Run A (fixed $\lambda$, $\tau_u^2 = 0$), C, and optionally B on the same genes and compare:

1. Link posteriors $q_k$ against hurdle $|t_k|$: A is a step function; C should show significant links with low $q_k$ where the eQTL side disagrees.
2. Per-link spurious probabilities: their distribution among significant links, and whether they concentrate at peaks whose caQTL fine-mapping is LD-inconsistent (from the screen) or far from the gene.
3. Mediated PIP against eQTL-only PIP: C should have far fewer variants with mediated PIP near 1 and eQTL-only PIP near 0.
4. Counts of variants above PIP 0.9 in genes with peaks, with gene-bootstrap CIs.
5. Estimated $\rho$: should be stable across iterations and plausible (a substantial fraction of significant hurdle links not mediating in this cell type).
6. Simulations from Model A's generative process with a fraction of links replaced by spurious ones, checking that C recovers the mediating peaks and A does not.
