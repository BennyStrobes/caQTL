# Chromatin-informed eQTL fine-mapping with a per-variant prior: proposed model

Ben Strober, 2026-09-21

## 1. Summary

This is a proposal, not yet implemented. It fine-maps a gene's eQTLs with SuSiE (several causal variants, valid PIPs under LD) and changes only the prior of each variant, using two things that chromatin data say about it:

1. **whether the variant has a chromatin path to the gene**: it is a caQTL variant for a peak that has a peak-gene link. Variants with a path are $e$ times more likely to carry an eQTL effect. $e$ is learned across genes.
2. **the direction the chromatin data predict**: the sign of (peak-gene link) $\times$ (caQTL effect). An eQTL effect on a path variant agrees with that sign with probability $\kappa$. $\kappa$ is learned across genes.

Effect sizes are never tied to the size of the link, because link sizes do not line up with eQTL effects across genes (the lesson of Models A to F). Only the link's significance and sign are used. With $e = 1$ and $\kappa = 1/2$ the model is exactly standard SuSiE.

The chromatin quantities come from stage 1 of the expression prediction work (`chromatin_only_expression_prediction.py`: SuSiE on each peak's caQTL data, spike and slab on the links). They are held fixed. Nothing flows back from the eQTL data into the caQTL or link fits.

## 2. Inputs for one gene

- eQTL marginal effects $\hat\beta$ with standard errors $s$, and the LD matrix $R$, over the gene's $p$ cis variants. The likelihood is the one used by `run_standard_susie_finemapping_of_eqtls.py`, so that the two sets of results differ only in the prior.
- From stage 1, for every variant $j$:
  - $\rho_j$, the probability of a chromatin path:
    $$\rho_j = 1 - \prod_k \big(1 - a_k\,PIP^c_{kj}\big),$$
    with $a_k$ the posterior probability that link $k$ is non-zero and $PIP^c_{kj}$ the caQTL PIP of variant $j$ for peak $k$.
  - $\tau_j$, the probability that the predicted effect $g_k b_{kj}$ is positive, given a path. For one peak,
    $$\tau_{kj} = P(g_k > 0)\,P(b_{kj} > 0) + P(g_k < 0)\,P(b_{kj} < 0),$$
    with $P(g_k > 0) = \Phi(\nu_k / t_k)$ from the link posterior and $P(b_{kj} > 0)$ from the caQTL posterior at $j$. Over several peaks, $\tau_j = \sum_k r_{kj}\tau_{kj} / \sum_k r_{kj}$ with $r_{kj} = a_k PIP^c_{kj}$. A weak link sign or a weak caQTL sign gives $\tau_j$ near $1/2$, which carries no sign information.

  The current chromatin-only results file already holds $\rho_j$ (column `eqtl_alpha`) and the mean and variance of the predicted effect given a path, from which $\tau_j \approx \Phi\big(\mu_j / \sqrt{\omega_j}\big)$ without rerunning stage 1.

## 3. Model

The eQTL effects are a sum of $L$ single effects, $\beta = \sum_{l=1}^{L} b_l\, \mathbf{1}_{\gamma_l}$, as in SuSiE. For each single effect:

**Where it is.** Each variant is a candidate with or without a path. The prior weight of "variant $j$, no path" is $1 - \rho_j$ and of "variant $j$, path" is $e\,\rho_j$. So

$$P(\gamma_l = j) = \frac{1 + (e - 1)\rho_j}{\sum_i \big[1 + (e - 1)\rho_i\big]}, \qquad \tilde\rho_j = P(c_l = 1 \mid \gamma_l = j) = \frac{e\,\rho_j}{1 + (e - 1)\rho_j}.$$

$e$ is the ratio of the prior probability of carrying an effect for a variant with a path to that for a variant without one.

**How big it is.**

- No path ($c_l = 0$): $b_l \sim N(0, V_l)$, as in standard SuSiE.
- Path ($c_l = 1$): the same size distribution with a preferred sign. With probability $\kappa$ the effect has the predicted direction and with probability $1 - \kappa$ the opposite one, and the predicted direction is positive with probability $\tau_j$. So the effect is positive with probability
  $$p^+_j = \kappa\,\tau_j + (1 - \kappa)(1 - \tau_j),$$
  and $b_l \sim p^+_j\, N^+(0, V_l) + (1 - p^+_j)\, N^-(0, V_l)$, with $N^\pm$ the half-normals on the positive and negative half-lines.

Integrating over the path indicator, the prior density of an effect at variant $j$ is a normal reweighted on its two halves:

$$p(b \mid \gamma_l = j) = N(b; 0, V_l)\,\big[a^+_j\,1(b > 0) + a^-_j\,1(b < 0)\big], \qquad a^\pm_j = 1 \pm \tilde\rho_j s_j,$$

$$s_j = 2p^+_j - 1 = (2\kappa - 1)(2\tau_j - 1) \in [-1, 1].$$

$s_j$ is the signed confidence: how sure the model is of the effect's sign, and which sign. Every prior is centred at zero, so an unused effect is switched off by $V_l \to 0$ in the usual way. The degenerate mode of Model K (a non-zero prior mean with $V_l \to 0$) cannot occur.

**Shared across genes:** $e$ and $\kappa$ only.

## 4. Single-effect update

Let $\hat b_j$ and $\hat s_j^2$ be the effect of variant $j$ on the current residual and its sampling variance (SuSiE's `betahat` and `shat2`). The normal-prior quantities are unchanged:

$$v_j = \Big(\frac{1}{V_l} + \frac{1}{\hat s_j^2}\Big)^{-1}, \qquad \hat\mu_j = v_j\,\frac{\hat b_j}{\hat s_j^2}, \qquad t_j = \frac{\hat\mu_j}{\sqrt{v_j}}, \qquad BF^N_j = \sqrt{\frac{\hat s_j^2}{V_l + \hat s_j^2}}\exp\Big(\frac{\hat b_j^2}{2\hat s_j^2}\,\frac{V_l}{V_l + \hat s_j^2}\Big).$$

**Bayes factor.** Integrating the likelihood against the reweighted normal gives the standard Bayes factor times one factor:

$$BF_j = BF^N_j \cdot F_j, \qquad F_j = a^+_j\Phi(t_j) + a^-_j\Phi(-t_j) = 1 + \tilde\rho_j\, s_j\,\big(2\Phi(t_j) - 1\big).$$

**Posterior over variants.**

$$\alpha_{lj} \propto \big[1 + (e - 1)\rho_j\big]\; BF^N_j\; F_j.$$

**Posterior moments of the effect given variant $j$** (a normal reweighted on its two halves):

$$E[b \mid j] = \hat\mu_j + \frac{2\tilde\rho_j s_j\,\sqrt{v_j}\,\phi(t_j)}{F_j}, \qquad E[b^2 \mid j] = \hat\mu_j^2 + v_j + \frac{2\tilde\rho_j s_j\,\hat\mu_j\sqrt{v_j}\,\phi(t_j)}{F_j}.$$

These replace `mu` and `mu2` in `SUSIE_RSS`. The rest of the algorithm (removing and adding back effects, the ELBO, PIPs, credible sets) is unchanged, because SuSiE only needs the single-effect log Bayes factor $\log\sum_j P(\gamma = j)\,BF_j$ and these two moments.

**Prior variance.** $V_l$ maximizes $\log \sum_j P(\gamma = j)\,BF^N_j(V)\,F_j(V)$ over $V$, by the same one-dimensional search `SUSIE_RSS` uses now, with $V_l = 0$ kept when no effect is at least as likely.

## 5. What each piece does

- $F_j \in [0, 2]$. A path variant whose effect on the residual confidently agrees with the prediction has its Bayes factor at most doubled. One that confidently disagrees has it driven toward zero.
- Within one eQTL signal, LD partners usually share the sign of their effect on the residual, so the sign rarely separates them directly. Its main job is to gate the enrichment: a caQTL whose predicted direction disagrees with the eQTL stops attracting the eQTL signal. About half of chance colocalizations have the wrong sign, so that route to a false positive is roughly halved. A concordant caQTL keeps the full enrichment and gains up to a factor of two.
- If the caQTL cannot tell five LD partners apart, $\rho_j$ is spread evenly over them and the prior favours none. The prior sharpens an eQTL signal only where the caQTL is better resolved than the eQTL.

## 6. Learning $e$ and $\kappa$ across genes

Expectation-maximization over all genes, using effects with $V_l > 0$.

**Posterior of the latent indicators** for effect $l$ at variant $j$:

$$\bar c_{lj} = P(c_l = 1 \mid \gamma_l = j, \text{data}) = \frac{\tilde\rho_j\,\big[1 + s_j(2\Phi(t_j) - 1)\big]}{F_j},$$

$$\bar h_{lj} = P(\text{concordant} \mid c_l = 1, \gamma_l = j, \text{data}) = \frac{\kappa\,u_j}{\kappa\,u_j + (1 - \kappa)(1 - u_j)}, \qquad u_j = \tau_j\Phi(t_j) + (1 - \tau_j)\Phi(-t_j).$$

**Update of $\kappa$** (closed form):

$$\kappa = \frac{\sum_{g,l}\sum_j \alpha_{lj}\,\bar c_{lj}\,\bar h_{lj}}{\sum_{g,l}\sum_j \alpha_{lj}\,\bar c_{lj}}.$$

**Update of $e$** (one-dimensional). With $n_g$ the number of effects of gene $g$ that are in use, $r_g = \sum_l\sum_j \alpha_{lj}\bar c_{lj}$ the expected number of them on a path, $p_g$ the number of variants and $P_g = \sum_j \rho_j$, the expected log prior is

$$\sum_g \Big[\, r_g \log e - n_g \log\big(p_g - P_g + e\,P_g\big) \Big],$$

maximized over $\log e$ on a bounded interval.

**Algorithm.** Start from standard SuSiE fits ($e = 1$, $\kappa = 1/2$). Alternate: fit every gene with the current $(e, \kappa)$; update $(e, \kappa)$. Stop when both change by less than a tolerance. Do not import the enrichment learned by the prediction model: that was an inclusion probability from a mean-field fit, and this is a location prior.

## 7. Outputs

- PIPs and credible sets, in the format of the eQTL-only SuSiE results.
- For each effect in use: the probability that it lies on a chromatin path, $\sum_j \alpha_{lj}\bar c_{lj}$, and the probability that it agrees in sign with the prediction.
- $e$ and $\kappa$. Together they measure how much co-accessibility links plus caQTLs say about where eQTLs are and which way they point.

## 8. Special cases

- $e = 1$, $\kappa = 1/2$: standard SuSiE, exactly. This is the baseline and the first implementation check.
- $\kappa = 1/2$: SuSiE with chromatin-informed prior weights only.
- $\tau_j \in \{0, 1\}$ and $\kappa = 1$: a hard sign constraint on path effects, as in Models G and I-a but on variants.

## 9. Relation to the earlier models

Models E, I and J put each peak in the SuSiE pool as one element with a free coefficient, and Models G and I-a put the link's sign on that coefficient. This model instead puts both the enrichment and the sign on each variant, which is what results from passing a peak element down to its caQTL variants when the path runs through one variant. It gives variant-level PIPs directly and needs nothing beyond `SUSIE_RSS`. It gives up two things the element models have: a caQTL spread over LD partners is no longer one predictor, and agreement across a peak's several caQTL signals is not used.

## 10. Known issues and open decisions

1. **Shared donors.** Expression and accessibility come from the same people, so the noise in an eQTL estimate and in a caQTL estimate at the same variant are correlated, with the sign of the link. At weak caQTLs this can create spurious sign agreement. $\rho_j$ is small there, which limits it. Checks: $\kappa$ estimated separately by caQTL strength, and a simulation with correlated noise. The principled remedy is a joint likelihood for expression and accessibility with a residual correlation.
2. **Where the gain can come from.** Only loci where the caQTL is better resolved than the eQTL. Check first: at loci where an eQTL credible set overlaps a caQTL credible set of a linked peak, compare credible-set sizes and lead z-scores.
3. **Links are not mediation.** A high $e$ partly reflects that caQTLs and eQTLs both concentrate in regulatory DNA. It helps fine-mapping either way, but it is not evidence that the peak mediates the gene.
4. **Predicted effect size is unused.** It could enter later as a further slab component centred on $\lambda\mu_j$, if it proves to line up across genes under the current standardization.
5. **Definition of $\tau_j$.** The exact per-peak form in Section 2 needs a small addition to stage 1. The normal approximation from the existing results file needs none.
6. **Genes with no linked peaks.** $\rho_j = 0$ for every variant, the prior weights are uniform, and the fit is standard SuSiE, as it should be.

## 11. Validation plan

1. With $e = 1$ and $\kappa = 1/2$, reproduce `SUSIE_RSS` exactly (alphas, PIPs, ELBO).
2. Simulate eQTL summary statistics from the real LD matrices and the real $(\rho_j, \tau_j)$, with causal variants placed at a known $e$ and $\kappa$, including $e = 1$ and $\kappa = 1/2$. Check PIP calibration, credible-set coverage and size, and recovery of $e$ and $\kappa$. PIPs must stay calibrated when the prior is uninformative.
3. The same with noise correlated between expression and accessibility, to bound issue 1.
4. Real data, against eQTL-only SuSiE, within genes that have a chromatin path: fraction of credible sets that shrink, number of variants with PIP above 0.9, enrichment of high-PIP variants in functional readouts that do not come from open chromatin (reporter-assay allelic effects, sequence-model predictions), and replication in independently fine-mapped eQTLs.
