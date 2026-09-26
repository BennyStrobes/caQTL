# Chromatin-informed SuSiE fine-mapping of eQTLs: the simple version (implemented)

Ben Strober, 2026-09-21

## 1. Summary

Implemented in `run_chromatin_informed_susie_finemapping_of_eqtls.py`, step 5 of `run_fine_mapping.sh`. Results are compared with standard SuSiE by `compare_susie_finemapping_results.py`.

The model is standard SuSiE on a gene's eQTL summary statistics. The only change is the prior probability that a single effect sits at each variant: a variant that has a chromatin path to the gene, in the direction the chromatin data predict, is $e$ times more likely to carry an effect. Nothing else in SuSiE changes, so the fit handles several causal variants, gives credible sets, and keeps PIPs valid under LD. With $e = 1$ it is exactly `run_standard_susie_finemapping_of_eqtls.py`.

## 2. Inputs

For one gene with $p$ cis variants:

- eQTL marginal effects $\hat\beta_j$ with standard errors $s_j$, and the LD matrix $R$. The likelihood is the one used by the standard SuSiE script (z-score model unless a sample size is given), so the two sets of results differ only in the prior.
- From the chromatin-only results (`run_chromatin_only_expression_prediction.py`, no eQTL data used), for every variant $j$:
  - $\rho_j$, the probability that $j$ has a chromatin path: it is a caQTL variant for a peak that has a peak-gene link,
    $$\rho_j = 1 - \prod_k\big(1 - a_k\,PIP^c_{kj}\big),$$
    with $a_k$ the posterior probability that link $k$ is non-zero and $PIP^c_{kj}$ the SuSiE caQTL PIP of variant $j$ for peak $k$ (column `eqtl_alpha` of that file);
  - $\bar m_j = \sum_k E[g_k]\,E[b_{kj}]$, the chromatin-predicted eQTL effect, (peak-gene effect) $\times$ (caQTL effect) summed over peaks (column `eqtl_posterior_mean`). Only its sign is used.

## 3. Model

**Gated path probability.** A variant's path counts only if it is not negligible and the predicted direction agrees with the direction of the variant's marginal eQTL effect:

$$\tilde\rho_j = \begin{cases} \min(\rho_j, 1) & \text{if } \rho_j \ge 0.01 \text{ and } \text{sign}(\bar m_j) = \text{sign}(\hat\beta_j), \\ 0 & \text{otherwise.}\end{cases}$$

The threshold is `--min_path_prob`. The sign gate is switched off by `--no_sign_gate`.

**Prior.** For each of the $L$ single effects of SuSiE,

$$P(\gamma_l = j) = \frac{1 + (e - 1)\,\tilde\rho_j}{\sum_{i=1}^{p}\big[1 + (e - 1)\,\tilde\rho_i\big]}, \qquad b_l \sim N(0, V_l),$$

with $V_l$ estimated for each effect as usual. In code this is the `prior_weights` argument of `SUSIE_RSS.fit`. $e$ is the enrichment: the ratio of the prior probability of carrying an effect for a variant with a concordant path to that for a variant without one. It is shared across genes.

**Everything else** (the single-effect updates, the estimation of $V_l$ and its use to switch unused effects off, PIPs, credible sets and their purity filter) is standard SuSiE.

A gene with no chromatin results, or with $\tilde\rho_j = 0$ everywhere, has uniform prior weights and is fit by standard SuSiE whatever $e$ is.

## 4. The enrichment

**Fixed.** `--enrichment 20` fixes $e$ and fits every gene once.

**Learned.** By default $e$ is estimated across genes by expectation-maximization, starting from $e = 1$ (standard SuSiE fits). Given the current fits, the update maximizes the expected log prior of the locations of the single effects that are in use ($V_l > 0$):

$$e \leftarrow \arg\max_e \sum_g\Big[\sum_{l:\,V_l > 0}\sum_j \alpha_{lj}\log\big(1 + (e - 1)\tilde\rho_j\big) \;-\; n_g\log\Big(p_g + (e - 1)\sum_i\tilde\rho_i\Big)\Big],$$

with $n_g$ the number of effects in use in gene $g$. It is a one-dimensional search over $\log e$. Genes with a path are then refit with the new $e$, and the two steps alternate until $\log e$ changes by less than 0.01 or `--max_em_rounds` (10) is reached. Genes without a path are fit once.

**Upper bound.** The search is bounded above by `--max_enrichment` (default 30), and the script prints a note when the bound is reached. The reason is that this EM is self-reinforcing: a larger $e$ concentrates the posteriors on path variants, which raises $e$ again. In simulation, left unbounded, it climbed to 75 in ten rounds and was still rising, and the top PIP bin became over-confident (Section 6). The cause is that the path probabilities are over-stated for links that are not real, and a large $e$ then pulls eQTL signals onto those variants. On real data the peak-gene links are co-accessibility estimates and not mediation probabilities, so the same applies. Treat $e$ as a setting and not an estimate: run it at 5, 10 and 20 and check that the conclusions hold.

## 5. Outputs

One line per gene, with the columns of the standard SuSiE results file (`gene_id`, `variant_ids_file`, `pips`, `credible_sets`, `converged`, `residual_variance`, `eqtl_posterior_mean_file`) followed by `enrichment` and `gated_path_probs` ($\tilde\rho_j$ for every variant).

`compare_susie_finemapping_results.py baseline_file informed_file` prints, separately for genes with and without a gated path: the number of credible sets and their sizes, the numbers of variants with PIP above 0.9 and above 0.5, and the size of the PIP changes.

## 6. Checks so far (simulation only)

150 simulated genes, 60 variants each with neighbouring variants at $r = 0.85$, three peaks per gene, and mediated eQTL effects in most genes. This setting is favourable to the method. Not yet run on real data.

- With `--enrichment 1`, PIPs and credible sets are identical to the standard SuSiE script (largest PIP difference exactly 0).
- PIP above 0.9:

| setting | variants | mean PIP | fraction truly causal |
|---|---|---|---|
| standard SuSiE | 8 | 0.95 | 1.00 |
| $e = 20$, sign gate | 47 | 0.95 | 1.00 |
| $e = 20$, no sign gate | 39 | 0.95 | 1.00 |
| $e$ learned without a bound (75 after 10 rounds) | 67 | 0.96 | 0.91 |

- PIP between 0.5 and 0.9, at $e = 20$ with the sign gate: mean PIP 0.72 and 0.57 truly causal (standard SuSiE: 0.69 and 0.67). So the middle of the PIP range is somewhat over-confident even at a moderate $e$.
- With $e$ learned under the default bound of 30: credible sets went from 25 (median size 3) to 37 (median size 1).

## 7. Known limitations

1. **The sign gate uses the eQTL data twice.** It looks at the sign of the marginal eQTL effect before fitting, and the fit then uses the same statistics. It also uses the marginal sign, which can differ from the conditional sign for a secondary signal. The softer alternatives (a sign preference inside the Bayes factor, or a prior mean in the predicted direction) are in `chromatin_informed_eqtl_fine_mapping_model.md` and `joint_eqtl_caqtl_susie_model.md`.
2. **Path probabilities are taken at face value.** They come from caQTL PIPs and from link posteriors that measure co-accessibility. Where they are over-stated, the prior is wrong in proportion to $e$.
3. **Only the direction of the prediction is used**, not its size.
4. **No joint fine-mapping.** The caQTL fit is done first on its own data and is not sharpened by the eQTL data. If the caQTL cannot tell LD partners apart, $\rho_j$ is spread over them and the prior favours none, so the gain comes only where the caQTL is better resolved than the eQTL.
5. **Shared donors.** Expression and accessibility come from the same people, so noise in the eQTL and caQTL estimates at a variant is correlated, with the sign of the link. This can create spurious concordant paths at weak caQTLs. The threshold on $\rho_j$ and SuSiE's caQTL PIPs limit it, but it is not modelled. The joint model in `inherited_effects_joint_fine_mapping_model.md` has a residual correlation for this.
6. **A high enrichment is not evidence of mediation.** caQTLs and eQTLs both concentrate in regulatory DNA.

## 8. What to look at on real data

In the "genes with a variant with a chromatin path" block of the comparison: whether credible sets shrink and more variants pass PIP 0.9, how much that depends on $e$ (5, 10, 20), and whether the variants whose PIP rises are enriched in functional readouts that do not come from open chromatin. If credible sets barely move, the caQTL signals are not better resolved than the eQTL signals at these loci, and the more elaborate models would not change that.
