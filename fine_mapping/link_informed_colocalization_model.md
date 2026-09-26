# Link-informed colocalization of eQTLs and caQTLs: proposed model

Ben Strober, 2026-09-21

## 1. Summary

This is a proposal, not yet implemented. It is standard single-causal-variant colocalization (coloc, Giambartolomei et al. 2014) between a gene's eQTL and the caQTL of each peak linked to the gene, with two additions:

1. **The enhancer-gene link sets the prior probability that the two traits share their causal variant.** How strongly it does so is learned across all gene-peak pairs.
2. **The caQTLs then fine-map the eQTL.** Marginalizing the colocalization model over each peak's causal variant gives the eQTL's causal-variant posterior in closed form: the eQTL evidence at each variant times a prior weight built from the caQTL posteriors and the links.

The direction of effect, sign(link $\times$ caQTL effect) against the sign of the eQTL effect, is not used by the base model, so it is available as an independent check. Section 7 gives the version that puts it inside the model.

No LD matrix is needed. Everything is computed from marginal effects and standard errors.

## 2. Data

For gene $g$ and its linked peaks $k = 1, \dots, K$, over the $Q$ variants tested for all of them:

- eQTL marginal effects $\hat\beta_j$ and variances $V^E_j = se_j^2$;
- caQTL marginal effects $\hat b_{kj}$ and variances $V^{A_k}_j$;
- for each peak, a link feature $x_k$: the estimated link $\hat g_k$ with its standard error, summarized for example by the bin of $|\hat g_k / se_k|$ or by the posterior probability that the link is non-zero.

## 3. Evidence at one variant

Wakefield's approximate Bayes factor for "variant $j$ is causal for trait $t$" against "no association", with prior effect variance $W_t$, $z = \hat\beta/\sqrt V$ and $r = W/(V + W)$:

$$ABF^t_j = \sqrt{1 - r}\;\exp\!\Big(\frac{z^2 r}{2}\Big).$$

Under the assumption of at most one causal variant per trait in the region, the likelihood of a trait's whole vector of marginal statistics given "variant $j$ is causal", relative to the null, is $ABF^t_j$. This is why no LD is needed.

## 4. Model for one gene-peak pair

Each trait has at most one causal variant. Let $c_E$ and $c_k$ be the causal variants of the eQTL and of peak $k$'s caQTL (or "none"). The prior follows coloc: a variant is causal for the eQTL only with probability $p_1$, for the caQTL only with probability $p_2$, and for both with probability $p_{12,k}$. The link enters through the last of these:

$$p_{12,k} = e(x_k)\;p_1\,p_2 .$$

$e(x_k)$ is the enrichment for sharing: how much more likely the two traits are to have the same causal variant than two unrelated signals would be. With links in bins, $e$ takes one value per bin; $\log e(x) = \theta_0 + \theta_1 x$ is the continuous version.

**Posterior odds of the five hypotheses against $H_0$** (no causal variant for either trait), with $S_E = \sum_j ABF^E_j$, $S_k = \sum_j ABF^{A_k}_j$ and $S_{Ek} = \sum_j ABF^E_j ABF^{A_k}_j$:

| hypothesis | meaning | odds against $H_0$ |
|---|---|---|
| $H_1$ | eQTL only | $p_1 S_E$ |
| $H_2$ | caQTL only | $p_2 S_k$ |
| $H_3$ | both, different variants | $p_1 p_2\,(S_E S_k - S_{Ek})$ |
| $H_4$ | both, the same variant | $e(x_k)\,p_1 p_2\,S_{Ek}$ |

Each posterior probability is its odds divided by one plus the sum of the four.

**The shared variant, given $H_4$:** $P(j \mid H_4, \text{data}) = ABF^E_j ABF^{A_k}_j / S_{Ek}$.

## 5. Using the caQTLs to fine-map the eQTL

Let $\omega_{kj}$ be the single-variant fine-mapping posterior of peak $k$'s caQTL from its own data, including the possibility of no caQTL:

$$\omega_{kj} = \frac{p_2\,ABF^{A_k}_j}{1 + p_2 S_k}.$$

Summing the colocalization model over every possible causal variant of every linked peak gives, exactly,

$$P(c_E = j \mid \text{data}) \;\propto\; ABF^E_j \;\prod_{k=1}^{K}\Big[\,1 + \big(e(x_k) - 1\big)\,\omega_{kj}\Big],$$

with "no eQTL" carrying weight $1/p_1$. (Checked against brute-force enumeration: agreement to $10^{-16}$.)

In words: the eQTL is fine-mapped by its own Bayes factors, with a prior weight on each variant that rises with the probability that the variant is the caQTL of a linked peak, by an amount set by the link. With $e = 1$ for every peak this is ordinary single-variant fine-mapping of the eQTL. Credible sets follow by ranking variants on this posterior.

This identity is also what connects the approaches in this folder. The prior weights $1 + (e - 1)\rho_j$ of the simple SuSiE version (`chromatin_informed_susie_simple_model.md`) are colocalization marginalized over the caQTL's causal variant, with the enrichment equal to $p_{12}/(p_1 p_2)$. Colocalization is the case of one causal variant per trait; the SuSiE version is the same prior with several.

## 6. Learning the priors across genes

Empirical Bayes over all gene-peak pairs, iterated to convergence. With $Q_{gk}$ the number of variants in the pair's region:

$$p_1 \leftarrow \frac{\sum (PP_{H_1} + PP_{H_3})}{\sum Q_{gk}}, \qquad p_2 \leftarrow \frac{\sum (PP_{H_2} + PP_{H_3})}{\sum Q_{gk}}, \qquad p_{12}(b) \leftarrow \frac{\sum_{\text{pairs in link bin } b} PP_{H_4}}{\sum_{\text{pairs in link bin } b} Q_{gk}},$$

and $e(b) = p_{12}(b)/(p_1 p_2)$. Each update sets the prior probability of a hypothesis to its average posterior probability. The curve $e(b)$ across link bins is a result in its own right: it measures how much an enhancer-gene link says about whether the gene's eQTL and the peak's caQTL are the same signal.

Fixed priors are the alternative. The defaults ($p_1 = p_2 = 10^{-4}$, $p_{12} = 10^{-5}$) were chosen for large regions. The prior odds of $H_4$ against $H_3$ are $e/(Q - 1)$, so the window size silently sets the prior: with the defaults ($e = 1000$) they are about 1 for 1,000 variants and about 3 for 300.

## 7. Direction of effect

**As a held-out check (recommended first).** The base model never looks at signs. Among pairs with a confident $H_4$, count how often sign(link) $\times$ sign(caQTL effect) equals the sign of the eQTL effect at the shared variant. The unit is the locus, not the variant-gene pair. A rate well above one half, rising with the posterior probability of $H_4$, is independent evidence that the colocalizations are real and that the links point the right way.

**Inside the model (optional).** Split $H_4$ into a concordant and a discordant version, with prior probabilities $\kappa\,p_{12,k}$ and $(1 - \kappa)\,p_{12,k}$. Under independent normal priors the two effects at variant $j$ have posteriors $N(m_t, v_t)$ with $m_t = r_t\hat\beta_t$ and $v_t = r_t V_t$; let $t_t = m_t/\sqrt{v_t}$ and let $o_k = \pm 1$ be the sign of the link. The posterior probability that the effects are concordant is

$$C_{kj} = \Phi(o_k t_E)\,\Phi(t_{A_k}) + \Phi(-o_k t_E)\,\Phi(-t_{A_k}),$$

and restricting the prior to the concordant (discordant) half of the plane multiplies the $H_4$ Bayes factor at that variant by $2C_{kj}$ (by $2(1 - C_{kj})$). So in every formula above $ABF^E_j ABF^{A_k}_j$ is replaced by

$$ABF^E_j\,ABF^{A_k}_j\;\cdot\;2\big[\kappa\,C_{kj} + (1 - \kappa)(1 - C_{kj})\big],$$

which is unchanged when $\kappa = 1/2$. $\kappa$ is learned as the share of the posterior $H_4$ mass that is concordant. A link whose sign is uncertain is handled by replacing $\kappa$ with $\kappa\tau_k + (1 - \kappa)(1 - \tau_k)$, where $\tau_k = P(g_k > 0)$. This version helps most where the eQTL signal is modest, because there a prior on the direction changes whether a shared variant is called at all. It gives up the held-out check.

## 8. Outputs

- For each gene-peak pair: the five posterior probabilities, the shared-variant posterior, and the direction of each effect at the shared variant.
- For each gene: the eQTL causal-variant posterior of Section 5 and its credible set, next to the credible set from the eQTL data alone. The change in credible-set size is the fine-mapping gain from chromatin, locus by locus.
- For each variant, peak and gene: the probability that the variant is the shared causal variant of that peak's caQTL and the gene's eQTL.
- Across genes: $p_1$, $p_2$, the enrichment curve $e(b)$, the fraction of eGenes with a colocalizing linked peak, and the sign-concordance rate among colocalized pairs.

## 9. Assumptions and known issues

1. **One causal variant per trait in the region.** When a trait has several signals the strongest dominates, and a shared secondary signal is read as "different variants". This is more defensible for a caQTL in a small window around its peak than for an eQTL. The standard extension (coloc with SuSiE) applies the same formulas to every pair of SuSiE single effects, one from each trait, using each effect's per-variant Bayes factors; this needs the LD-based SuSiE fits that already exist. Run the single-variant version everywhere, the SuSiE version at genes with several eQTL signals, and compare.
2. **Shared donors.** Expression and accessibility come from the same people, so the two traits' statistics are correlated under the null, and the product of Bayes factors overstates the evidence for a shared variant. The inflation grows with the phenotypic correlation of the pair, which is what the link measures, so a rise of $e$ with link strength is partly this artifact. Controls: distance-matched unlinked peaks for the same genes; a simulation with correlated noise; and eventually a two-trait Bayes factor with a residual correlation (`inherited_effects_joint_fine_mapping_model.md`, Section 5.1).
3. **Several co-accessible peaks with the same caQTL variant.** The product in Section 5 multiplies their weights as if they were independent evidence, and pairwise colocalization cannot say which peak carries the effect. Options: cluster peaks that share a caQTL credible set and count each cluster once; or keep the largest single factor.
4. **Links are not mediation.** A shared causal variant with concordant direction is consistent with mediation and with a variant that affects both traits separately.
5. **Empirical Bayes on the sharing prior** uses the same data twice. With a handful of parameters and thousands of pairs the over-fitting is negligible, but the estimate inherits issue 2.

## 10. Controls and validation

1. Colocalization rate and enrichment for linked peaks against distance-matched unlinked peaks of the same genes. This says what links add beyond proximity.
2. Held-out sign concordance among colocalized pairs, by posterior probability of $H_4$ and by link strength.
3. Credible-set size with and without the chromatin prior, and enrichment of the variants that gain probability in functional readouts that do not come from open chromatin.
4. Agreement between the single-variant version and the SuSiE version at genes with several eQTL signals.
5. Simulation from real marginal statistics with a known sharing rate, with and without correlated noise, to check the calibration of $PP_{H_4}$ and the recovery of $e$.
