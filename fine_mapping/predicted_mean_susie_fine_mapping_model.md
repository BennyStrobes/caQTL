# Chromatin-informed eQTL fine-mapping: the predicted-mean model

Ben Strober, 2026-09-21

## 1. The model in one paragraph

Fine-map each gene's eQTLs with SuSiE. The prior on a causal eQTL effect is a spike at zero plus a slab, and the slab depends on what chromatin says about the variant. If the variant has no chromatin path to the gene, the slab is the ordinary one, centred at zero. If the variant is a caQTL variant for a peak that is linked to the gene, the slab is centred on the effect predicted through chromatin, (scale) $\times$ (peak-gene link) $\times$ (caQTL effect), and is tight. The caQTL effects and the links are estimated first, from their own data, and their uncertainty is carried into the slab. Three numbers are learned across genes: the scale, and the two slab widths.

This is Model L (agreed on 2026-09-16, implemented in `fine_mapping_old/ca_qtl_predicted_mean_susie.py` and `susie_rss_mixture.py`), restated with what was learned since. Section 7 lists the differences.

## 2. Inputs for one gene

Over the gene's $p$ cis variants, with LD matrix $R$:

- eQTL z-scores $z^E$;
- for each linked peak $k$: caQTL z-scores $z^{A_k}$;
- for each peak: the estimated link $\hat g_k$ with standard error $se_k$.

Everything is in z-score units, so that effect sizes are comparable across genes. The scale $\lambda$ converts (caQTL z units $\times$ link units) into eQTL z units.

## 3. Step 1: chromatin, from its own data

**caQTL effects.** SuSiE on each peak's caQTL z-scores. For peak $k$ and variant $j$ this gives the PIP, $PIP^c_{kj}$, and the posterior mean and variance of the effect $b_{kj}$ given that it is non-zero.

**Links.** A spike and slab on each link, $g_k \sim \pi_g N(0, \tau^2) + (1 - \pi_g)\delta_0$ with $\hat g_k \sim N(g_k, se_k^2)$, and $(\pi_g, \tau^2)$ learned across all links. This gives $a_k = P(g_k \ne 0)$ and the posterior mean and variance of $g_k$ given that it is non-zero.

**Paths.** Variant $j$ has a path through peak $k$ when it is a caQTL variant for $k$ and link $k$ is on. That has probability $r_{kj} = a_k\,PIP^c_{kj}$, and given the path the predicted effect $g_k b_{kj}$ has a mean and a variance that follow from the two posteriors.

## 4. Step 2: eQTL fine-mapping with the predicted-mean prior

$\beta = \sum_{l=1}^{L}\theta_l\mathbf{1}_{\gamma_l}$, as in SuSiE. Each single effect is absent with probability $1 - \rho_e$, and otherwise sits at variant $j$ with probability $1/p$. Given that it sits at variant $j$, its size has a mixture prior over which of the variant's paths are active. For a set $A$ of active paths,

$$w_{jA} = \prod_{k \in A} r_{kj}\prod_{k \notin A}(1 - r_{kj}),$$

$$\theta_l \mid \gamma_l = j, A \;\sim\; \begin{cases} N\big(0,\; \sigma_0^2\big) & A = \varnothing,\\[4pt] N\big(\lambda\,m_{jA},\; \sigma_1^2 + \lambda^2 v_{jA}\big) & A \ne \varnothing,\end{cases}$$

where $m_{jA}$ and $v_{jA}$ are the mean and variance of $\sum_{k \in A} g_k b_{kj}$ from step 1. In practice a variant has zero or one path of any weight, so this is: no path, the ordinary slab; a path, a slab centred on the prediction.

**Shared across genes and learned:** $\lambda$ (with a weak prior $N(0, 10^2)$), $\sigma_0^2$, $\sigma_1^2$, and $\rho_e$.

**The single-effect update** is a softmax over "absent" and every (variant, active set) pair. For a slab with mean $M$ and variance $T$, and with $\tilde z_j$ the z-score of variant $j$ after removing the other effects,

$$V = \Big(1 + \frac{1}{T}\Big)^{-1}, \qquad \hat\theta = V\Big(\tilde z_j + \frac{M}{T}\Big), \qquad \log BF = \frac12\log\frac{V}{T} + \frac{\hat\theta^2}{2V} - \frac{M^2}{2T},$$

and the posterior weight of each outcome is its prior weight times its Bayes factor. Given the other effects this is exact: nothing is factorized between where an effect is, which path it follows and how big it is. PIPs and credible sets follow as usual.

**Learning.** Alternate: fit every gene; update the shared parameters in closed form or by a small numerical search ($\lambda$ and $\sigma_1^2$ together, because $\lambda$ appears in the slab variance), as in `chromatin_informed_expression_prediction.py`. Start from $\lambda = 0$ and $\sigma_1^2 = \sigma_0^2$, which is SuSiE on the eQTL data alone.

## 5. Why each piece is there

- **The prior mean carries direction and size.** A variant whose eQTL effect looks like what chromatin predicts, in sign and size, is favoured over an LD partner with no prediction. Nothing else is rewarded: there is no separate enrichment of caQTL variants, no sign gate and no second use of the eQTL data.
- **Two slab widths.** $\sigma_0^2$ is the ordinary slab and has to exist, because most eQTLs have no chromatin path. $\sigma_1^2$ is how far real effects sit from their prediction. The leverage of the prior comes from $\sigma_1$ being smaller than $\sigma_0$: for an effect that matches its prediction, the gain over an LD partner without one is roughly $(\sigma_0/\sigma_1)\times 1.6$. With one shared width the gain is at most 1.6, which is why the first joint prediction model did nothing.
- **$\sigma_1^2$ cannot be dropped.** One global $\lambda$ cannot match every gene's and peak's scale, mediation is often partial, and the links measure co-accessibility. Without $\sigma_1^2$, a confident path would give a near point-mass prior, any mismatch would drive that variant's Bayes factor to zero and push the signal onto an LD partner, and the only way to absorb mismatch across genes would be $\lambda \to 0$.
- **The mixture with the empty set.** A path variant can still carry an effect that ignores the prediction, at the cost of the factor $\prod_k(1 - r_{kj})$. So a wrong prediction costs the variant a little and never pulls a signal toward it. The model fails conservatively.
- **No feedback.** The caQTL and link fits are not updated by the eQTL data. In every simulation this month feedback moved them very little, and in Models A, D and E it was the source of instability.
- **An explicit "absent" outcome and shared slab widths.** With a non-zero prior mean, switching an effect off by shrinking its own variance to zero does not mean "no effect" (the Model K problem).

## 6. What to read off a fit

1. $\sigma_1^2$ against $\sigma_0^2$. This is the main diagnostic. If they are close, the predictions do not line up with eQTL effects and the fit is essentially standard SuSiE. In the stage 2 prediction run the path slab was about ten times tighter than the null slab, in raw units.
2. $\lambda$, and whether it is clearly positive.
3. Against standard SuSiE, in genes that have a path: credible-set sizes and the number of variants with PIP above 0.9 (`compare_susie_finemapping_results.py`).
4. For each variant, the posterior probability that its effect follows a path, and which peak.

## 7. Differences from the Model L code

1. An explicit "absent" outcome with probability $1 - \rho_e$ for each single effect, in place of the rule that switches effects on and off.
2. One $\lambda$ by default. The code allows a $\lambda$ for each bin of link significance; keep that as an option.
3. Links with a zero or non-finite standard error are dropped (the code already does this).
4. Optional, off by default: a path-dependent location prior, $P(\gamma_l = j) \propto 1 + (e - 1)\rho_j$ with $\rho_j = 1 - \prod_k(1 - r_{kj})$. It adds a lot of power (the enrichment was about 20 to 30 in two separate fits), but it is the less defensible ingredient: part of it is that caQTLs and eQTLs both sit in regulatory DNA, its estimate is self-reinforcing, and PIPs became over-confident in simulation when it was large. Keep $e = 1$ for the main analysis and report $e > 1$ as a sensitivity analysis.

## 8. Known limitations

1. **It relies on sizes lining up across genes through one $\lambda$.** Under the old standardization they did not. $\sigma_1^2$ reports how well they do now, and the model degrades to standard SuSiE when they do not.
2. **Leverage is moderate**, about five-fold at $\sigma_0/\sigma_1 \approx 3$. Among two equal LD partners that moves PIPs from 0.5 and 0.5 to about 0.83 and 0.17.
3. **Gains need a well-resolved caQTL.** If the caQTL cannot tell LD partners apart, all of them carry the prediction and none is favoured.
4. **Shared donors.** Expression and accessibility come from the same people, so noise in the two sets of z-scores is correlated, with the sign of the link. At weak caQTLs this can make predictions look better than they are. Check $\sigma_1^2$ and $\lambda$ by caQTL strength, and simulate with correlated noise.
5. **Shared slab widths.** SuSiE usually gives each effect its own width. In z-score units one width for all effects is a fair approximation and PIPs are not sensitive to it for strong signals, but weak secondary signals may lose a little power.
6. **Two effects on one path variant** would each claim the prior mean. Rare; check the fits for it.

## 9. Validation

1. With $\lambda = 0$ and $\sigma_1^2 = \sigma_0^2$, the fit must equal SuSiE with a null outcome on the eQTL data alone, and its PIPs should agree closely with `SUSIE_RSS`.
2. Simulate eQTL z-scores from the real LD matrices and the real step 1 fits, with a known $\lambda$ and $\sigma_1^2$, including the case where predictions carry no information. Check PIP calibration, credible-set coverage and size, and recovery of the shared parameters.
3. The same with correlated noise between the traits.
4. Real data against standard SuSiE, as in Section 6, plus enrichment of the variants whose PIP rises in functional readouts that do not come from open chromatin.
