# Joint fine-mapping of eQTLs and caQTLs with inherited effects: proposed model

Ben Strober, 2026-09-21

## 1. Summary

This is a proposal, not yet implemented. It is a single generative model for a gene's eQTLs and the caQTLs of its linked peaks.

- Each peak's caQTL effects are SuSiE single effects.
- Each peak-gene link has a spike and slab.
- An eQTL effect is one of two kinds. An **inherited** effect is the expression arm of a caQTL single effect: it sits at the same variant, and its size is (scale) $\times$ (link) $\times$ (caQTL effect) plus noise. A **direct** effect is an ordinary SuSiE single effect.

So the prior probability of an eQTL effect at a variant, and its prior mean, are both determined by the caQTL effects and the link, inside the model and not through a plug-in annotation. Because an inherited effect shares its location with its caQTL effect, the two are fine-mapped together: the posterior over the variant multiplies the caQTL evidence by the eQTL evidence at the same variant.

Every prior is centred at zero (the tie between the two sizes lives in a $2 \times 2$ prior covariance), so standard SuSiE pruning by $V \to 0$ is safe and no explicit null outcome is needed. With the inheritance probability set to zero the model is separate SuSiE fits of every trait.

## 2. Data and scale

For gene $g$ with $p$ cis variants and LD matrix $R$ (the same for all traits):

- eQTL marginal effects $\hat\beta$ with standard errors $s^E$, and $z^E = \hat\beta / s^E$;
- for each linked peak $k = 1, \dots, K$: caQTL marginal effects $\hat b_k$ with standard errors $s^{A_k}$, and $z^{A_k} = \hat b_k / s^{A_k}$;
- for each peak: the estimated link $\hat g_k$ with standard error $se_k$.

Genotypes are standardized. eQTL effects are per standard deviation of genotype in raw expression units, and caQTL effects in the units of the caQTL statistics. Each trait has the summary-statistic likelihood used elsewhere in this repo, $\hat\beta \mid \beta \sim N(S R S^{-1}\beta,\ S R S)$, and the same for each peak.

## 3. Model

**caQTL effects.** For peak $k$ and $l = 1, \dots, L_2$:

$$\delta_{kl} \sim \text{Uniform}\{1, \dots, p\}, \qquad \phi_{kl} \sim N(0, V_{kl}), \qquad b_{kj} = \sum_l \phi_{kl}\,1[\delta_{kl} = j].$$

**Links.** $h_k = u_k g_k$ with

$$u_k \sim \text{Bernoulli}(\pi_g), \qquad g_k \sim N(0, \tau^2), \qquad \hat g_k \mid h_k \sim N(h_k, se_k^2).$$

**Inherited eQTL effects.** Each caQTL effect is passed on to expression with probability $\psi$:

$$t_{kl} \sim \text{Bernoulli}(\psi), \qquad \theta_{kl} \mid \phi_{kl}, h_k, t_{kl} = 1 \;\sim\; N\big(\lambda\, h_k\, \phi_{kl},\; \sigma_\gamma^2\, V_{kl}\big), \qquad \theta_{kl} = 0 \text{ if } t_{kl} = 0.$$

The inherited effect sits at the same variant $\delta_{kl}$. Its noise variance is proportional to $V_{kl}$, so the pair $(\phi_{kl}, \theta_{kl})$ is jointly normal with covariance

$$V_{kl}\,\Omega_k, \qquad \Omega_k = \begin{pmatrix} 1 & \lambda h_k \\ \lambda h_k & \lambda^2 h_k^2 + \sigma_\gamma^2 \end{pmatrix},$$

and $V_{kl} \to 0$ switches off both arms together.

**Direct eQTL effects.** For $l = 1, \dots, L$: $\gamma_l \sim \text{Uniform}\{1, \dots, p\}$ and $\theta^d_l \sim N(0, V^d_l)$.

**Total eQTL effect.**

$$\beta_j = \sum_{k,l} \theta_{kl}\,1[\delta_{kl} = j] + \sum_l \theta^d_l\,1[\gamma_l = j].$$

**Shared across genes and estimated:** $\psi$, $\lambda$, $\sigma_\gamma^2$, $\pi_g$, $\tau^2$. The prior variances $V_{kl}$ and $V^d_l$ are estimated for each effect, as in standard SuSiE.

## 4. What the model says

- **Prior probability of an eQTL at a variant.** It is raised exactly where a linked peak has a caQTL effect, by $\psi$. That is the enrichment of eQTLs at caQTL variants, and it is part of the generative story.
- **Prior mean of that eQTL effect.** Given the caQTL effect and the link, it is $\lambda h_k \phi_{kl}$: the chromatin prediction. Agreement in direction and size between (link $\times$ caQTL effect) and the eQTL effect appears as $\lambda > 0$ and a small $\sigma_\gamma$.
- **Escape hatches.** A caQTL effect need not be inherited ($t_{kl} = 0$). A link can be off ($u_k = 0$) however significant its co-accessibility; an inherited effect then has mean zero and relative spread $\sigma_\gamma$, so a sizable inherited effect is evidence that the link is on. An eQTL at a caQTL variant whose direction disagrees with the prediction is explained by a direct effect at that variant, and earns no credit from the chromatin data.
- **Scale.** $\sigma_\gamma$ is dimensionless (it is relative to the caQTL effect), and the gene- and peak-specific scale of each signal is carried by $V_{kl}$. Only $\lambda$ carries units.

## 5. Inference

Variational, in the SuSiE style (one effect updated at a time against the residual left by the others):

$$q = \prod_{k,l} q(\delta_{kl}, t_{kl}, \phi_{kl}, \theta_{kl}) \;\prod_l q(\gamma_l, \theta^d_l)\; \prod_k q(u_k, g_k).$$

Location, inheritance and both sizes of a caQTL effect are kept together in one factor. Nothing is factorized between an inherited eQTL effect and its caQTL effect.

### 5.1 Two-trait single-effect update for caQTL effect $(k, l)$

Remove every other effect from both traits:

$$\tilde z^{A}_j = z^{A_k}_j - \sum_i R_{ji}\frac{\bar b^{(-l)}_{ki}}{s^{A_k}_i}, \qquad \tilde z^{E}_j = z^{E}_j - \sum_i R_{ji}\frac{\bar\beta^{(-kl)}_i}{s^{E}_i},$$

where $\bar b^{(-l)}_k$ is the posterior mean caQTL effect of peak $k$ without effect $l$, and $\bar\beta^{(-kl)}$ is the posterior mean eQTL effect (all direct effects and all inherited arms) without the arm of $(k, l)$.

At variant $j$ let $x_j = (\tilde z^A_j, \tilde z^E_j)'$, $D_j = \text{diag}(s^{A_k}_j, s^E_j)$ and $\Sigma_k = \begin{pmatrix}1 & r_k\\ r_k & 1\end{pmatrix}$, where $r_k$ is the correlation of the two traits' z-scores under the null (zero to start with; see issue 2). In z-score units the two outcomes have prior covariances

$$U^0_j = V_{kl}\,D_j^{-1}\begin{pmatrix}1 & 0\\0 & 0\end{pmatrix}D_j^{-1}, \qquad U^1_j = V_{kl}\,D_j^{-1}\,\bar\Omega_k\,D_j^{-1}, \qquad \bar\Omega_k = \begin{pmatrix}1 & \lambda\bar h_k\\ \lambda\bar h_k & \lambda^2 E[h_k^2] + \sigma_\gamma^2\end{pmatrix},$$

with $\bar h_k = a_k\nu_k$ and $E[h_k^2] = a_k(\nu_k^2 + \tau_k^2)$ from the link posterior. Using $E[h_k^2]$ and not $\bar h_k^2$ widens the inherited arm when the link is uncertain, and keeps $\bar\Omega_k$ positive semi-definite.

For $t \in \{0, 1\}$:

$$\log BF^t_j = \frac12\Big[\log|\Sigma_k| - \log|\Sigma_k + U^t_j|\Big] + \frac12\,x_j'\Big[\Sigma_k^{-1} - (\Sigma_k + U^t_j)^{-1}\Big]x_j,$$

$$\alpha^0_{klj} \propto \frac{1 - \psi}{p}\,e^{\log BF^0_j}, \qquad \alpha^1_{klj} \propto \frac{\psi}{p}\,e^{\log BF^1_j}, \qquad \sum_j\big(\alpha^0_{klj} + \alpha^1_{klj}\big) = 1.$$

Given variant $j$ and outcome $t$, the two sizes in z-score units are normal with

$$\text{mean } U^t_j(\Sigma_k + U^t_j)^{-1}x_j, \qquad \text{covariance } U^t_j - U^t_j(\Sigma_k + U^t_j)^{-1}U^t_j,$$

and multiplying by $D_j$ returns them to the original units. These give $E[\phi]$, $E[\theta]$, $E[\phi^2]$, $E[\theta^2]$ and $E[\phi\theta]$ for every $(j, t)$. With $r_k = 0$ and $t = 0$ this is the ordinary SuSiE update of the caQTL effect.

$V_{kl}$ maximizes $\log\sum_j\big[(1 - \psi)BF^0_j(V) + \psi BF^1_j(V)\big]$ over $V$ (one-dimensional, as in `SUSIE_RSS`), and is set to zero when no effect is at least as likely.

The posterior over the variant is proportional to the caQTL evidence times, for the inherited outcome, the eQTL evidence at the same variant. This is where joint fine-mapping happens.

### 5.2 Direct eQTL effect $l$

The ordinary SuSiE single-effect update on the expression residual, which excludes all inherited arms and the other direct effects.

### 5.3 Link $k$

The inherited arms of peak $k$ act as a regression of $\theta$ on $\phi$ with coefficient $\lambda h_k$. They add a precision and a linear term to the link's posterior:

$$P_k = \lambda^2\sum_l \frac{1}{\sigma_\gamma^2 V_{kl}}\sum_j \alpha^1_{klj}\,E[\phi^2 \mid j, 1], \qquad Q_k = \lambda\sum_l \frac{1}{\sigma_\gamma^2 V_{kl}}\sum_j \alpha^1_{klj}\,E[\phi\,\theta \mid j, 1],$$

$$\frac{1}{\tau_k^2} = \frac{1}{se_k^2} + \frac{1}{\tau^2} + P_k, \qquad \nu_k = \tau_k^2\Big(\frac{\hat g_k}{se_k^2} + Q_k\Big), \qquad \text{logit}(a_k) = \text{logit}(\pi_g) + \frac12\log\frac{\tau_k^2}{\tau^2} + \frac{\nu_k^2}{2\tau_k^2}.$$

Sums run over effects with $V_{kl} > 0$. With nothing inherited, $P_k = Q_k = 0$ and the link posterior is the one implied by $\hat g_k$ alone.

### 5.4 Shared parameters

Pooled over genes, over caQTL effects with $V_{kl} > 0$, with $w_{klj} = \alpha^1_{klj} / V_{kl}$:

$$\psi = \frac{\sum_{k,l}\sum_j\alpha^1_{klj}}{\#\{(k,l): V_{kl} > 0\}}, \qquad \lambda = \frac{\sum w_{klj}\,\bar h_k\,E[\phi\theta \mid j,1]}{\sum w_{klj}\,E[h_k^2]\,E[\phi^2 \mid j,1]},$$

$$\sigma_\gamma^2 = \frac{\sum w_{klj}\Big(E[\theta^2 \mid j,1] - 2\lambda\bar h_k E[\phi\theta \mid j,1] + \lambda^2E[h_k^2]E[\phi^2 \mid j,1]\Big)}{\sum_{k,l,j}\alpha^1_{klj}},$$

$$\pi_g = \frac{\sum_k a_k}{\#\text{links}}, \qquad \tau^2 = \frac{\sum_k a_k(\nu_k^2 + \tau_k^2)}{\sum_k a_k}.$$

Put a floor on $\sigma_\gamma^2$ and $\tau^2$.

### 5.5 Algorithm

1. Fit every peak's caQTL and the gene's eQTL separately with SuSiE, and the links from $\hat g_k$ alone.
2. For each gene, sweep: every caQTL effect (5.1), every direct effect (5.2), every link (5.3), until the gene's ELBO settles.
3. Update the shared parameters (5.4). Repeat from step 2 until they change by less than a tolerance.

**Two starts for each gene.** A colocalized eQTL signal can be explained as inherited or as direct, and whichever claims it first leaves the other an empty residual. Run step 2 from (a) direct effects empty, so that inherited arms see the full eQTL signal first, and (b) direct effects from the eQTL-only fit. Keep the fit with the higher ELBO. Model A needed the same device.

The single-effect update uses a moment-matched normal prior for the inherited arm, and the link update uses the expected log prior. The objective being monitored is therefore approximate.

## 6. Outputs

- eQTL PIP of variant $j$: $1 - \prod_l\big(1 - \alpha^d_{lj}\big)\prod_{k,l}\big(1 - \alpha^1_{klj}\big)$, and credible sets for each effect.
- caQTL PIPs and credible sets for each peak, sharpened by the eQTL data where an effect is inherited.
- For each variant, peak and gene: the probability that the variant is a caQTL for the peak and that the effect is inherited by the gene, $1 - \prod_l(1 - \alpha^1_{klj})$, with the link posterior $a_k$ alongside.
- $\psi$, $\lambda$, $\sigma_\gamma$, $\pi_g$. Compare $\psi$ with the rate of direct effects per variant to express the enrichment of eQTLs at caQTL variants of linked peaks.

## 7. Special cases

- $\psi = 0$: separate SuSiE fits of the eQTL and of each caQTL. This is the first implementation check.
- $\lambda = 0$: colocalization with a learned prior probability $\psi$, shared locations and unrelated sizes. The link plays no part.
- $\sigma_\gamma \to 0$ with the link on: every inherited effect is exactly $\lambda h_k$ times its caQTL effect, which is the mediated part of Model A.

## 8. Relation to the other models in this repo

- **Model A (`ca_qtl_mediated_susie.py`).** The same additive structure, with four changes: noise on the inherited size, an inheritance probability for each caQTL effect, one shared location for a caQTL effect and its inherited arm, and room for a residual correlation between the traits.
- **`joint_eqtl_caqtl_susie_model.md`.** There an eQTL effect and a caQTL effect are separate objects that must land on the same variant, helped by a prior computed from the caQTL fit. That model is what results from fixing the locations from the caQTL data alone and passing them forward.
- **Element models E, I, J.** They test a peak's whole caQTL profile against the eQTL signal with one coefficient. Here each caQTL effect is inherited separately, but all the effects of a peak share $h_k$, so a peak with several caQTL signals still has to reproduce the eQTL signal with a common ratio, up to $\sigma_\gamma$.

## 9. Known issues and open decisions

1. **Attribution can get stuck** between inherited and direct explanations of the same signal. The two starts address it. A final pass that tries moving each direct effect that overlaps a linked caQTL credible set into the inherited arm is a further option.
2. **Shared donors.** Expression and accessibility come from the same people, so the two traits' z-scores are correlated under the null, with the sign of the link. $r_k$ handles this inside the likelihood. It can be estimated from variants far from any signal, or from the donor-level correlation of the two phenotypes. Start with $r_k = 0$ and measure how much the results change. Direct effects use expression alone, which ignores $r_k$ for them.
3. **The size tie.** $\lambda h_k$ assumes that link size predicts the ratio of eQTL to caQTL effect across genes. Under the old standardization it did not. Recent results under the current standardization suggest it does better. If it fails, $\sigma_\gamma$ grows and the model tends to the colocalization special case. A sign-only version would replace the normal tie by a preference for the predicted direction.
4. **A variant that is a caQTL for several co-accessible peaks** gives several caQTL effects at one variant, each of which may be inherited. They compete through the expression residual, so the inherited eQTL effect is not counted twice, but which peak gets the credit is weakly determined.
5. **Peaks shared between genes.** Each gene currently gets its own copy of a peak's caQTL statistics, so different genes can pull the same peak's caQTL fit in different directions. Start with per-gene copies.
6. **Links are not mediation.** Inheritance is a statement about shared causal variants with proportional effects. It does not distinguish mediation from a variant that affects both traits separately, except through the common ratio across a peak's several signals.
7. **Cost.** $K L_2$ two-trait updates per sweep, each a $2 \times 2$ computation per variant. Screen out peaks with no caQTL signal first.

## 10. Validation plan

1. With $\psi = 0$, reproduce separate `SUSIE_RSS` fits exactly.
2. Simulate from the model with real LD: known $\psi$, $\lambda$, $\sigma_\gamma$ and link states, with inherited and direct eQTLs, including $\psi = 0$. Check PIP calibration for both traits, credible-set size against separate fits, recovery of the shared parameters, and how often the two starts disagree.
3. Simulate correlated noise between the traits and fit with $r_k = 0$ and with the true $r_k$, to measure the bias from shared donors and how well $r_k$ removes it.
4. Real data against eQTL-only SuSiE: credible-set sizes and numbers of high-PIP variants in genes with an inherited effect, enrichment in functional readouts that do not come from open chromatin, replication in independently fine-mapped eQTLs, and agreement of $a_k$ with external enhancer-gene benchmarks.
