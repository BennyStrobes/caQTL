# Model L as fitted by `run_ca_qtl_predicted_mean_eqtl_finemapping.py`: the probability distributions

Temporary note. Notation per gene $g$ with $p$ variants, LD matrix $R$, peaks $k \in \mathcal{K}_g$ linked to $g$.

## Data

$$z^c_k \mid b_k \sim N(R\,b_k,\ R) \qquad \text{caQTL z-scores of peak } k$$

$$\hat g_{kg} \mid g_{kg} \sim N(g_{kg},\ se_{kg}^2) \qquad \text{hurdle link estimate}$$

$$z^e_g \mid \beta_g \sim N(R\,\beta_g,\ R) \qquad \text{eQTL z-scores}$$

## Latent effects, each a spike-and-slab

**Variant-to-peak**, for every peak $k$ and variant $j$:

$$\delta_{kj} \sim \text{Bernoulli}(\pi_c), \qquad b_{kj} \mid \delta_{kj}=1 \sim N(0,\tau_c^2), \qquad b_{kj}=0 \text{ otherwise.}$$

**Peak-to-gene**, for every pair $(k, g)$:

$$u_{kg} \sim \text{Bernoulli}(\pi_g), \qquad g_{kg} \mid u_{kg}=1 \sim N(0,\tau_g^2), \qquad g_{kg}=0 \text{ otherwise.}$$

**Causal eQTL effect**, for every variant $j$ of gene $g$:

$$\gamma_{gj} \sim \text{Bernoulli}(\pi_e), \qquad \beta_{gj} \mid \gamma_{gj}=1,\ b,\ g \sim N\!\Big(\sum_b \lambda_b\, m_{gj}^{(b)},\ \sigma^2_{gj}\Big), \qquad \beta_{gj}=0 \text{ otherwise,}$$

$$m_{gj}^{(b)} = \sum_{k \in \mathcal{K}_g,\ k \in \text{bin } b} b_{kj}\, g_{kg}, \qquad \sigma^2_{gj} = \begin{cases} \sigma_0^2 & \text{if } m_{gj}=0 \ \text{(no active link)} \\ \sigma_1^2 & \text{otherwise.} \end{cases}$$

The bins $b$ are the hurdle $|t|$ bins (`--t_bin_edges`), one $\lambda_b$ each.

## Hyperparameters and priors

$$\lambda_b \sim N(0,\ 10^2) \ \text{independently (}\texttt{--lambda\_prior\_sd}\text{)}, \qquad \sigma_0^2,\ \sigma_1^2 \ge \text{floor (flat)}, \qquad \pi_g,\ \tau_g^2 \ \text{by empirical Bayes over all links.}$$

$\pi_c$ and $\tau_c^2$ are handled inside the caQTL fits ($\tau_c^2$ per single effect, $\pi_c$ implicit in the cap `--L_caqtl`); $\pi_e$ is implicit in `--L_eqtl` with uniform weights over variants.

## What the code computes

**1. Link posterior**, closed form:

$$P(u_{kg}=1 \mid \hat g) = \frac{\pi_g\, BF_{kg}}{1-\pi_g+\pi_g BF_{kg}}, \qquad BF_{kg} = \frac{N(\hat g_{kg};\ 0,\ \tau_g^2+se_{kg}^2)}{N(\hat g_{kg};\ 0,\ se_{kg}^2)},$$

$$g_{kg} \mid u_{kg}=1,\ \hat g \sim N\!\Big(\frac{\hat g_{kg}\,\tau_g^2}{\tau_g^2+se_{kg}^2},\ \frac{\tau_g^2\, se_{kg}^2}{\tau_g^2+se_{kg}^2}\Big).$$

**2. caQTL posterior per peak**, $p(b_k \mid z^c_k)$ under the spike-and-slab above, computed with SuSiE (sum of single effects; exact for `--L_caqtl 1`). Gives $\text{PIP}^c_{kj} = P(\delta_{kj}=1 \mid z^c_k)$ and the conditional mean and variance of $b_{kj}$ given $\delta_{kj}=1$.

**3. eQTL posterior**, marginalizing 1 and 2 (exact by conditional independence of the three data types given $b, g$):

$$p(\gamma_g,\beta_g \mid z^e, z^c, \hat g) \;\propto\; p(z^e_g \mid \beta_g)\, p(\gamma_g) \int p(\beta_g \mid \gamma_g, b, g)\, p(b \mid z^c)\, p(g \mid \hat g)\, db\, dg.$$

The integral turns each variant's slab into a mixture over subsets $S$ of its links being active:

$$\beta_{gj} \mid \gamma_{gj}=1 \;\sim\; \sum_{S} w_{jS}\; N\!\Big(\sum_b \lambda_b\, \bar m^{(b)}_{jS},\ \ \sigma^2_{S} + \sum_b \lambda_b^2\, v^{(b)}_{jS}\Big), \qquad w_{jS} = \prod_{k\in S} q_{kj} \prod_{k \notin S}(1-q_{kj}),$$

with $q_{kj} = \text{PIP}^c_{kj}\; P(u_{kg}=1 \mid \hat g)$, $\bar m^{(b)}_{jS}$ and $v^{(b)}_{jS}$ the posterior mean and variance of $\sum_{k \in S,\ k \in b} b_{kj} g_{kg}$ from steps 1 and 2, and $\sigma^2_S = \sigma_0^2$ for the empty subset, $\sigma_1^2$ otherwise. Subsets are over the peaks with $q_{kj} \ge$ `--min_caqtl_pip`, at most `--max_peaks_per_variant` of them. The posterior over $\gamma_g$ (which variants are causal, allowing several in LD) is computed with SuSiE's sum of single effects; the mixture slab inside each single effect is handled exactly.

**4. Shared hyperparameters** $\lambda_b,\ \sigma_0^2,\ \sigma_1^2$ by EM across genes, from the posterior moments of the causal effects (weighted least squares of posterior effects on their predictions for $\lambda$; pooled second moments for the variances), repeated over rounds of gene fits until they stop moving.

## Outputs per variant

$\text{PIP}_{gj} = P(\gamma_{gj}=1 \mid \text{all data})$; its split into "causal via the no-link component" (`direct_pip`) and "causal via a chromatin prediction" (`mediated_pip`); the posterior mean of $\beta_{gj}$; the prior expected prediction $\sum_S w_{jS} \sum_b \lambda_b \bar m^{(b)}_{jS}$; and $1 - w_{j\emptyset}$, the prior probability that the variant has a prediction at all.
