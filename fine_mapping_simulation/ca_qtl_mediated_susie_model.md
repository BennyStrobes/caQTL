# caQTL-mediated eQTL fine-mapping: the probabilistic model

This documents the generative model fit by `CAQTL_MEDIATED_SUSIE` in `ca_qtl_mediated_susie.py`,
and how the eQTL PIPs are computed from its posterior. Inference updates are not described here.

## Notation (one gene)

- $p$ cis SNPs. $X_A$ ($n_A \times p$) and $X_E$ ($n_E \times p$) are the standardized genotype matrices of
  the caQTL and eQTL samples (each column mean 0, variance 1). They share the LD matrix $R$, so
  $X_A^\top X_A = (n_A - 1) R$ and $X_E^\top X_E = (n_E - 1) R$.
- $K$ peaks assigned to the gene. $A_k$ is the accessibility of peak $k$ ($n_A$-vector), $E$ the gene's
  expression ($n_E$-vector), both centered.
- $\hat g_k$ and $s_k$ are the observed peak-gene link estimate and its standard error for peak $k$.

## Generative model

**caQTL row, one per peak**

$$A_k = X_A\, b_k + \epsilon_k, \qquad \epsilon_k \sim N(0,\ \sigma_k^2 I), \qquad k = 1,\dots,K$$

**eQTL row**

$$E = X_E \Big( \sum_{k=1}^{K} g_k\, b_k + d \Big) + \epsilon, \qquad \epsilon \sim N(0,\ \sigma^2 I)$$

so the causal eQTL effect vector is $\beta_E = \sum_k g_k b_k + d$: the caQTL effects of each linked peak,
scaled by that peak's effect on the gene, plus direct effects.

**Link row, one per peak**

$$\hat g_k \mid g_k \sim N\big(\lambda\, g_k,\ s_k^2 + \tau_u^2\big)$$

$\lambda$ is a global scale between the observational link and the genetic link, and $\tau_u^2$ is the
variance of a per-link observational bias that has been integrated out. Defaults are $\lambda = 1$ and
$\tau_u^2 = 0$, so the observed link is an unbiased noisy measurement of $g_k$.

## Priors

**caQTL effects: SuSiE with $L_A$ single effects per peak**

$$b_k = \sum_{l=1}^{L_A} b_{kl}, \qquad b_{kl} = \gamma_{kl}\, \beta_{kl}, \qquad
\gamma_{kl} \sim \text{Multinomial}(1, \pi_{\text{snp}}), \qquad \beta_{kl} \sim N(0,\ V^A_{kl})$$

**Direct eQTL effects: SuSiE with $L_E$ single effects**

$$d = \sum_{m=1}^{L_E} d_m, \qquad d_m = \gamma_m\, \beta_m, \qquad
\gamma_m \sim \text{Multinomial}(1, \pi_{\text{snp}}), \qquad \beta_m \sim N(0,\ V^E_m)$$

$\gamma$ is a one-hot vector choosing the SNP that carries the effect; $\pi_{\text{snp}}$ is uniform,
$1/p$, unless prior weights are supplied.

**Peak-gene effects: spike and slab**

$$g_k = z_k\, h_k, \qquad z_k \sim \text{Bernoulli}(\pi), \qquad h_k \sim N(0,\ \tau_g^2)$$

**What is shared and what is gene-specific**

- Shared across all genes and estimated: $\pi$, $\tau_g^2$, and optionally $\lambda$, $\tau_u^2$.
- Gene-specific and estimated: the single-effect prior variances $V^A_{kl}$, $V^E_m$ (as in SuSiE, an
  estimate of exactly zero switches that effect off) and the residual variances $\sigma_k^2$, $\sigma^2$.
- Defaults: $L_E = 10$, $L_A = 5$.

## Summary-statistic form

The two Gaussian rows depend on the data only through sufficient statistics. For a trait $y$ with
marginal OLS effects $\hat\beta$ and standard errors $\hat s$ on standardized genotypes,

$$X^\top X = (n-1) R, \qquad X^\top y = (n-1)\hat\beta, \qquad
y^\top y = (n-1)\big(\hat s_j^2 (n-2) + \hat\beta_j^2\big) \ \text{for any } j,$$

and $\log p(y \mid b) = -\tfrac{n}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^2}\big(y^\top y - 2 b^\top X^\top y + b^\top X^\top X\, b\big)$.
With in-sample LD this is exact, so the fit is identical to one on individual-level data.

## Variational posterior

Inference is mean-field. The posterior factorizes over every single effect and every link:

$$q = \prod_{k}\prod_{l} q(\gamma_{kl}, \beta_{kl}) \ \prod_{m} q(\gamma_m, \beta_m) \ \prod_k q(z_k, h_k)$$

with

$$q(\gamma_{kl} = e_j) = \alpha_{klj}, \quad q(\beta_{kl} \mid \gamma_{kl} = e_j) = N(\mu_{klj}, s^2_{klj}); \qquad
q(\gamma_m = e_j) = \alpha_{mj}, \quad q(\beta_m \mid \gamma_m = e_j) = N(\mu_{mj}, s^2_{mj});$$

$$q(z_k = 1) = q_k, \qquad q(h_k \mid z_k = 1) = N(m_k, v_k).$$

## eQTL PIPs

SNP $j$ has a non-zero effect on expression, $\beta_{E,j} \neq 0$, if it carries a direct single effect, or
if some peak $k$ has a non-zero link ($z_k = 1$) and SNP $j$ carries one of that peak's caQTL single effects.
Under the factorized posterior these events are independent, giving:

**caQTL PIP of SNP $j$ for peak $k$** (over single effects whose estimated prior variance is non-zero)

$$\text{PIP}^A_{kj} = 1 - \prod_{l:\, V^A_{kl} > 0} (1 - \alpha_{klj})$$

**Direct PIP**

$$\text{PIP}^D_{j} = 1 - \prod_{m:\, V^E_{m} > 0} (1 - \alpha_{mj})$$

**Mediated PIP**: probability that at least one linked peak carries an effect at $j$

$$\text{PIP}^M_{j} = 1 - \prod_{k=1}^{K} \big(1 - q_k\, \text{PIP}^A_{kj}\big)$$

**eQTL PIP**: probability of a direct or a mediated effect at $j$

$$\text{PIP}^E_{j} = 1 - \big(1 - \text{PIP}^D_{j}\big)\big(1 - \text{PIP}^M_{j}\big)
= 1 - \prod_{m}(1 - \alpha_{mj}) \prod_{k}\Big[1 - q_k \Big(1 - \prod_{l}(1 - \alpha_{klj})\Big)\Big]$$

The posterior mean eQTL effect reported alongside is

$$\mathbb{E}_q[\beta_{E,j}] = \sum_k q_k\, m_k \sum_l \alpha_{klj}\, \mu_{klj} + \sum_m \alpha_{mj}\, \mu_{mj}.$$
