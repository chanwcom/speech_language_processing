

# Maximum Likelihood (ML) and Expectation-Maximization (EM)

## Introduction
In the realm of statistical machine learning, we often face the challenge of estimating the parameters of a model that best explains our observed data. This chapter provides an in-depth technical exploration of **Maximum Likelihood Estimation (MLE)**—the gold standard for complete data—and the **Expectation-Maximization (EM) Algorithm**, the essential tool for dealing with latent variables and missing data.

---

## 1. Maximum Likelihood Estimation (MLE): The Foundation

### 1.1 Philosophical Underpinning
MLE is built on a simple yet powerful frequentist principle: **"The best parameters are those under which the observed data is most likely to have occurred."**

If we have a set of independent and identically distributed (i.i.d.) observations 
$X = \{x_0, x_1,  \cdots, x_{I-1}\}$, and a probability model $P(X|\theta)$, the likelihood function $L(\theta)$ is given by:
$$L(\theta) = P(x_0, x_1, \cdots, x_{I-1} \mid \theta) = \prod_{i=0}^{N-1} P(x_i \mid \theta).$$

### 1.2 The Necessity of the Log-Likelihood

Even though the above equation is conceptually useful, in Maximum Likelihood Estimation (MLE), 
we usually use the log likelihood defined by the following equations:
$$\ell(\theta) = \sum_{i=0}^{I-1} \log P(x_i | \theta).$$
-  **Mathematical and Computational Convenience (Addition)**
    Calculating the gradient of a sum is significantly simpler than using the product rule for a long chain of terms. 

  -  **Prevention of Numerical Underflow (Stability)**  
    Probabilities are values between $0$ and $1$, and in fields like Speech Recognition or NLP, we often deal with sequences involving thousands of observations. Multiplying thousands of small probabilities (e.g., $10^{-5} \times 10^{-7} \dots$) quickly leads to a value so small that it exceeds the precision limits of floating-point representations (e.g., `float32`). This results in **Numerical Underflow**, where the computer simply rounds the value to zero.  
    Logarithms map these tiny probabilities to a manageable range of negative numbers 
    (e.g., $10^{-100}$ becomes $-100$ in $\log_{10}$). This ensures that the gradients 
    remain stable and the model continues to learn without losing numerical precision.


Since the logarithm is a monotonically increasing function, the $\theta$ that maximizes $\ell(\theta)$ also maximizes $L(\theta)$.

### 1.3 Step-by-Step Analytical Derivation
1. **Define the Model:** Choose a distribution (e.g., Gaussian, Bernoulli).
2. **Construct the Log-Likelihood:** Sum the logs of individual densities.
3. **Compute Derivatives:** Find the score function $\nabla_{\theta} \ell(\theta)$.
4. **Solve the Score Equation:** Set $\nabla_{\theta} \ell(\theta) = 0$.
5. **Verify Concavity:** Ensure the Hessian matrix $H(\theta)$ is negative semi-definite to confirm a maximum, not a minimum.

### 1.4 Case Study: Parameter Estimation in Gaussian Models

To see MLE in action, consider a **Gaussian Model** where we aim to estimate the parameters $\theta = \{\mu, \sigma\}$ for a dataset $X = \{x_0, x_1, \dots, x_{I-1}\}$. For $I$ independent observations from $\mathcal{N}(\mu, \sigma^2)$, the log-likelihood is:

$$\ell(\mu, \sigma) = \sum_{i=0}^{I-1} \left( -\frac{1}{2} \log(2\pi) - \log(\sigma) - \frac{(x_i - \mu)^2}{2\sigma^2} \right)$$

#### A. Estimating the Mean ($\mu$)
Taking the **derivative with respect to $\mu$** and setting it to zero:
$$\frac{\partial \ell}{\partial \mu} = \sum_{i=0}^{I-1} \frac{(x_i - \mu)}{\sigma^2} = 0 \implies \sum_{i=0}^{I-1} x_i - I\mu = 0$$
Solving this yields the maximum likelihood estimator for the mean: 
$$\hat{\mu}_{MLE} = \frac{1}{I} \sum_{i=0}^{I-1} x_i$$

#### B. Estimating the Standard Deviation ($\sigma$)
Following the standard derivation, we take the **derivative with respect to $\sigma$** and set it to zero:
$$\frac{\partial \ell}{\partial \sigma} = \sum_{i=0}^{I-1} \left( -\frac{1}{\sigma} + \frac{(x_i - \mu)^2}{\sigma^3} \right) = 0$$

* **Analytical Steps:**
    1.  Rearrange the summation: $-\frac{I}{\sigma} + \frac{1}{\sigma^3} \sum_{i=0}^{I-1} (x_i - \mu)^2 = 0$
    2.  Multiply both sides by $\sigma^3$: $-I\sigma^2 + \sum_{i=0}^{I-1} (x_i - \mu)^2 = 0$
    3.  Isolate $\sigma^2$: $I\sigma^2 = \sum_{i=0}^{I-1} (x_i - \mu)^2$

Solving this yields the maximum likelihood estimator for the variance:
$$\hat{\sigma}^2_{MLE} = \frac{1}{I} \sum_{i=0}^{I-1} (x_i - \hat{\mu}_{MLE})^2$$


## 2. The Breakthrough: Expectation-Maximization (EM) Algorithm

### 2.1 The Latent Variable Problem
MLE fails when our data is "incomplete." Consider a **Gaussian Mixture Model (GMM)**:
- We see the data points $X$.
- We **do not see** which Gaussian component $Z$ generated each point.
If we knew $Z$, we could use MLE. If we knew the parameters $\theta$, we could guess $Z$. This circular dependency is what EM solves.

### 2.2 The Mathematical Mechanism
EM is an iterative optimization strategy that moves toward a local maximum of the marginal likelihood $P(X|\theta) = \sum_{Z} P(X, Z | \theta)$.

#### A. The E-Step (Expectation)
We don't know the latent variables $Z$, so we calculate the "responsibility" or the posterior probability of $Z$ given the current parameter estimate $\theta^{(t)}$:
$$w_{ik} = P(z_i = k | x_i, \theta^{(t)})$$
Then, we define the **Auxiliary Function (Q-function)**:
$$Q(\theta | \theta^{(t)}) = \sum_{Z} P(Z|X, \theta^{(t)}) \log P(X, Z | \theta)$$

#### B. The M-Step (Maximization)
We update the parameters by maximizing the Q-function:
$$\theta^{(t+1)} = \arg \max_{\theta} Q(\theta | \theta^{(t)})$$

### 2.3 Why Does it Work? (Jensen's Inequality)
The EM algorithm works by creating a lower bound (surrogate function) on the log-likelihood at each step. By maximizing this lower bound, we guarantee that the true log-likelihood $\ell(\theta)$ never decreases:
$$\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$$
This property ensures convergence, although it may settle at a **local optimum** rather than the global one.

---

## 3. Comparative Analysis and Use Cases

### 3.1 MLE vs. EM: A Technical Summary
| Criteria | MLE | EM |
| :--- | :--- | :--- |
| **Data Nature** | Fully Observed | Partially Observed / Latent |
| **Optimization** | Gradient Descent / Analytical | Coordinate Ascent on Q-function |
| **Guarantees** | Global optimum for convex models | Local optimum convergence |
| **Computational Cost** | Low to Moderate | High (due to iterations) |

### 3.2 Real-World Applications
1. **Gene Sequencing:** Estimating haplotype frequencies (latent variables).
2. **Computer Vision:** Image segmentation using GMMs.
3. **NLP:** Training Hidden Markov Models (HMM) via the Baum-Welch algorithm (a specific case of EM).
4. **Finance:** Estimating volatility in models with regime switching.

---

## 4. Advanced Topics and Limitations
- **Initialization Sensitivity:** Since EM converges to local maxima, the choice of $\theta^{(0)}$ is critical. Techniques like K-means initialization are often used.
- **Convergence Rate:** EM can be slow to converge near the optimum compared to second-order methods like Newton-Raphson.
- **Variational Inference:** For models where the E-step is computationally intractable, we use Variational EM to approximate the posterior.

---
*End of Chapter. Created for specialized study on Statistical Inference.*