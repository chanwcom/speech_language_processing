

# Maximum Likelihood (ML) and Expectation-Maximization (EM)

## Introduction
In the realm of statistical machine learning, we often face the challenge of estimating the parameters of a model that best explains our observed data. This chapter provides an in-depth technical exploration of **Maximum Likelihood Estimation (MLE)**—the gold standard for complete data—and the **Expectation-Maximization (EM) Algorithm**, the essential tool for dealing with latent variables and missing data.

---

## 1. Maximum Likelihood Estimation (MLE): The Foundation

### 1.1 Philosophical Underpinning
MLE is built on a simple yet powerful frequentist principle: **"The best parameters are those under which the observed data is most likely to have occurred."**

If we have a set of independent and identically distributed (i.i.d.) observations 
$X = \{x_0, x_1,  \cdots, x_{I-1}\}$, and a probability model $P(X|\theta)$, the likelihood function $L(\theta)$ is given by:

$$L(\theta) = P(x_0, x_1, \cdots, x_{I-1} \mid \theta) = \prod_{i=0}^{I-1} P(x_i \mid \theta).$$

The Maximum Likelihood Estimation (MLE) is given by the following equation:

$$ 
\begin{align}
    \hat{\theta} = \arg \max_{\theta} L(\theta)
\end{align}
$$

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

Since log function is a monotonic function, the Maximum Likelihood Estimation (MLE)
is  given by the following equation in terms of log-likelihood:

$$ 
\begin{align}
    \hat{\theta} = \arg \max_{\theta} \ell(\theta).
\end{align}
$$

### 1.3 Step-by-Step Analytical Derivation
1. **Define the Model:** Choose a distribution (e.g., Gaussian, Bernoulli).
2. **Construct the Log-Likelihood:** Sum the logs of individual densities.
3. **Compute Derivatives:** Find the score function $ \frac{\partial \ell(\theta) }{\partial \theta \hfill }$.
4. **Solve the Equation:** Set $ \nabla_{\theta} \ell(\theta) = 0$.
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


## 2. Expectation-Maximization (EM) Algorithm

In statistics, an expectation–maximization (EM) algorithm is an iterative method to find (local) maximum likelihood or maximum a posteriori (MAP) estimates of parameters in statistical models, where the model depends on unobserved latent variables.


### 2.1 The Latent Variable Problem
Maximum Likelihood Estimation (MLE) fails when our data is "incomplete." Consider a **Gaussian Mixture Model (GMM)**:
- We see the data points $X = \{x_0, x_1,  \cdots, x_{I-1}\}$.
- We **do not see** which Gaussian component $Z = \{z_0, z_1,  \cdots, z_{I-1}\}$ generated each point.
If we knew $Z$, we could use MLE. If we knew the parameters $\theta$, we could guess $Z$. This circular dependency is what EM solves.

### 2.2 The Mathematical Mechanism
In Maximum Likelihood (ML) framework, the estimated parameter $\hat{\theta}$ is given by the following equation:

$$
    \begin{aligned}
        \hat{\theta} = \arg \max_{\theta} p(X \mid \theta).
    \end{aligned}
$$

EM is an iterative optimization strategy that moves toward a local maximum of the marginal likelihood $p(X|\theta) = \sum\limits_{Z \in \mathcal{Z}} p\left(X, Z \mid \theta\right)$. As mentioned in the above section, $X$ and $Z$ are observed data and latent data, respectively:

$$
\begin{aligned}
    X & = \{x_0, x_1,  \cdots, x_{I-1}\}, \\
    Z & = \{z_0, z_1,  \cdots, z_{I-1}\}
\end{aligned}
$$

The fundamental assumption is that while the **complete-data likelihood** $p(X, Z \mid \theta)$ is defined, $Z$ is not observed, making direct maximization of $p(X \mid \theta)$ difficult.

#### A. The E-Step (Expectation)
We do not know the latent variables $Z$, so we calculate the "responsibility" or the posterior probability of $Z$ given the current parameter estimate $\theta^{(t)}$:
$$W = P\left(Z \mid X, \theta^{(t)}\right)$$
Then, we define the **Auxiliary Function (Q-function)**:

$$
Q(\theta \mid \theta^{(t)}) =
    \sum_{Z \in \mathcal{Z}^{I}} P\left(Z \mid X,\, \theta^{(t)}\right)
    \log P\left(X,\, Z \mid \theta\right)
$$

#### B. The M-Step (Maximization)
We update the parameters by maximizing the Q-function:

$$\theta^{(t+1)} = \arg \max_{\theta} Q(\theta | \theta^{(t)})$$

### 2.3 Further Modification of the E-Step Equation

Let us modify the **auxilary function** given in the following form:

$$
\begin{aligned}
Q(\theta | \theta^{(t)}) & = 
    \sum_{Z \in \mathcal{Z}^{I}} P\left(Z \mid X, \theta^{(t)}\right) 
    \log P\left(X, Z \mid \theta\right), 
\end{aligned}
$$

From the iid assumption, we have:

$$
\begin{align}
    \log p\left(X, Z \mid \theta\right) = \sum_{i=0}^{I-1} 
        \log p\left(x_i, z_i \mid \theta\right), \\
    xx
\end{align}
$$ 

The auxiliary function can be written:

$$
\begin{align}
    Q(\theta | \theta^{(t)}) 
     & =   \sum_{Z \in \mathcal{Z}^I} p\left(Z \mid X, \theta^{(t)}\right) 
       \sum_{i=0}^{I-1}  \log p\left(x_i, z_i \mid \theta\right),  \nonumber \\
    & = 
       \sum_{i=0}^{I-1} \sum_{Z \in \mathcal{Z}^I} p\left(Z \mid X, \theta^{(t)}\right) 
        \log p\left(x_i, z_i \mid \theta\right), 
\end{align}
$$
Under the independence assumption,
$$
\begin{align}
    p\left(Z \mid X, \theta^{(t)}\right)  = p(Z_{\\i})
\end{align}
$$


Marginalization of $p\left(Z \mid X, \theta^{(t)}\right)$ over all $Z = \{z_0, z_1, \cdots z_{I-1} \}$ except the index $i$ leads to
$$
\begin{align}
    \sum_{Z \in \mathcal{Z}^I} p\left(Z \mid X, \theta^{(t)}\right) 
        = \sum_{z_i \in \mathcal{Z}} p\left(z_i \mid X, \theta^{(t)}\right) 
\end{align}
$$


### 2.4 Why Does it Work? (Jensen's Inequality)
The EM algorithm works by creating a lower bound (surrogate function) on the log-likelihood at each step. By maximizing this lower bound, we guarantee that the true log-likelihood $\ell(\theta)$ never decreases:
$$\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$$
This property ensures convergence, although it may settle at a **local optimum** rather than the global one.

### 2.5 Alternative Representation of the Auxiliary Function

Instead of using $X$ and $Z$, we may define the complete
data by:

$$ K = (X, Z), $$

where $k_i = (x_i, z_i), \qquad 0 \le i \le I -1 $.

Noting that $P\left(Z \mid X, \theta^{(t)} \right) = P(X, Z \mid X, \theta^{(t)} ) = P\left(K \mid X, \theta^{(t)}\right) $, the **Auxiliary Function** can be represented by:

$$
\begin{aligned}
Q\left(\theta \mid \theta^{(t)}\right) & = \sum_{Z \in \mathcal{Z}^I} 
    P\left(Z \mid X, \theta^{(t)}\right)  \log P(X, Z \mid \theta) \\
& =   \sum_{k \in \mathcal{K}^I} P\left(K \mid X, \theta^{(t)}\right)  \log P(K \mid \theta). 
\end{aligned}
$$

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