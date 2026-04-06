
# Entropy




# Cross-Entropy



## Differentiation of Cross-Entropy

To find the derivative of the Cross-Entropy Loss $\mathcal{L}$ with 
respect to the logit $\mathbf{h}^{\text{(logit)}}$, 
we typically look at a single data point in a multi-class 
classification setting.

The cross-entropy loss $\mathcal{L}$ is given by the following equation:
$$
    \begin{align}
        \mathcal{L} & = - \mathbf{y}^{\intercal} \log(\mathbf{\hat{y}}) \nonumber \\
                    & = - \sum_{j=0}^{C-1} p(j) \log \hat{p}(j),
    \end{align}
$$
where $p$ and $\hat{p}$ are ground-truth and model-prediction proabibilities
respectively.

The softmax layer is characterized by the following equation:
$$
    \begin{align}
        \hat{y}_j &  = \sigma \left(h^{\text{(logit)}}_j \right) \nonumber \\  
                  &  =  \dfrac{\exp\left(h^{\text{(logit)}}_j\right)  }
                    { \sum\limits_{k=0}^{C-1} \exp \left( h^{\text{(logit)}}_k\right) } ,
    \end{align}
$$
where $\mathbf{\hat{y}}$ is the softmax output, which is also the model output.

From the Chain Rule, we obtian the following relationship:

$$
    \begin{align}
        \dfrac{\partial \mathcal{L} \hspace{6mm}}{\partial \mathbf{h}^{\text{(logit)}}} = 
        \dfrac{\partial \mathcal{L} }{\partial \mathbf{\hat{y}}}
        \dfrac{\partial \mathbf{\hat{y}} \hspace{7mm}}{\partial \mathbf{h}^{\text{(logit)}}} 
    \end{align}
$$

For the $j$-th element of $\dfrac{\partial \mathcal{L} \hspace{6mm}}{\partial \mathbf{h}^{\text{(logit)}}}$, 
which is $\left(\dfrac{\partial \mathcal{L} \hspace{6mm}}{\partial \mathbf{h}^{\text{(logit)}}}\right)_j$,
this chain-rule can be expressed by the following summation:

$$
    \begin{align}
  \left(\dfrac{\partial \mathcal{L} \hspace{6mm}}{\partial \mathbf{h}^{\text{(logit)}}}\right)_j 
    = \sum_{k=0}^{C-1}  \left( \frac{\partial \mathcal{L}}{\partial \hat{y}_k}  \right)
        \left( \frac{\partial \hat{y}_k \hspace{6mm}}{\partial h^{\text{(logit)}}_j  } \right)
    \end{align}
$$

Now, let us further arrange equation (4) by calculating 
$\dfrac{\partial \mathcal{L}}{\partial \hat{y}_k} $ and
$\dfrac{\partial \hat{y}_k \hspace{6mm}}{\partial h^{\text{(logit)}}_j  }$.

We follow a three-step derivation: 
i) differentiating the loss with respect to the softmax output, 
ii) differentiating the softmax function with respect to the logits, and 
iii) combining them using the chain rule.

### 1. Derivative of Loss with respect to Softmax Output.

The Cross-Entropy loss for a single data point is defined as:

$$\mathcal{L} = - \sum_{k=0}^{C-1} y_k \log \hat{y}_k$$

Taking the partial derivative with respect to a specific softmax output 
$\hat{y}_k$:

$$
\begin{align}
    \frac{\partial \mathcal{L}}{\partial \hat{y}_k} = \frac{\partial}{\partial \hat{y}_k} \left( - y_k \log \hat{y}_k \right) = -\frac{y_k}{\hat{y}_k}
\end{align}
$$

### 2. Derivative of Softmax Output with respect to Logits

The softmax function is given by $\hat{y}_k = \dfrac{e^{h_k}}{\sum_{m} e^{h_m}}$. We must consider two cases when differentiating $\hat{y}_k$ with respect to the $j$-th logit $h_j$:

Case 1: $k = j$ (Diagonal elements)Using the quotient rule:

$$
\begin{align}
\frac{\partial \hat{y}_j}{\partial h_j} = \frac{e^{h_j} \sum e^{h_m} - e^{h_j} e^{h_j}}{(\sum e^{h_m})^2} = \frac{e^{h_j}}{\sum e^{h_m}} \left( \frac{\sum e^{h_m} - e^{h_j}}{\sum e^{h_m}} \right) = \hat{y}_j(1 - \hat{y}_j)
\end{align}
$$

Case 2: $k \neq j$ (Off-diagonal elements)

$$
\begin{align}
\frac{\partial \hat{y}_k}{\partial h_j} = \frac{0 \cdot \sum e^{h_m} - e^{h_k} e^{h_j}}{(\sum e^{h_m})^2} = -\frac{e^{h_k}}{\sum e^{h_m}} \frac{e^{h_j}}{\sum e^{h_m}} = -\hat{y}_k \hat{y}_j
\end{align}
$$

We can unify these using the Kronecker delta $\delta_{kj}$:

$$
    \begin{align}
        \frac{\partial \hat{y}_k}{\partial h_j} = \hat{y}_k (\delta_{kj} - \hat{y}_j)
    \end{align}
$$

### 3. Combining via the Chain Rule

Now, we substitute the results from Step 1 and Step 2 into the summation formula:

$$\begin{align}
\frac{\partial \mathcal{L}}{\partial h_j} &= \sum_{k=0}^{C-1} \frac{\partial \mathcal{L}}{\partial \hat{y}_k} \frac{\partial \hat{y}_k}{\partial h_j} \nonumber \\
&= \sum_{k=0}^{C-1} \left( -\frac{y_k}{\hat{y}_k} \right) \hat{y}_k (\delta_{kj} - \hat{y}_j) \nonumber \\
&= - \sum_{k=0}^{C-1} y_k (\delta_{kj} - \hat{y}_j) \nonumber \\
&= - \left( \sum_{k=0}^{C-1} y_k \delta_{kj} - \sum_{k=0}^{C-1} y_k \hat{y}_j \right)
\end{align}$$

Since $\sum y_k \delta_{kj} = y_j$ and the ground-truth probabilities sum to one ($\sum y_k = 1$):

$$\frac{\partial \mathcal{L}}{\partial h_j} = -(y_j - \hat{y}_j) = \hat{y}_j - y_j$$

### 4. Vector Form Conclusion

Expressing this for the entire vector $\mathbf{h}^{\text{(logit)}}$:

$$
    \begin{align}
        \frac{\partial \mathcal{L}\hspace{7mm}}{\partial \mathbf{h}^{\text{(logit)}}} = 
            -(\mathbf{y} - \mathbf{\hat{y}})^\intercal.
    \end{align}
$$



# Sequence Cross-Entropy

In sequence-to-sequence tasks (like machine translation, text generation, and
speech recognition), we consider the following seqeunce:

$$
\begin{align}
    X & = [x_0\,, x_1\,, \cdots ,\,x_{L-1}],\\
\end{align}
$$

Using the definition of Cross-Entropy (CE), the CE 
for this sequence is given as follows:

$$
\begin{align}
    H(p, \hat{p}) & = -\mathbb{E}_{X \sim p} \Big[ \log \hat{p} \left( X \mid  \theta  \right)  \Big]
           \nonumber \\ 
           & =  - \sum _{X \in \mathcal{X}^L } p(X) \log \hat{p}\left(X \;|\; \theta \right)
\end{align}
$$

Before having further discussion, let us define the partial sequence as follows:

$$
\begin{align}
    X_{a:b} & = [x_a\,, x_{a+1}\,, \cdots ,\,x_{b-1}],\\
\end{align}
$$



Using the definition of conditional probability, $\hat{p}(X \mid \theta)$ can be
expressed as:

$$
\begin{align}
    \hat{p}(X \mid \theta) & = 
        \hat{p}(x_{L-1},\,\underbrace{x_{L-2}, \cdots ,\,x_0}_{X_{0:L-1}} \mid \theta) \nonumber \\
         & = \hat{p}\left(x_{L-1} \mid  X_{0:L-1},\,\theta \right) 
            \hat{p}\left(x_{0:L-1},\, \theta \right)  \nonumber \\
         & = \hat{p}\left(x_{L-1} \mid  X_{0:L-1},\, \theta \right) 
            \hat{p}\left(x_{L-2} \mid  X_{0:L-2},\, \theta \right) 
                \hat{p}(x_{0:L-2} \mid \theta ) \nonumber \\
         &   \hspace{30mm}           \vdots \nonumber \\
        & = \prod_{l=1}^{L-1} \hat{p}\left(x_l \mid X_{0:l},\, \theta \right) \hat{p}(x_0 \mid \theta).
\end{align}
$$

Let us define $p(x_0 \mid x_{0:0})$

$$
    \begin{align}
        \hat{p}(x_0 \mid X_{0:0},\,\theta) \coloneqq \hat{p}(x_0 \mid \theta),
    \end{align}
$$

(3) can be represented by:

$$
\begin{align}
    \hat{p}(X) = \prod_{l=0}^{L-1} \hat{p}(x_l \mid X_{0:l},\,\theta).
\end{align}
$$


Thus, we obtain:

$$
\begin{align}
    \log \hat{p} \left(X  \mid \theta \right) = \sum\limits_{l=0}^{L-1} \log \hat{p} \left(x_l \mid X_{0:l},\, \theta \right)
\end{align}
$$


$$
\begin{align}
    H\big(p(X),\, \hat{p}(X \mid \theta) \big)
        & = -\mathbb{E}_{X \sim p} \Big[ \log \hat{p} \left( X \mid  \theta  \right)  \Big]
\nonumber \\
        & =  - \sum_{X \in \mathcal{X}^L} p(X) \log \hat{p}(X \mid \theta) \nonumber \\
        & = - \sum_{X \in \mathcal{X}^L}
            p(X)
             \sum\limits_{l=0}^{L-1} \log \hat{p} \left(x_l \mid X_{0:l},\, \theta \right) \nonumber \\ 
        & = - \sum_{l=0}^{L-1}  \sum_{X \in \mathcal{X}^L}
            p(X)
                \log \hat{p} \left(x_l \mid X_{0:l},\, \theta \right) \nonumber \\
        & = - \sum_{l=0}^{L-1} \sum_{X_{0:l+1} \in \mathcal{X}^{l+1}}
           \underbrace{\left( \sum_{X_{l+1:L} \in \mathcal{X}^{L-l-1}}
                p(x_0,\, \cdots,\, x_{L-1} )     \right)    }_{\text{Marginalization}}
                \log \hat{p} \left( x_l \mid X_{0:l},\, \theta  \right)  \nonumber \\
        & = - \sum_{l=0}^{L-1} \sum_{X_{0:l+1} \in \mathcal{X}^{l+1}}
            p \left(X_{0:l+1} \right)
            \log \hat{p} \left( x_l \mid X_{0:l},\, \theta  \right)   \nonumber \\
        & =  - \sum_{l=0}^{L-1} \sum_{X_{0:l} \in \mathcal{X}^{l}}
              p\left(X_{0:l}\right) 
              \underbrace{  
                \sum_{x_l \in \mathcal{X}} p \left( x_l \mid X_{0:l} \right)
            \log \hat{p} \left( x_l \mid X_{0:l},\, \theta  \right)}_{
                    - H \left(
                        p \left(x_l \mid X_{0:l} \right), \,
                        \hat{p} \left( x_l \mid X_{0:l},\, \theta \right) \right) 
                        }   \nonumber \\
       & = - \sum_{l=0}^{L-1} \mathbb{E}_{X_{0:l} \sim p} 
            \left[H \Big(
                p \big(x_l \mid X_{0:l} \big), \,
                \hat{p} \left( x_l \mid X_{0:l},\, \theta \right) \Big)    \right] 
\end{align}
$$

If we respresent the loss at the label index $l$ by $\mathcal{L}_l$:

$$
    \mathcal{L}_l =H \Big(
                p \big(x_l \mid X_{0:l} \big), \,
                \hat{p} \left( x_l \mid X_{0:l},\, \theta \right) \Big).
$$

Then, we represent the entire loss $\mathcal{L}$ by

$$
    \mathcal{L} = \sum_{l=0}^{L-1} \mathcal{L}_l
$$

Now, let us reprsent $H\big(p(X),\, \hat{p}(X \mid \theta) \big)$ by $\mathcal{L}$.
In this case, the derivative of $\mathcal{L}$ with respect to $\mathbf{h}^{\text{(logit)}}_l$
is given by:

$$
\begin{align}
    \dfrac{\partial \mathcal{L} \hspace{6mm}}{\partial \mathbf{h}_l^{\text{(logit)}} }
        & = \frac{\partial \mathcal{L}_l  \hspace{6mm}} 
            {\partial \; \mathbf{h}_l^{\text{(logit)}}} \nonumber \\
        & = - \left(\mathbf{y}_l  -  \mathbf{\hat{y}}_l \right)^\intercal
\end{align}
$$