## Unaligned Sequence Loss

In this section, we derive the derivative of the sequence entropy loss
when the target and model output are not aligned. This problem
frequently occurs when modalities are different.

Suppose that we have a target sequence given by:

$$
    \begin{align}
        C = \left[c_0, \, c_1, \cdots , c_{L-1}\right]
    \end{align}
$$

The model output sequence is given by:

$$
    \begin{align}
        \hat{Z} = \left[\hat{z}_0, \, \hat{z}_1, \cdots \hat{z}_{S-1} \right]    
    \end{align}
$$

$\hat{Z}$ and $C$ are not aligend. Thus, let us consider an alignment function
, which maps the step index $s$ into a tuple of time index $t$ and target token 
index $l$.

$$
    \begin{align}
        \phi(s) =  (r(s), q(s)) = (t, l).
    \end{align}
$$

Then, using this alignment function $\phi(s)$ , we obtian the following
model-output-aligned target:

$$
    \begin{align}
    K = \left[k_0, \cdots, k_l, \cdots, k_{S-1} \right]
    \end{align}
$$

where 

$$
    \begin{align}
        k_s = c_{q(s)}, \quad 0 \le s \le S-1.
    \end{align}
$$

This model-output-aligned target sequence $K$ is not directly 
observable, thus, it can be considered a latent variable.


For a specific $K$, now, we can calculate the loss $\mathcal{L}(K)$ as follows:

$$
    \begin{align}
    \mathcal{L}({K}) & = -  \log  \hat{p} \left( K \mid X,\, \theta \right) \nonumber \\
              &  = -  \sum_{s=0}^{S-1} \log  \hat{p} \left( k_s \mid K_{0:s},\, X,\, \theta \right).
    \end{align}
$$

where $X$ is the input sequence given by:

$$
    X = \left[\mathbf{x}_0, \mathbf{x}_1, \cdots, \mathbf{x}_{T^{\text{(in)}} - 1} \right]
$$


Let us represent the model output at the step index $s$ by $\hat{\mathbf{z}}_s(K)$
where the $j$-th component is given by:

$$
    \hat{z}_{s,j}(K) = \hat{p} \left(k_s = j  \mid K_{0:s}, \, X,\, \theta \right).  
$$



When employing the softmax layer with a cross-entropy loss, the
gradient of the loss $\mathcal{L}$ with respect to the logit vector
$\mathbf{h}^{\text{(logit)}}_s$ simplifies to the difference between the
prediction and the ground truth:

$$
    \begin{align}
    \dfrac{\partial \mathcal{L} \hspace{8mm}}
            {\partial   {\mathbf{h}^{\text{(logit)}}}_s } 
    = - \left(\mathbf{e}_{k_s} - \hat{\mathbf{z}}_s(K) \right)^\intercal,
    \end{align}
$$

where $\hat{\mathbf{z}}_s(K)$
represents the vector of predicted probabilities, and $\mathbf{e}_{k_s}$ is a
one-hot encoded vector whose $k_s$-th element is 1 and all other elements are
0, representing the ground truth label for the $s$-th sample.  During the
training phase, if we can assume that $\hat{\mathbf{z}}_s(K)$ does not depend on
$K$, the derivative with respect to the logit becomes easier. We will
discuss when we make this assumption in the next setion. When this assumption
holds, we obtain the following derivatie:

$$
    \begin{align}
    \dfrac{\partial \mathcal{L} \hspace{8mm}}
            {\partial   {\mathbf{h}^{\text{(logit)}}}_s } 
    = - \left(\mathbf{e}_{k_s} - \hat{\mathbf{z}}_s \right)^\intercal
    \end{align}
$$

Motivated by the idea of Expectation-Maximization (EM), the entire loss
$\mathcal{L}$ is given by considering the distribution of latent variables
$K$:


$$
    \begin{align}
    \mathcal{L}_Q
        & =  \mathbb{E}_{K \sim \hat{p}(\cdot \mid C, X, \theta')} 
            { \log \hat{p}(K \mid X,\, \theta)   } \nonumber \\
        & =  \sum_{K \in \mathcal{K}^{S} } \hat{p}(K \mid C,\, X,\, \theta')
            { \log \hat{p}(K \mid X,\, \theta)   } 
    \end{align}
$$


Performing differentiation with respect to $\mathbf{h}^{\text{(logit)}}_s$ 
leads to:

$$
    \begin{align}
    \dfrac{\partial \mathcal{L} \hspace{8mm}}{\partial   {\mathbf{h}^{\text{(logit)}}}_s } 
    & = - \sum_{K \in \mathcal{K}^S} p(K \mid C,\,X,\,\theta') 
        \left(\mathbf{e}_{k_s} - \hat{\mathbf{z}}_s \right)^\intercal \nonumber \\
    \end{align}
$$

Let us define A and B as follows:

$$
    \begin{align}
    A & :=\sum_{K \in \mathcal{K}^S} p(K \mid C,\,X,\,\theta')  \mathbf{e}_{k_s}  \nonumber \\
    B & :=\sum_{K \in \mathcal{K}^S} p(K \mid C,\,X,\,\theta')  \hat{\mathbf{z}}_{s}(K)  \nonumber \\
    \end{align}
$$

Let's first arrange A. Separating the summation 
$\sum_{K \in \mathcal{K}^S} $ into two summations
$\sum_{k_s \in \mathcal{K}} \sum_{K_{\setminus s} \in \mathcal{K}^{S-1}} $,
and noting that $\sum_{k_s \in \mathcal{K}}$ is identical to $\sum_{k_s = 1}^{C-1}$ leads to:

$$
    \begin{align}
        A & =   \sum_{k_s=0}^{C-1} \underbrace{\sum_{K_{\setminus s} \in \mathcal{K}^{S-1}}
                p(K_{\setminus s} k_s  \mid C,\,X,\,\theta')}_{\text{Marginalization}}  \mathbf{e}_{k_s} \nonumber \\
         & = \sum_{j=0}^{C-1}  p(k_s = j \mid C,\,X,\,\theta') \mathbf{e}_{j} 
    \end{align}
$$

In case of the second term $B$, since $\hat{\mathbf{z}}_s(K)$ also depends on $K$, 
we cannot further simply the expression. If we may assume $\hat{\mathbf{z}}_s(K)$ 
does not depend on $K$, and simply represent it by $\hat{\mathbf{z}}$, then 
$B$ is simplified using marginalization as follows:

$$
    \begin{align}
    B = \hat{\mathbf{z}}_{s}  
    \end{align}
$$

In the next section, we show examples of three cases where this assumption
is met.

Using (XX) and (XX), the derivative in (XX) is simplified as:

$$
    \begin{align}
    \dfrac{\partial \mathcal{L} \hspace{8mm}}{\partial   {\mathbf{h}^{\text{(logit)}}}_s } 
    & = - \left[ \sum\limits_{j=0}^{C-1} p(k_s =j \mid C,\,X,\,\theta') \mathbf{e}_j
            - \hat{\mathbf{z}}_s  \right]^{\intercal}
    \end{align}
$$

From the above equation, we define an Estimated Target (ET) vector 
$\tilde{\mathbf{z}}_s$ whose $j$-th element $(0 \le j \le C-1)$ is defined by:

$$
    \begin{align}
        \tilde{z}_{s, j} := p(k_s =j \mid C,\,X,\,\theta').
    \end{align}
$$

The Estimated Target vector sequence $\hat{Z}$ is given by:

$$
    \tilde{Z} = \left[\tilde{\mathbf{z}}_0, \, \tilde{\mathbf{z}}_1, \cdots \tilde{\mathbf{z}}_{S-1} \right].
$$

Then, the above equation can be written as 

$$
    \begin{align}
    \dfrac{\partial \mathcal{L} \hspace{8mm}}{\partial   {\mathbf{h}^{\text{(logit)}}}_s } 
    & = - \left(\tilde{\mathbf{z}}_s  - \hat{\mathbf{z}}_s \right)^\intercal 
    \end{align}
$$

then $\mathcal{L}$ is the sequence cross entropy-loss between 
$\tilde{Z}$ and $\hat{Z}$. Thus $\tilde{\mathbf{z}} = (\tilde{z}_j)_{0}^{C-1}$
can be considered as Estimated Target (ET).

## Factoring Out Dependency from Model Prediction

In the above derivation, we made the following assumption:

$$
    \hat{z}(k_s=j \mid  K_{0:s},\, \theta) = \hat{z}(k_s=j \mid \theta)
$$


There are three ways of achieving this objective:

 - Conditional Independence Assumption Approach

In this case, we assume that the model prediction probability at step index $s$ 
does not depend on the seqeunce history at all:
$$
    (\hat{z}_s)_j = \hat{p} \left(k_s = j  \mid X,\, \theta \right).  
$$
This is a very strong assumption, which will impact the speech recognition
accuracy to some degree.

 - Axis Separation with Teacher Forcing Approach
 

Instead of imposing a strict conditional independence assumption, 
we can maintain the dependency on the previous label sequence 
by restructuring the step index. We decompose the unified index $s$ 
into two distinct axes: time ($r(s)$) and label index ($q(s)$):

$$
    \begin{align}
        (t, l) = (r(s), q(s)).
    \end{align}
$$

In this framework, the model prediction at a specific grid point $(t, l)$ 
is represented as:

$$
    \begin{align}
        (\hat{z}_{t,l})_j = \hat{p} \left(k_{t,l} = j \mid K_{0:s}, X, \theta \right).
    \end{align}
$$

Under the Teacher-Forcing paradigm, we replace the dependency on the 
potentially stochastic, model-aligned latent sequence $K_{0:s}$ with 
the fixed ground-truth prefix $C_{0:l}$. Consequently, the probability 
becomes:

$$
    \begin{align}
    (\hat{z}_{t,l})_j = \hat{p} \left(k_{t,l} = j \mid C_{0:l}, X, \theta \right).
    \end{align}
$$

Since $C_{0:l}$ is a fixed constant provided by the training data, the
model prediction $(\hat{z}_{t,l})_j$ at each coordinate $(t, l)$ no longer
depends on the specific path taken by the latent variable $K$ during the
expectation calculation. This allows us to factor $(\hat{z}_{t,l})$ out of the
summation over the latent paths $\mathcal{K}^S$ when computing the gradient,
effectively simplifying the derivative of the total sequence loss $\mathcal{L}$
while still preserving the essential linguistic context provided by $C_{0:l}$.


 - Model Output Feedback Approach
 
 In this approach, we do not perform teacher-forcing, but directly apply
model output as the feedback. Then the probability depdens on $\hat{K}_{0:s}$
rather than $K_{0:s}$.

$$
    \begin{align}
        \hat{z}_s = \hat{p} \left(k_{t,l} = j \mid \hat{K}_{0:s}, X, \theta \right).
    \end{align}
$$
