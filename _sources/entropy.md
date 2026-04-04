
# Entropy




# Cross-Entropy





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

