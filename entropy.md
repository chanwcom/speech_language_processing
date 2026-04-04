
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


$$
\begin{align}
    \mathcal{L} = - \sum _{X \in \mathcal{X}^L } p(X) \log \hat{p}\left(X \;|\; \theta \right)
\end{align}
$$


Using the definition of conditional probability, $p(X)$ can be
expressed as:
$$
\begin{align}
    p(X) & = p(x_{L-1},\,\underbrace{x_{L-2}, \cdots ,\,x_0}_{x_{0:L-1}}) \nonumber \\
         & = p\left(x_{L-1} \mid  x_{0:L-1} \right) 
            p\left(x_{0:L-2} \right)  \nonumber \\
         & = p\left(x_{L-1} \mid  x_{0:L-1} \right) 
            p\left(x_{L-2} \mid  x_{0:L-2} \right) 
                p(x_{0:L-2} ) \nonumber \\
         &   \hspace{30mm}           \vdots \nonumber \\
        & = \prod_{l=1}^{L-1} p\left(x_l \mid x_{0:l} \right) p(x_0).
\end{align}
$$

Let us define $p(x_0 \mid x_{0:0})$
$$
    \begin{align}
        p(x_0 \mid x_{0:0}) \coloneqq p(x_0),
    \end{align}
$$

(3) can be represented by:
$$
\begin{align}
    p(X) = \sum_{l=0}^{L-1} p(x_l \mid x_{0:l}).
\end{align}
$$

In a similar way, we obtain:
$$
\begin{align}
x
\end{align}
$$

Using (3), we represent (2) as follows:

$$
\begin{align}
    \mathcal{L} = - \sum_{}^{} p(X) 
\end{align}
$$

