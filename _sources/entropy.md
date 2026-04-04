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
    p(X) = p(x_{L-1},\,x_{L-2}, \cdots, ,x_0)
\end{align}
$$



