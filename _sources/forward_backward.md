# The forward-backward algorithm

From E-M theory, we obtained the following relationship:

$$
\begin{aligned}
Q\left(\theta \mid \theta^{(t)}\right) & = \sum_{Z \in \mathcal{Z}^L} 
    \hat{p}\left(Z \mid X, \theta^{(t)}\right)  \log \hat{p}(X, Z \mid \theta) \\
& =   \sum_{K \in \mathcal{K}^L} \hat{p}\left(K \mid X, \theta^{(t)}\right)  \log \hat{p}(K \mid \theta). 
\end{aligned}
$$
where $K$ is a sequence of complete data, and $X$ is a sequence of observed data.

Since $\log \hat{p}(K \mid \theta)$ can be represented in the following way:

$$
    \begin{align}
    \log \hat{p}\left(K \mid \theta \right) = \sum_{l=0}^{L-1} \log \hat{p} 
        \left(k_l  \mid K_{0:l}, \, \theta \right),
    \end{align}
$$


$$
    \begin{align}
    Q\left(\theta \mid \theta^{(t)}\right) & =  
        \sum\limits_{l=0}^{L-1}
            \sum_{K \in \mathcal{K}^L}  \hat{p}\left(K \mid X, \theta^{(t)}  \right) \log \hat{p} \left(k_l  \mid K_{0:l}, \, \theta \right) \nonumber \\
        & = \sum\limits_{l=0}^{L-1} \sum_{K \in \mathcal{K}^L} 
            \hat{p}\left(K | X,\, \theta^{(t)}\right)  
                \log \hat{p} \left(k_l  \mid K_{0:l}, \, \theta \right)  \nonumber \\ 
        & = \sum\limits_{l=0}^{L-1} \sum_{K_{0:l+1} \in \mathcal{K}^{l+1}} 
           \underbrace{ \sum_{K_{l+1:L} \in \mathcal{K}^{L-l-1}}
            \hat{p}\left(K | X,\, \theta^{(t)}\right)  }_{\text{Marginalization}}
                \log \hat{p} \left(k_l  \mid K_{0:l}, \, \theta \right)  \nonumber \\
       & = \sum\limits_{l=0}^{L-1} \sum_{K_{0:l+1} \in \mathcal{K}^{l+1}} 
            \hat{p}\left(K_{0:l+1} | X,\, \theta^{(t)}\right)  
                \log \hat{p} \left(k_l  \mid K_{0:l}, \, \theta \right)   \nonumber \\
        & = \sum\limits_{l=0}^{L-1} \sum_{K_{0:l} \in \mathcal{K}^{l}} 
            \hat{p} \left(K_{0:l}  \mid X,\,\theta^{(t)}\right) \sum_{k_{l} \in \mathcal{K}} 
            \hat{p}\left(k_l \mid X,\,K_{0:l},\, \theta^{(t)}\right)  
                \log \hat{p} \left(k_l  \mid K_{0:l}, \, \theta \right)  \nonumber \\
        & = 
    \end{align}
$$

If we assume conditional independence, 

$$
    \begin{align}
        \log \hat{p} \left(k_l  \mid K_{0:l}, \, \theta \right) 
            = \log \hat{p} \left(k_l  \mid \theta \right) 
    \end{align}
$$

Under this assumption:

$$
\begin{align}
    Q\left(\theta \mid \theta^{(t)}\right) 
        & = \sum\limits_{l=0}^{L-1} \underbrace{ \sum_{K_{0:l+1} \in \mathcal{K}^{l}}  
            \hat{p} \left(K_{0:l}  \mid X,\,\theta^{(t)}\right)}_{\text{Marginalization}} \sum_{k_{l} \in \mathcal{K}} 
            \hat{p}\left(k_l \mid X,\, \theta^{(t)}\right)  
                \log \hat{p} \left(k_l  \mid \theta \right) \nonumber \\
        & = \sum_{l=0}^{L-1} \sum_{k_l \in \mathcal{K}} 
            \hat{p}\left(k_l \mid X,\, \theta^{(t)}\right)  
                \log \hat{p} \left(k_l  \mid \theta \right) 
\end{align}
$$

Let us define the estimated target (ET) $\mathbf{z}_l$ as

$$
    \begin{align}
    (\mathbf{z}_l)_j = \hat{p}\left(j \mid X,\, \theta^{(t)}\right) 
    \end{align}
$$
