# Neural Network Basics

There are various applications of neural-network models. In this section, for
simplicity of discussion, we will focus only on *the classification problem*.

## Model

The model can be considered a function to predicts the output class given the
input.


When the number of output classes is $V$, then $y$ may take a value in the
following range:
```{math}
  0 \le y \le V - 1.
```
In stead of the above *sparse representation of the output class*, we may use
*the one-hot vector* representation as below:


## Training Set

In this section, we consider the supervised training case, which might be the
most basic case of training neural-network models.
Other cases such as unsupervised or semi-supervised training cases will 
be covered in TODO(chanw.com). 

In the supervised training, we are given a set of labeled data. Each element of
this set is usually the input $\bsf{x}^{(i)}$

\begin{align}
  \mathcal{T} = \big \{ <\bsf{x}^{(i)},\, y^{(i)} > | 0 \le i \le N_{\text{tr}} - 1 \big \}
\end{align}


## Loss Function
Suppose that the neural-network model $f$ generates output $\hat{y}$  

```{math}
  :label: entire_loss
  \mathbb{L} = \frac{1}{N_{\text{tr}}} \sum_{i=0}^{N_{\text{tr} - 1}}
    \text{loss}(y^{(i)}, \hat{y}^{(i)}) 
```

Now, the question is what would be a good candidate for the loss function in
{eq}`entire_loss`.

As will be discussed later, an appropriate loss function is different 
depending on applications. TODO(chanw.com) Mention where it will be or was
discussed.

In the classification problem, we assume that for a specific input $\bsf{x}$

\begin{align}
  \mathbb{L} = - E \left[ \log(f( | )) \right]
\end{align}


## Gradient Descent 

\begin{align}
  \bsf{w} \leftarrow  \bsf{w} - \mu \nabla_{\bsf{w}} \mathbb{L}
\end{align}

Gradient Descent (GD) is not a practical approach when the training set size is
sufficiently large for the following two reasons.

 * Inefficiency in computation

 * Slow convergence


The Gradient Descent (GD) approach described above is not practical when the
training set size is large for the following two reagons. 

This is because the parameter update represented by

## Stochastic Gradient Descent (SGD)




() happens only once for the entire training set.

\begin{align}
  \bsf{w} \leftarrow  \bsf{w} - \mu \nabla_{\bsf{w}} \mathbb{L}_i
\end{align}



## Back-Propagation 



