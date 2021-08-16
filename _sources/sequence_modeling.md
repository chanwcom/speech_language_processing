# Sequence Modeling

## Recurrent Neural Network (RNN)




## Sequence Loss

In the previous Chapter, we observe that the Cross Entropy (CE) loss is given
by the following equation:

```{math}
  :label: ce_loss_none_seq
  \mathbb{L}_{\text{CE}} 
         & =  -E_{\bsf{y} \sim \text{training_data}} \left[ \log \hat{\bsf{y}} \right]. \\
```
where $\hat{\bsf{y}}$ is the *class probability* predicted by the model.

If we represent $\hat{\bsf{y}}$ as the predicted probability, then we obtain the
following equation:

```{math}
  :label: class_prob
  \hat{\bsf{y}} = \hat{p}(\bsf{y} | \bsf{x}).
```

In the sequence modelling, we use the same equation as in 
{eq}`ce_loss_none_seq`. Note that in the above equation {eq}`class_prob`,
$\bsf{y}$ is not the ground-truth label, but just a dummy vector variable.

Now, the input and the predicted output may be sequences as follows:
```{math}
  \bsf{x}_{0:M}  & = \left[ \bsf{x}_0,\, \bsf{x}_1,\,  
                           \bsf{x}_2,\, \cdots, \,
                           \bsf{x}_{M-1} \right]   \\
  \bsf{y}_{0:L}  & = \left[ \bsf{y}_0,\, \bsf{y}_1,\,  
                           \bsf{y}_2,\, \cdots, \,
                           \bsf{y}_{L-1} \right]
```

Using this, the equation {eq}`class_prob` may be expressed as:
```{math}
  :label: seq_prob
  \hat{\bsf{y}}_{0:L} = \hat{p}(\bsf{y}_{0:L} | \bsf{x}_{0:M}).
```

Usually, it is not tractable to directly calculate the sequence probability in
{eq}`seq_prob`. Thus, we usually take the conditional independence assumption:
```{math}
  :label: seq_prob_assumption
  \hat{\bsf{y}}_{0:L} & = \Pi_{l=0}^{L-1} \hat{p}(\bsf{y}_l | \bsf{x}_{0:M}) \\
                & = \Pi_{l=0}^{L-1} \hat{y}_l
```

By putting {eq}`seq_prob_assumption` into {eq}`ce_loss_none_seq`, we obtain the
following sequence loss:
```{math}
  :label: ce_loss_seq
  \mathbb{L}_{\text{CE}} 
         & =  -E_{\bsf{y}_{0:L} \sim \text{training_data}} \left[ \sum_{l=0}^{L-1} \log \hat{\bsf{y}}_l \right]. \\
```
Even though we use a fixed value of $L$ in {eq}`ce_loss_seq`, the length may
vary for each example in the training data set.
When there are $V$ classes, then {eq}`ce_loss_seq` is represented by:
```{math}
  :label: ce_loss_seq_element
  \mathbb{L}_{\text{CE}} 
         & =  -E_{\bsf{y}_{0:L} \sim \text{training_data}} \left[ 
            \sum_{l=0}^{L-1} \sum_{v=0}^{V-1} (y_l)_v \log (\hat{\bsf{y}}_l)_v \right]. \\
```

In Tensorflow, it is implemented as tfa.seq2seq.sequence_loss method.



## Back-Propagation Through Time (BPTT)

For deep neural network models, we may not directly obtain the gradient. Thus
we use the *chain rule* to obtain the gradient with respect to a certain 


