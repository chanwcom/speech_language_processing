# Neural Network Basics

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

\begin{align}
  \mathbb{L} = \text{loss}(y, \hat{y}) 
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



## Back-Propagation Through Time (BPTT)


### Using a directive

At its simplest, you can insert a directive into your book's content like so:

````
```{mydirectivename}
My directive content
```
````

This will only work if a directive with name `mydirectivename` already exists
(which it doesn't). There are many pre-defined directives associated with
Jupyter Book. For example, to insert a note box into your content, you can
use the following directive:

````
```{note}
Here is a note
```
````

This results in:

```{note}
Here is a note
```

In your built book.

For more information on writing directives, see the
[MyST documentation](https://myst-parser.readthedocs.io/).


### Using a role

Roles are very similar to directives, but they are less-complex and written
entirely on one line. You can insert a role into your book's content with
this pattern:

```
Some content {rolename}`and here is my role's content!`
```

Again, roles will only work if `rolename` is a valid role's name. For example,
the `doc` role can be used to refer to another page in your book. You can
refer directly to another page by its relative path. For example, the
role syntax `` {doc}`intro` `` will result in: {doc}`intro`.

For more information on writing roles, see the
[MyST documentation](https://myst-parser.readthedocs.io/).


### Adding a citation

You can also cite references that are stored in a `bibtex` file. For example,
the following syntax: `` {cite}`holdgraf_evidence_2014` `` will render like
this: {cite}`holdgraf_evidence_2014`.

Moreoever, you can insert a bibliography into your page with this syntax:
The `{bibliography}` directive must be used for all the `{cite}` roles to
render properly.
For example, if the references for your book are stored in `references.bib`,
then the bibliography is inserted with:

````
```{bibliography}
```
````

Resulting in a rendered bibliography that looks like:

```{bibliography}
```


### Executing code in your markdown files

If you'd like to include computational content inside these markdown files,
you can use MyST Markdown to define cells that will be executed when your
book is built. Jupyter Book uses *jupytext* to do this.

First, add Jupytext metadata to the file. For example, to add Jupytext metadata
to this markdown page, run this command:

```
jupyter-book myst init markdown.md
```

Once a markdown file has Jupytext metadata in it, you can add the following
directive to run the code at build time:

````
```{code-cell}
print("Here is some code to execute")
```
````

When your book is built, the contents of any `{code-cell}` blocks will be
executed with your default Jupyter kernel, and their outputs will be displayed
in-line with the rest of your content.

For more information about executing computational content with Jupyter Book,
see [The MyST-NB documentation](https://myst-nb.readthedocs.io/).
