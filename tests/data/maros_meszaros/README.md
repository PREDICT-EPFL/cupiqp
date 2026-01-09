# Maros Meszaros problem files

These files originate from [osqp\_benchmarks](https://github.com/osqp/osqp_benchmarks/tree/5c79e870c4bd697383f66f5dff26aea29dc1ebfa/problem_classes/maros_meszaros_data). They were converted to the MAT format from the Maros and Meszaros SIF problem files. See the original repository for conversion details.

All problems have the form:

```math
\begin{align*}
    \textrm{minimize}_x \ & 0.5 x^T P x + q^T x + r \\
    \textrm{subject to} \ & l \leq A x \leq u
\end{align*}
```

where $x \in \mathbb{R}^n$ is the optimization variable. The objective function is defined by a positive semidefinite matrix $P \in \mathbb{S}^n_+$, a vector $q \in \mathbb{R}^n$ and a scalar $r \in R$. The linear constraints are defined by matrix $A \in \mathbb{R}^{m \times n}$ and vectors $l \in \mathbb{R}^m \cup \{-\infty\}^m$, $u \in \mathbb{R}^m \cup \{+\infty\}^m$.
