# Layer-Wise Histograms for Hierarchical Nested-Lattice MatVec

This note shows how to compute **\(y = Wx\)** efficiently when each column of \(W\) is stored with a **hierarchical (depth-\(M\)) lattice code**, using **layer-wise histograms**. It includes a concrete, worked example and explains why the computation is (also) an inner product.

---

## Problem Setup

We want
\[
y \;=\; W x \;=\; \sum_{j=1}^d x_j\,w_j \in \mathbb{R}^n,
\]
but each column \(w_j\) is stored in a **layered** (refinement) form
\[
\widehat w_j \;=\; \sum_{m=0}^{M_j-1} q^m\,\lambda\!\big(b_{j,m}\big),
\]
where:

- \(q\ge 2\) is the nesting base, \(M_j\) is the (possibly column-dependent) number of decoded layers,
- \(\lambda(k)\in\mathbb{R}^n\) is the codeword for index \(k\in\mathcal K\),
- \(b_{j,m}\in\mathcal K\) is column \(j\)’s code index at layer \(m\).

Then
\[
y \;=\; \sum_{j=1}^d x_j \sum_{m=0}^{M_j-1} q^m \lambda(b_{j,m})
\;=\; \sum_{m=0}^{M-1} q^{m}\!\!\sum_{\substack{j:\ m<M_j}} x_j\,\lambda(b_{j,m}).
\]

Define the **layer-wise histogram (pooled weights)**:
\[
\boxed{\quad s_{m,k} \;\triangleq\; \sum_{\substack{j:\ m<M_j\\ b_{j,m}=k}} x_j,\qquad k\in\mathcal K \quad}
\]
so that
\[
\boxed{\quad y \;=\; \sum_{m=0}^{M-1} q^{m} \sum_{k\in\mathcal K} s_{m,k}\,\lambda(k).\quad}
\]

---

## Tiny Worked Example

**Parameters**

- Output dim: \(n=4\)  
- Columns: \(d=5\)  
- Base: \(q=3\)  
- Depth: \(M=3\) (layers \(m=0,1,2\))  
- Codebook \(\{\lambda(k)\}_{k=0}^2 \subset \mathbb{R}^4\):
  \[
  \lambda(0)=\begin{bmatrix}1\\0\\0\\0\end{bmatrix},\quad
  \lambda(1)=\begin{bmatrix}0\\1\\0\\0\end{bmatrix},\quad
  \lambda(2)=\begin{bmatrix}1\\1\\1\\0\end{bmatrix}.
  \]

**Layer indices \(b_{j,m}\)**

| col \(j\) | \(b_{j,0}\) | \(b_{j,1}\) | \(b_{j,2}\) |
|---:|---:|---:|---:|
| 1 | 0 | 2 | 1 |
| 2 | 1 | 0 | 1 |
| 3 | 2 | 2 | 2 |
| 4 | 0 | 1 | — |
| 5 | 1 | 1 | 0 |

**Query** and **per-column truncation depths**

\[
x = (0.7,\,-1.2,\,0.0,\,0.5,\,2.0),\qquad
(M_1,M_2,M_3,M_4,M_5)=(3,\,2,\,1,\,2,\,3).
\]
Note \(x_3=0\), so column 3 contributes nothing (regardless of its indices).

---

### Step 1 — Build \(s_{m,k}\)

By definition:
\[
s_{m,k}=\sum_{\substack{j:\ m<M_j\\ b_{j,m}=k}} x_j,\quad m\in\{0,1,2\},\; k\in\{0,1,2\}.
\]

**Layer \(m=0\)** (include \(j\) with \(M_j>0\): \(j=1,2,3,4,5\))

- Assignments: \(b_{1,0}=0\), \(b_{2,0}=1\), \(b_{3,0}=2\), \(b_{4,0}=0\), \(b_{5,0}=1\).
- Sums:
  \[
  s_{0,0}=x_1+x_4=1.2,\quad
  s_{0,1}=x_2+x_5=0.8,\quad
  s_{0,2}=x_3=0.
  \]
- Result: \(s_{0,\cdot}=(1.2,\ 0.8,\ 0.0)\).

**Layer \(m=1\)** (include \(j\) with \(M_j>1\): \(j=1,2,4,5\))

- Assignments: \(b_{1,1}=2\), \(b_{2,1}=0\), \(b_{4,1}=1\), \(b_{5,1}=1\).
- Sums:
  \[
  s_{1,0}=x_2=-1.2,\quad
  s_{1,1}=x_4+x_5=2.5,\quad
  s_{1,2}=x_1=0.7.
  \]
- Result: \(s_{1,\cdot}=(-1.2,\ 2.5,\ 0.7)\).

**Layer \(m=2\)** (include \(j\) with \(M_j>2\): \(j=1,5\))

- Assignments: \(b_{1,2}=1\), \(b_{5,2}=0\).
- Sums:
  \[
  s_{2,0}=x_5=2.0,\quad
  s_{2,1}=x_1=0.7,\quad
  s_{2,2}=0.
  \]
- Result: \(s_{2,\cdot}=(2.0,\ 0.7,\ 0.0)\).

**Summary of histograms**
\[
s_{0,\cdot}=(1.2,\,0.8,\,0.0),\quad
s_{1,\cdot}=(-1.2,\,2.5,\,0.7),\quad
s_{2,\cdot}=(2.0,\,0.7,\,0.0).
\]

---

### Step 2 — Accumulate layer contributions

\[
y \;=\; \sum_{m=0}^{2} q^{m}\sum_{k=0}^{2} s_{m,k}\,\lambda(k),\qquad q=3.
\]

- **Layer 0** (\(q^0=1\)):
  \[
  \sum_k s_{0,k}\lambda(k) \;=\;
  1.2\,\lambda(0)+0.8\,\lambda(1)+0\cdot\lambda(2)
  \;=\; \begin{bmatrix}1.2\\0.8\\0\\0\end{bmatrix}.
  \]
- **Layer 1** (\(q^1=3\)):
  \[
  \sum_k s_{1,k}\lambda(k)=(-1.2)\lambda(0)+2.5\lambda(1)+0.7\lambda(2)
  = \begin{bmatrix}-0.5\\3.2\\0.7\\0\end{bmatrix},
  \]
  then multiply by \(3\): \(\begin{bmatrix}-1.5\\9.6\\2.1\\0\end{bmatrix}\).
- **Layer 2** (\(q^2=9\)):
  \[
  \sum_k s_{2,k}\lambda(k)=2.0\,\lambda(0)+0.7\,\lambda(1)+0\cdot\lambda(2)
  = \begin{bmatrix}2.0\\0.7\\0\\0\end{bmatrix},
  \]
  then multiply by \(9\): \(\begin{bmatrix}18.0\\6.3\\0\\0\end{bmatrix}\).

**Final**
\[
y \;=\;
\begin{bmatrix}1.2\\0.8\\0\\0\end{bmatrix}+
\begin{bmatrix}-1.5\\9.6\\2.1\\0\end{bmatrix}+
\begin{bmatrix}18.0\\6.3\\0\\0\end{bmatrix}
=\boxed{\begin{bmatrix}17.7\\16.7\\2.1\\0\end{bmatrix}}.
\]

---

## Sanity Check (Equivalence)

Each (truncated) column
\[
\widehat w_j \;=\; \sum_{m=0}^{M_j-1} q^m\,\lambda\!\big(b_{j,m}\big)
\]
produces the same result if you compute \(\sum_j x_j\,\widehat w_j\) directly.  
The histogram method just **pools identical \(\lambda(\cdot)\) once per layer**, which is why it’s faster when many columns share code indices.

---

## Inner-Product Interpretation

There are (at least) two views:

1. **Row-wise classical view:**  
   \[
   y_i = \sum_{j=1}^d w_{ij}\,x_j = \langle W_{i,:},\,x\rangle.
   \]

2. **Feature-space view using pooled histograms:**  
   Stack all \(s_{m,k}\) into
   \[
   s \;=\; \big(s_{0,0},\ldots,s_{0,K-1};\; s_{1,0},\ldots,s_{1,K-1};\; \ldots;\; s_{M-1,0},\ldots,s_{M-1,K-1}\big)\in\mathbb{R}^{MK}.
   \]
   For each output coordinate \(i\), define the fixed vector
   \[
   d_i \;=\; \big(\lambda_i(0),\ldots,\lambda_i(K-1);\; q\,\lambda_i(0),\ldots,q\,\lambda_i(K-1);\; \ldots;\; q^{M-1}\lambda_i(0),\ldots,q^{M-1}\lambda_i(K-1)\big).
   \]
   Then
   \[
   y_i \;=\; \langle d_i,\; s\rangle.
   \]
   So computing \(y\) equals evaluating \(n\) inner products between **fixed** feature vectors \(d_i\) and the **query-dependent** pooled weights \(s\).

---

## Pseudocode (Layer-Wise Histogram MatVec)

```python
# Inputs:
#   x            : (d,) weights
#   M            : max depth
#   M_j          : (d,) per-column truncation depths in {0..M}
#   b[j][m]      : code index of column j at layer m  (in {0..K-1})
#   lambda_k[k]  : (n,) codeword vector for index k
#   q            : base
# Output: y ≈ W x using only the layers allowed by M_j

import numpy as np

def matvec_hier_columnwise(x, M, M_j, b, lambda_k, q):
    n = lambda_k[0].shape[0]
    K = len(lambda_k)
    y = np.zeros(n, dtype=float)

    for m in range(M):
        # pooled weights (histogram) for layer m
        s = np.zeros(K, dtype=float)
        for j, xj in enumerate(x):
            if xj == 0.0:         # skip zero coefficients
                continue
            if m >= M_j[j]:       # column not decoded at this layer
                continue
            s[b[j][m]] += xj      # reduce-by-code

        if np.any(s):
            # y += q^m * sum_k s[k] * lambda(k)
            layer = np.zeros(n, dtype=float)
            for k, sk in enumerate(s):
                if sk != 0.0:
                    layer += sk * lambda_k[k]
            y += (q**m) * layer
    return y
