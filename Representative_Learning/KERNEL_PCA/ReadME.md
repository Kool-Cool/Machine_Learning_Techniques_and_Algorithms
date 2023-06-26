# Kernel PCA
This file explains the issues with Principal Component Analysis (PCA) and why we need Kernel PCA. It also includes the mathematics behind it and how to implement it using numpy only and using a library.

## Issues with PCA
PCA is a dimensionality reduction technique that transforms a set of correlated features into a set of uncorrelated components. PCA assumes that the data is linearly separable, meaning that the components can be expressed as a linear combination of the original features. However, this assumption may not hold for some real-world data sets that have nonlinear patterns or structures. In such cases, PCA may fail to capture the most important variations in the data and produce components that are not meaningful or useful.

## Why Kernel PCA
Kernel PCA is an extension of PCA that can handle nonlinear data. Kernel PCA applies a nonlinear function, called a kernel, to map the original features into a higher-dimensional space where they become linearly separable. Then, it applies PCA in the new space to obtain the components. Kernel PCA can capture the nonlinear variations in the data and produce components that are more representative of the underlying structure.

# Kernel PCA

Kernel PCA is a technique for dimensionality reduction that can handle nonlinear data. It is based on the idea of applying a kernel function to the data before performing PCA, which effectively maps the data to a higher-dimensional feature space where it becomes more linearly separable.

The mathematics behind kernel PCA can be derived from the standard PCA. Recall that PCA aims to find the directions of maximum variance in the data, which are given by the eigenvectors of the covariance matrix. The covariance matrix can be written as:

$$
\Sigma = \frac{1}{n} X^T X
$$

where $X$ is the data matrix with $n$ rows (samples) and $d$ columns (features). The eigenvectors of $\Sigma$ can be obtained by solving:

$$
\Sigma v = \lambda v
$$

where $v$ is an eigenvector and $\lambda$ is the corresponding eigenvalue. Multiplying both sides by $X$, we get:

$$
X \Sigma v = X \lambda v
$$

which can be rewritten as:

$$
\frac{1}{n} XX^T (Xv) = \lambda (Xv)
$$

This shows that $Xv$ is an eigenvector of $XX^T$, which is an $n \times n$ matrix known as the kernel matrix or Gram matrix. The kernel matrix contains the pairwise dot products of the data points, i.e.,

$$
K_{ij} = x_i^T x_j
$$

where $x_i$ and $x_j$ are the $i$-th and $j$-th rows of $X$, respectively. The kernel matrix can be seen as a measure of similarity between the data points.

Now, suppose we apply a nonlinear transformation $\phi$ to each data point, mapping it to a higher-dimensional feature space:

$$
\phi: x_i \mapsto \phi(x_i)
$$

The kernel matrix in the feature space becomes:

$$
K_{ij} = \phi(x_i)^T \phi(x_j)
$$

However, computing $\phi(x_i)$ explicitly may be computationally expensive or even impossible, depending on the dimensionality of the feature space. This is where the kernel trick comes in handy. The kernel trick is a way of computing the dot product in the feature space without ever knowing $\phi$. It relies on a kernel function that satisfies:

$$
k(x_i, x_j) = \phi(x_i)^T \phi(x_j)
$$

for any pair of data points $x_i$ and $x_j$. A common example of such a kernel function is the radial basis function (RBF) kernel:

$$
k(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)
$$

where $\gamma$ is a parameter that controls the width of the kernel.

Using the kernel trick, we can perform PCA in the feature space by computing the eigenvectors and eigenvalues of the kernel matrix:

$$
K v = n \lambda v
$$

Note that we multiplied both sides by $n$ to make it consistent with the standard PCA formulation. The eigenvectors of $K$ are called the alphas, and they are related to the eigenvectors of $\Sigma$ by:

$$
v = \frac{1}{\sqrt{n \lambda}} X^T \alpha
$$

To project a new data point $x$ onto the principal components, we need to compute its dot product with each eigenvector $v$. Using the kernel trick, this becomes:

$$
x^T v = \frac{1}{\sqrt{n \lambda}} x^T X^T \alpha = \frac{1}{\sqrt{n \lambda}} \sum_{i=1}^n k(x, x_i) \alpha_i
$$

This shows that we only need to store the alphas and the kernel values to perform kernel PCA.



## Implementation using sklearn 
To implement kernel PCA in Python using sklearn, we can use the `KernelPCA` class from the `sklearn.decomposition` module. Here is an example of how to use it:
```python

# Import modules
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# Generate some nonlinear data
X, y = make_moons(n_samples=100, random_state=123)

# Create a kernel PCA object with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)

# Fit and transform the data
X_kpca = kpca.fit_transform(X)

# Plot the results
import matplotlib.pyplot as plt
plt.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```