# QB-Vine
This repository contains the code for "Quasi-Bayes meets Vines" ([preprint](https://arxiv.org/abs/2406.12764)), accepted at NeurIPS 2024.


# Installation

A requirements.txt is provided for pip python installation.
```commandline
pip install -r requirements.txt
```

To install MPI on Windows, follow this [link](https://www.microsoft.com/en-us/download/details.aspx?id=105289). This can be tested by running the test_mpi.py file.


# Example Usage
As MPI is in usage, code must be ran in the following way: < mpiexec -n \<N> python \<NAME/py> >. This ensures that the correct number of threads are used in teh runtime. For example, running the example.py file with 2 threads would look like:
```
mpiexec -n 2 python example.py
```


# Code Structure
### Functions and Their Corresponding Equations

#### `AllDimRecursion.py`
- **Class: `Method`**  
  An abstract base class for inference methods. This serves as a framework for implementing specific recursive Bayesian methods.  
  *No direct equation reference.*

---

#### `GaussianCop.py`
- **`inverse_std_normal(cumulative_prob)`**  
  Computes the inverse of the standard normal CDF. This is crucial for Gaussian KDE on copulas, aligning with the latent space transformations described in the paper.  
  **Related to Section A.3, Copula Transformations**.

---

#### `OneRecursion.py`
- **Class: `MarRecur`**  
  Implements the univariate recursive Bayesian predictive updates, based on the Recursive Bayesian Predictive (R-BP) method.  
  **Related Equations**:  
  - **Equation (2)**: Recursive density updates using copulas.  
  - **Equation (3)**: Recursive updates for CDFs using Gaussian copulas.  

---

#### `utils.py`
- **`cdf_lomax(x, a)`**  
  Computes the CDF of the Lomax distribution, often used in empirical copula constructions.  
  *No direct equation reference.*  

- **`pdf_lomax(x, a)`**  
  Computes the PDF of the Lomax distribution.  
  *No direct equation reference.*  

- **`alpha(step)`**  
  Computes the weight parameter \(\alpha_n\) used in recursive updates. This aligns with the weight sequence in **Equation (2)** of the paper.  


