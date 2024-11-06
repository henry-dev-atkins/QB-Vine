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

