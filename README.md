# Computational Studies for Master's Thesis

This repository contains the code used for the computational studies conducted as part of the Master's thesis titled **"Strong Consistency of Iterative Least Square Regression Estimates"**. These studies aim to empirically validate theoretical results from the thesis, particularly focusing on the behavior of the piecewise polynomial partitioning estimate for random variables \(X\) and \(Y\).

## Project Overview

In this study, we evaluate the piecewise polynomial partitioning estimator under various conditions, such as dimensionality, partitioning strategies, and optimization methods. Specifically, the study includes:

1. **Dimensionality and Partition Size**  
   We use a two-dimensional random variable \(X\) and vary the partition size from coarse to fine, observing its effect on the estimator's accuracy.

2. **Partitioning Strategies**  
   - **Deterministic Partitions**: These are fixed partitions independent of the data distribution.
   - **Data-Dependent Partitions**: These partitions adapt dynamically based on the data points observed.

3. **Choice of Basis Functions**  
   We compare the performance of two bases:
   - **Orthonormal Basis**
   - **Local Monomial Basis**

4. **Optimization Methods**  
   We analyze the impact of two gradient descent approaches on estimator accuracy:
   - **Fixed Step Size**
   - **Armijo Step Size Condition**

The results are assessed in terms of accuracy and convergence rates to support the theoretical results established in the thesis.

