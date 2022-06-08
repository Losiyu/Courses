## Bagging: Bootstrap AGGregatING

> A ensemble method
>
> Ensemble method: combine multiple weak model to one powerful model

#### Pre-knowledge: Bias vs Variance

> DT has a problem of High Variance

<img src="https://nvsyashwanth.github.io/machinelearningmaster/assets/images/bias_variance.jpg" alt="Bias-Variance | Machine Learning Master" style="zoom: 33%;" />
$$
Bias:  \mathbb{E}[\hat{\theta}] - \theta \\
Var: \mathbb{E}[\hat{\theta}^2] - \mathbb{E}^2[\hat{\theta}]
$$
Bias is worse than Var, after solving Bias, we consider about variance

#### Sampling(Create multiple model)

drawn with **replacement** from $D_n=(x_1,..., x_n)$ to

1. Simulate the $D_n$ 
2. Reduce variance without changing the bias

A sample will never be sampled has a rate
$$
(1-\frac{1}{n})^n = \frac{1}{e} = 0.368
$$
which means:

We can expect 0.632 of data will use as training data, the rest of them (out of bagging) can be used for test the predictor

After sampling, we can get $D^1_n, D^2_n, ... , D^B_n$ from original data D

#### Aggregation

$\hat{f}_1, \hat{f}_2, ... , \hat{f}_B$ are the prediction function result form $D^1_n, D^2_n, ... , D^B_n$

**Bagged Prediction function**:
$$
\hat{f}_{mode}=mode(\hat{f}_1, \hat{f}_2, ... , \hat{f}_B)
$$

#### Pres & Cons

Pres: sampling reduce variation, increase the number of does not lead to overfit

Cons: Combine of multiple trees is not interpretable

#### ==Out of Bag==

For ith training point, let 

$S_i$ = {b | $D^b$ does not contain ith point}

OOB prediction:
$$
\hat{f}_{OOB}(x_i) = \frac{1}{|S_{x_i}|}\sum_{b\in S_i}\hat{f}_b(x_i)
$$

## Random Forest

#### Apply Bagging when splitting the root

![image-20220407034837001](C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220407034837001.png)

Sampling B samples with size of n from $D_n$

### Random

Compare with bagging, RF construct each tree node, restrict choice of splitting variable to a randomly when choose **subset of features**

How many feature to select:
$$
m = \sqrt{p}
$$
