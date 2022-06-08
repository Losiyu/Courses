## What is Regularization

1. Keep only the relevant features
2. Reduce the model's dependency on each feature as much as possible

## Method 1:  Number of Feature

Penalized ERM
$$
ERM = \min_{f \in \mathcal{F} } \frac{1}{n} \sum^n_{i=1} \ell(f(x_i), y_i) + \lambda\Omega(f)\\
\Omega: \mathsf{number\,of\,features\,in\,}f(x)\\
\lambda: \mathsf{determined\,by\,validation\,set}
$$
However, number of features as complexity measure is hard to optimize because ==what==

==越低越好，没了最好？==

## Method 2: L2 -norm (Ridge)

$$
ERM = \min_{w \in R^d } \frac{1}{n} \sum^n_{i=1} \ell(f(x_i), y_i) + \lambda||w||^2_2\\
||w||^2_2 = w_1^2+\cdots+w_d^2
$$

==希普思连续是为了证明什么 14/41==

==Lagrangian duality theory推导Tikhonov到Ivanov Form和希普思连续什么关系 22/41==

## Method 3: L1-norm(Lasso)

$$
ERM = \min_{w \in R^d } \frac{1}{n} \sum^n_{i=1} \ell(f(x_i), y_i) + \lambda||w||_1\\
||w||_1 = w_1+\cdots+w_d
$$

## Ridge in Loss and Gradient

Loss:
$$
\ell(w) = \frac{1}{2}||Xw-y||^2_2+\frac{\lambda}{2}||w||^2_2\\
\frac{1}{2}: \mathsf{Easy\,to\,calculate\,gradient}
$$
Gradient:
$$
\nabla\ell(w)=X^T(Xw-y)+\lambda w\\
这里2xwy
$$
Close form Solution
$$
w = (X^TX+\lambda I)^{-1} X^Ty\\
I:\mathsf{To\,get\,a\,matrix\,to\,work\,with\,X^{T}X\,,\lambda\,is\,just\,a\,scalar}
$$

## Ivanov vs Tikhonov

> Previous 3 method are written by Tikhonov Form, but Ivanov form can get the same $f^*$. We can use whichever is convenient

number of feature:
$$
ERM = \min_{f \in \mathcal{F} } \frac{1}{n} \sum^n_{i=1} \ell(f(x_i), y_i)\\
\Omega(f)\leq r\\
$$
l2-norm Ridge
$$
ERM = \min_{||w||^2_2\leq r^2} \frac{1}{n} \sum^n_{i=1} \ell(f(x_i), y_i)
$$
l1-norm Lasso
$$
ERM = \min_{||w||_1\leq r} \frac{1}{n} \sum^n_{i=1} \ell(f(x_i), y_i)
$$


## Why lasso yields sparse weights?

![51873729545_b401ca0d20_z](https://raw.githubusercontent.com/Losiyu/image-bed/master/51873729545_b401ca0d20_z.jpg?token=ANF6TD23TGG4P7CYZ3HNSXTCBMXP4)
$$
xAxis:
&r=0,||w||_t/||w||_t = 0\\
&r=\infty,||w||_t/||w||_t = 1\\
yAxis: &Weight
$$
**Proof base on shape:**

$|w|_1+|w|_2=r$ tends to touch the corner, corner means a weight is 0

![51873832243_a0cc295c24_z](https://raw.githubusercontent.com/Losiyu/image-bed/master/51873832243_a0cc295c24_z.jpg?token=ANF6TD4IR3NU6A6UYXUJRRLCBMXSC)

![](https://raw.githubusercontent.com/Losiyu/image-bed/master/51874082159_de4f6107b5_m.jpg?token=ANF6TD3MHBAKZF2M3FWOUJTCBMXSC)
$$
xAxis: & w_1\\
yAxis: & w_2\\
Contour: & EMR
$$
**Proof base on equation:**

if $w_1$ = 0.01 gradient of L2 penalty is $\lambda |0.0001|$, which is less helpful, but L1 could preserver the penalty 

## Conclusion about L1 and L2

Recap what is regularization:

1. Keep only the relevant features

   E.g., L1 good at limit the amount of features (feature selection), use at beginning

2. Reduce the model's dependency on each feature as much as possible

   E.g., L2 good at limit the size of each features (reduce weight)
