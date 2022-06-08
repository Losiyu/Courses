## Boost

> reduce error rate of **high bias** estimator by assembling many estimators trained in sequence

> Also an ensemble method, and improve bagging. But deferent with vote in RF, it learn the weight from each weak model

**General Weighted Sum**
$$
D=((x_1,x_2),...,(x_n, y_n))\\
\hat{R}_n^w(f)=\frac{1}{W}\sum_{i=1}^nw_i\ell(f(x_i), y_i)\\
where W = \sum_{i=1}^nw_i
$$
**AdaBoost Sum of weight** 

AdaBoost evaluate each weak classifier model
$$
G(x)=sign[\sum_{m=1}^n\alpha_mG_m(x)]
$$
we want $\alpha_m$ to be larger when $G_m$ fits well
$$
err_m=\frac{1}{W}\sum_{i=1}^nw_i1(y_i\neq G_m(x_i))\\
err_m \in [0,1]
$$
$1(y_i\neq G_m(x_i))$ means equal to 1 if $\ y_i\neq G_m(x_i)$, The weight of classifier $G_m(x)$ is
$$
\alpha_m = ln(\frac{1-err_m}{err_m})
$$
<img src="C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220416222422014.png" alt="image-20220416222422014" style="zoom:50%;" />

Higher weighted err =>  Lower weight
$$
w_i \leftarrow w_ie^{\alpha_m} = w_i(\frac{1-err_m}{err_m})
$$

### Compare RF and AdaBoost

They are both typical ensemble method

RF generate more general model, but cannot fit 'very' well

AdaBoost learn slow, may overfit, but fit very well

