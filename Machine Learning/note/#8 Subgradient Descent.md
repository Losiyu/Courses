### Subgradient is solving this problem

$$
\ell'(m)=
\left\{
\begin{array}{ll}
0 &m>1\\
-1 &m<1\\
undefine &m=1
\end{array}
\right.
$$

$$
m = y_iw^Tx_i
$$

$$
\begin{array}{ll}
\nabla_wJ(w) & =\nabla_w(\frac{1}{n}\sum\ell(y_iw^\intercal x_i)+\lambda||w||^2)\\
& = \frac{1}{n}\sum\nabla_w\ell(y_iw^\intercal x_i)+2\lambda w)\\
& = 
    \left\{
        \begin{array}{ll}
            \frac{1}{n}\sum-y_ix_i+2\lambda w & y_iw^\intercal x_i<1\\
            undefine & otherwise
        \end{array}
    \right.
\end{array}
$$

### Convex Problem

<img src="C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220301214028563.png" alt="image-20220301214028563" style="zoom:50%;" />
$$
\frac{f(x) - f(x_0)}{x - x_0} \geq g^\intercal
$$

> $g^{\intercal} $ are not unique, it is a range

### Subgradient on SVM

$$
J(w) & =\frac{1}{n}\sum\max(0, 1-y_iw^\intercal x_i)+\lambda||w||^2\\
$$

$$
\eta_t = \frac{1}{t}\lambda\\
w_{t+1} = 
\left\{
\begin{array}{ll}
(1-\eta_t\lambda)w_t + \eta_ty_ix_i & y_iw^\intercal x_i<1 \\
(1-\eta_t\lambda)w_t & otherwise\\
\end{array}
\right.
$$

