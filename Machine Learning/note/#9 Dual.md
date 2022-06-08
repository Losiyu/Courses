### SVM Objective Function

$$
\xi = max(0, 1-yw^\intercal x)\\
\frac{1}{2}||w||^2+\frac{c}{n}\sum^n_{i=1}\xi_i\\
$$

constraint:
$$
& -\xi \le 0 \\
&(1-y_i[w^\intercal x_i+b])-\xi_i\le0
$$


### Use Lagrange dual remove constraint

> One purpose of *Lagrange duality* is to find a lower bound on a minimization problem

Lagrange multiplier:
$$
\lambda_i:& -\xi \le 0 \\
\alpha_i: &(1-y_i[w^\intercal x_i+b])-\xi_i\le0
$$
==Lagrange dual function is the inf over primal variables of L:==

