# Gradient Boost

> **参数空间的梯度下降利用梯度信息调整参数降低损失，函数空间的梯度下降利用梯度拟合一个新的函数降低损失。**

$$
J(f) = \frac{1}{n}\sum^m_{i=1}(y_i-f(x_i))^2\\
\frac{\partial J}{\partial f(x_i)}=-2(y_i-f(x_i))\\
f \leftarrow f + vh 
$$

## Gradient Descent

$$
\begin{align}
J(f)&=\sum^n_{i=1}\ell(y_i, f(x_i))\\
f &= (f(x_1),..., f(x_n))^T\\
J(f)&=\sum^n_{i=1}\ell(y_i, f_i)\\
-g & = -\nabla_fJ(f)=-\partial_{f1}\ell(y_i, f_i),...
\end{align}
$$

### Learning Rate

$$
f \leftarrow f + \lambda vh
$$

$\lambda$ can be a constant

