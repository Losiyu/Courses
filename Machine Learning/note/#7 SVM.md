#### Geometric Margin

<img src="C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220224082836779.png" alt="image-20220224082836779" style="zoom:50%;" />

#### Maximize the Margin

Get distance between point($x_i$) and hyperplane($H:w^Tx_i+b$)

<img src="C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220224083511851.png" alt="image-20220224083511851" style="zoom:50%;" />
$$
d(x_i, H) 
= |v*\frac{w}{||w||_2}|
= |\frac{w^T(x_i-x_0)}{||w||_2}|
= |\frac{w^Tx_i+b}{||w||_2}|
$$

>  Because $w^Tx_i+b<0$ when point are under the hyperplane, so we

$$
y_i=
\left \{
\begin{array}{ll}
1  & w^Tx_i + b>0 \\
-1 & w^Tx_i + b<0\\
\end{array}
\right.
$$

$$
d(x_i,H) = y_i*\frac{w^Tx_i+b}{||w||_2}
$$

Minimize the distance, use $\varepsilon$ evaluate the loss and

<img src="C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220224094115980.png" alt="image-20220224094115980" style="zoom:50%;" />
$$
y_i(w^Tx_i+b)>1-\varepsilon_i
$$
$\varepsilon = 0, x_i $ lie on margin

$\varepsilon = 1, x_i $ lie on hyperplane

$\varepsilon > 2\ or\ \varepsilon < 0, x_i$ out of margin

#### Minimize Hinge Loss

<img src="https://pic3.zhimg.com/50/v2-3c6aa9626ee8e4609b0d7c5712baf624_720w.jpg?source=1940ef5c" alt="怎么样理解SVM中的hinge-loss？ - 知乎" style="zoom:50%;" />
$$
\ell_{Hinge} = max(0, 1-m)\\
\ell_{Hinge}'=
\left \{
\begin{array}{ll}
0 & m > 1\\
-1 & m < 1\\
undefine & m = 1
\end{array}
\right.
$$
