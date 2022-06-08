Gradient Descent(GD)
$$
initial\, x\leftarrow 0\\
Repeat\, x\leftarrow x - \eta \nabla f(x)\\
Until\ the\ stopping\ criterion\ is\ satisfied
$$
Two method for GD

![SS220215_642](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220215_642.png)

### Fix Step

**Lipschitz Continue**
$$
||\nabla f(x)-\nabla f(x') \leq L||x-x'||\\
||f(x^K)-f(f^*)|| \leq \frac{||x^0-x^*||^2}{2\eta k}
$$
Then Gradient with $\eta \leq \frac{1}{L}$ converge, This say it guaranteed to converge with rate $O(\frac{1}{k})$

**Early Stop**
$$
||\nabla f(x)||_2 \leq \varepsilon
$$

- Evaluate loss on validation data after each iteration
- Stop when the loss does not improve (or gets worse)

**Big or small Step Size?**

Small enough size guarantee to converge but cost too much time

Big step size may cause cross over the best result, it will cause diverge 

**Cons?**

We have to iterate over all n training points to take a single step. [O(n)]
Will not scale to “big data”!

### How to solve big data issue

**GD** 

Use full data set(batch = n) to determine step

**Minibatch** 

use **random** subset of size N for each epoch

**Stochastic Gradient Descent(SGD)(online learning)**

use batch size 1**(More efficient)**



