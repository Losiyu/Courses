## Decision Tree 

![image-20220406122033370](C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220406122033370.png)

- Each node contains a subset of data

- The data splits created by each node involved only **one feature**

- For continuous var, the splits are always the form of $x_i\leq t$, **so no slash allowed**
- Prediction made by the lowest node

### Avoid overfitting when keep splitting

Method 1: limited split time

Method 2: limit node

**Method 3: Backward Pruning**

- Build a big tree with each region has <= 5 points, 
  now valid error is big because overfitting 

- Pruning until validation error increase (from back to the start)

### How to Evaluate the split: Node Impurity Measures

>  Makes the point of each region purer

$\hat{p}_{mk}$: Best proportion($\hat{p}$) in mode m for class k

**Method 1: misclassification**

> find out the majority class

$\hat{p}_{mk(m)}$: Best proportion($\hat{p}$) in mode m for **majority** class k, k(m) = $\arg\max_k\hat{p}_{mk}$

Error: $1- \hat{p}_{mk(m)}$ 

**Method 2: Gini index**

> Do not need find majority class, encourage to 0 and 1

Error: $1-\sum_{k=1}^{K}\hat{p}_{mk}(1-\hat{p}_{mk})$

**Method 3: Entropy**

Error: $-\sum_{k=1}^{K}\hat{p}_{mk}log\hat{p}_{mk}$

<img src="C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220406133951332.png" alt="image-20220406133951332" style="zoom:50%;" />

**Gini vs Entropy**

<img src="C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220406135010282.png" alt="image-20220406135010282" style="zoom:50%;" />

- Range: Entropy [0, 1], Gini [0, 0.5]

- Entropy is much lower, because it used log
- Entropy get result slightly better

### Tree vs DT

> Depend on data

<img src="C:\Users\siyon\AppData\Roaming\Typora\typora-user-images\image-20220406135553304.png" alt="image-20220406135553304" style="zoom:50%;" />

### Feature of DT

- Non-Linear

- Non-metric: ==这是什么==
- Non-parametric:
- Easy to understand and visualize

But:

- Perform bad in linear model
- Have high variance and tend to overfit (sensitive to small changes in training data)

