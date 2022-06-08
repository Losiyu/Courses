# HW3 Siyong Liu

### 1

$$
max_if_i(z) \geq f_k(z) \geq f_k(x)+g^T(z-x) = f(x)+g^T(z-x) \\
g \in \partial f(x)
$$

### 2

$$
\partial J(w)=\
    \begin{cases}\
    0 & 1-yw^Tx<0 \\
    -yx^T & otherwise
    \end{cases}
$$



### 3

$\exists a,b,\theta  \ \theta f(a)+(1-\theta)f(b) < f(\theta \alpha + (1-\theta)b)$

Let $x_0=\theta \alpha + (1-\theta)b$

$f(x_0)> \theta f(a)+(1-\theta) f(b)$

$\exists x_0, g$ $f(a)\leq f() 



### 4

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

### 5

$$
w_{t+1} = w_t - \eta(\lambda w - y_ix) \\
w_{t+1} = 
\left\{
\begin{array}{ll}
(1-\eta_t\lambda)w_t + \eta_ty_ix_i & y_iw^\intercal x_i<1 \\
(1-\eta_t\lambda)w_t & otherwise\\
\end{array}
\right.
$$

<div STYLE="page-break-after: always;"></div>

### 6

```python
# 6
def to_sparse(l):
  return Counter(l)
```

### 7

```python
# 7
def load_data():
  reviews = load_and_shuffle_data()
  train = reviews[:1500]
  test = reviews[1500:]
  X_train = [x[:-1] for x in train]
  y_train = [x[-1] for x in train]
  X_test = [x[:-1] for x in test]
  y_test = [x[-1] for x in test]
  return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()
```

### 8

```python
# 8
def pegasos1(X, y, lambda_reg=0.25, max_epoch=30):
    epoch = 0
    w = {}
    t = 0
    order_list = range(len(X))
    while epoch < max_epoch:
        random.shuffle()
        for i in order_list:
            t += 1
            eta = 1 / (t * lambda_reg)
            if y[i] * dotProduct(X[i], w) < 1:
                increment(w, - eta * lambda_reg, w)
                increment(w, eta * y[i], X[i])
            else:
                increment(w, - eta * lambda_reg, w)
        epoch += 1
    return w
```

### 9

$$
\begin{array}{ll}
s_{t+1}W_{t+1} 
& = s_t(W_t+\frac{1}{s_{t+1}}\eta_ty_jx_j)(1-\eta_t\lambda)\\
& = (w_t + \frac{s_t}{s_{t+1}}\eta_ty_jx_j)(1-\eta_t\lambda)\\
& = (1-\eta_t\lambda)w_t + \frac{s_t}{s_{t+1}}(1-\eta_t\lambda)\eta_ty_jx_j\\
& = (1-\eta_t\lambda)w_t + \eta_ty_jx_j \\
& = w_{t+1}
\end{array}
$$

```python
# 9
def pegasos2(X, y, lambda_reg=0.1, max_epoch=30, tolerance=1e-2, useConverge=True):
    epoch = 0
    w = {}
    t = 1
    scale = 1
    order_list = range(len(X))
    while epoch < max_epoch:
        epoch += 1
        prev_sum = sum(w[weight]**2 for weight in w)
        for i in order_list:
            t += 1
            eta = 1 / (t * lambda_reg)
            scale = (1 - eta * lambda_reg) * scale
            if y[i] * scale * dotProduct(w, X[i]) < 1:
                increment(w, eta * y[i] / scale, X[i])
        cur_sum = sum(w[weight]**2 for weight in w)
        if useConverge and np.abs(scale**2 * (prev_sum - cur_sum)) < tolerance:
            break
    for k, v in w.items():
        w[k] = v * scale
    return w
```

### 10

```python
%%time
w1 = pegasos1(X_train, y_train)
>>>
Wall time: 6min 43s
```

```python
%%time
w2 = pegasos2(X_train, y_train, useConverge=False)
>>>
Wall time: 5.41 s
```

```
print("w1['friends']: ", w1['friends'])
print("w2['friends']: ", w2['friends'])
>>>
w1['friends']:  0.020533333333333324
w2['friends']:  0.017555165440767868
```

### 11

```python
#11 classification error
def classification_error(w, X, y):
  cnt = 0
  for i in range(len(X)):
    if np.sign(dotProduct(X[i],w)) != y[i]:
      cnt += 1
  return cnt/len(X)

w1_err = classification_error(w1, X_test, y_test)
w2_err = classification_error(w2, X_test, y_test)
print('w1_err: ', w1_err)
print('w2_err: ', w2_err)
>>>
w1_err:  0.246
w2_err:  0.276
```

### 12

```python
def test_lambda(lambda_list, X_train, y_train, X_test, y_test):
    err_list = []
    for lambda_reg in lambda_list:
        w = pegasos2(X_train, y_train, lambda_reg)
        err_list.append(classification_error(w, X_test, y_test))
    return err_list
```

```python
lambda_list = np.linspace(0.001, 10, 100)
err_list = test_lambda(lambda_list, X_train, y_train, X_test, y_test)
plt.plot(lambda_list, err_list)
```

```python
lambda_list = np.linspace(0.001, 0.2, 10)
err_list = test_lambda(lambda_list, X_train, y_train, X_test, y_test)
plt.plot(lambda_list, err_list)
```

![12.1](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220304_011.png)

![12.2](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220304_207.png)

best lambda is around [0, 0.025]

<div STYLE="page-break-after: always;"></div>

### 15

$$
\partial_wJ(w) = 2X^T(Xw-y) + 2\lambda w &=& 0\\
X^TXw+\lambda Iw &=& X^Ty\\
(X^TX + \lambda I) w &=& X^Ty\\
w &=& (X^TX + \lambda I)^{-1}X^Ty
$$

### 16

$$
X^TXw+\lambda Iw &=& X^Ty\\
\lambda w &=& X^Ty - X^TXw\\
w &=& \frac{1}{\lambda}(X^Ty-X^TXw)\\
$$

$$
w &=& X^T \alpha\\
\alpha &=& \frac{1}{\lambda}(y-Xw)
$$

### 17

$$
w = X^T\alpha=\sum_{i=1}^m\alpha_ix_i
$$



### 18

$$
\alpha &=& \frac{1}{\lambda}(y-Xw)\\
\alpha &=& \frac{1}{\lambda}(y-XX^T\alpha)\\
\lambda \alpha &=& (y-XX^T\alpha) \\
\lambda \alpha + X^TX \alpha &=& y\\
\alpha &=& (\lambda I + X^TX)^{-1}y\\
$$

### 19

$$
Xw = XX^T\alpha = X^TX(\lambda I + X^TX)^{-1}y
$$

### 20

$$
f(x) = x^Tw^* = x^TX^T\alpha = k_x^T\alpha
$$

### 21

```python
def linear_kernel(X1, X2):
    return np.dot(X1,np.transpose(X2))
 
def RBF_kernel(X1,X2,sigma):
    distance = scipy.spatial.distance.cdist(X1,X2,'sqeuclidean')
    return np.exp(- 0.5 * distance / pow(sigma, 2))

def polynomial_kernel(X1, X2, offset, degree):
    return pow((offset + np.inner(X1, X2)), degree)
```

### 22

```python
array = np.array([[-4, -1, 0, 2]]).T
linear_kernel(array, array)
>>>
array([[16,  4,  0, -8],
       [ 4,  1,  0, -2],
       [ 0,  0,  0,  0],
       [-8, -2,  0,  4]])
```

### 23

```python
# polynomial
y = polynomial_kernel(prototypes, xpts, 1, 3) 
for i in range(len(prototypes)):
    label = "polynomial@"+str(prototypes[i,:])
    plt.plot(xpts, y[i,:], label=label)
plt.legend(loc = 'best')
plt.show()
```

![23.1](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220306_199.png)

```python
y = RBF_kernel(prototypes, xpts, 1)
for i in range(len(prototypes)):
    label = "RBF@"+str(prototypes[i,:])
    plt.plot(xpts, y[i,:], label=label)
plt.legend(loc = 'best')
plt.show()
```

![23.2](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220306_679.png)

### 24

```python
def predict(self, X):
	return self.kernel(X, self.training_points) @ self.weights
```

```python
rbf_kernel = functools.partial(RBF_kernel, sigma=1)
xpts = np.arange(-6.0, 6, plot_step).reshape(-1,1)
k_machine = Kernel_Machine(rbf_kernel,
    np.array([-1, 0, 1]).reshape(-1, 1),
    np.array([1, -1, 1]).reshape(-1, 1))
plt.plot(xpts, k_machine.predict(xpts))
plt.show()
```

![SS220306_199](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220306_067.png)

### 25

```python
plt.scatter(x_train, y_train)
plt.plot()
```

![25](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220306_615.png)

### 26

```python
def train_kernel_ridge_regression(X, y, kernel, l2reg):
    alpha = np.linalg.inv(np.identity(X.shape[0])*l2reg + kernel(X, X)) @ y
    return Kernel_Machine(kernel, X, alpha)	
```

### 27

![26](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220306_745.png)

overfit: 0.01

underfit: 1

best: 0.1

### 28

![27](https://raw.githubusercontent.com/Losiyu/image-bed/master/SS220306_341.png)

when $\lambda \rightarrow \infty$ , model become underfit

