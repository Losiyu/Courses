# Feature Learning(NN)

**Task** Predict popularity of restaurants. 

**Raw features** #dishes, price, wine option, zip code, #seats, size

<img src="https://raw.githubusercontent.com/Losiyu/image-bed/master/image-20220508225200089.png" alt="image-20220508225200089" style="zoom:50%;" />

based on domain knowledge, the result should be: w1 · food quality+w2 · walkable+w3 · noise

**Don't wanna learn intermediate feature manually? Try Perceptron**

<img src="https://raw.githubusercontent.com/Losiyu/image-bed/master/image-20220508225934849.png" alt="image-20220508225934849" style="zoom:50%;" />

## Perceptron

> In [machine learning](https://en.wikipedia.org/wiki/Machine_learning), the **perceptron** is an algorithm for [supervised learning](https://en.wikipedia.org/wiki/Supervised_classification) of [binary classifiers](https://en.wikipedia.org/wiki/Binary_classification). A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.

<img src="https://raw.githubusercontent.com/Losiyu/image-bed/master/image-20220508230411096.png" alt="image-20220508230411096" style="zoom: 67%;" />

### Composition

1. Objective Function

   $w_1i_1+w_2i_2$

2. Predict Function (It called activation function in hidden layer)

   $\geq \theta$

To make a perceptron, we need to learn $W$ and $\theta$

### Solve Logic Problem by using Perceptron

1. AND

   Question:

   | $i_1$ | $i_2$ | o    |
   | ----- | ----- | ---- |
   | 0     | 0     | 0    |
   | 0     | 1     | 0    |
   | 1     | 0     | 0    |
   | 1     | 1     | 1    |

   Solution:

   $w_1 = 1, w_2 = 1, \theta = 2 $

   | $i_1$ | $i_2$ | $w_1i_1+w_2i_2$ | $\geq 2$ |
   | ----- | ----- | --------------- | -------- |
   | 0     | 0     | 0               | 0        |
   | 0     | 1     | 1               | 0        |
   | 1     | 0     | 1               | 0        |
   | 1     | 1     | 2               | 1        |

   Perceptron Learn in linear way, but how about non-linear? Like XOR

2. XOR

   <img src="https://raw.githubusercontent.com/Losiyu/image-bed/master/image-20220508223310693.png" alt="image-20220508223310693" style="zoom:50%;" />

   Divide the problem and learn **'intermediate feature'**

   Question:
   
   | $i_1$ | $i_2$ | o    |
   | ----- | ----- | ---- |
   | 0     | 0     | 1    |
   | 0     | 1     | 0    |
   | 1     | 0     | 0    |
   | 1     | 1     | 1    |
   
   Solution:
   
   <img src="https://raw.githubusercontent.com/Losiyu/image-bed/master/image-20220508235740383.png" alt="image-20220508235740383" style="zoom:50%;" />
   
   | $i_1$ | $i_2$ | $i_{h1}$ | $i_{h2}$ | $o_{h1}$ | $o_{h2}$ | o    |
   | ----- | ----- | -------- | -------- | -------- | -------- | ---- |
   | 0     | 0     | 0        | 0        | 0        | 1        | 1    |
   | 0     | 1     | 1        | -1       | 0        | 0        | 0    |
   | 1     | 0     | 1        | -1       | 0        | 0        | 0    |
   | 1     | 1     | 2        | -2       | 1        | 0        | 1    |
   
   Note:
   
   $i_{h1}$: input of node h1
   
   $o_{h1}$: output of node h1



## Neural Network

> which made by multiple and multi-layer Perceptron

**Describe it by using function**

manually:
$$
\hat{F}(x) = w^T\Phi(x)
$$
Neural Network: (K unit)
$$
h(x) = [h1(x), h2(x), ..., h_K(x)]\\
f(x) = w^Th(x)
$$
**So, for 2-layer NN with K hidden unit(perceptron), describe by activation function($\sigma$)**
$$
\hat{F}(x) = \sum^K_{i=1}w_kh_k(x)=\sum^K_{i=1}w_k\sigma(v_kx)
$$
Note:

$w_k$: input of hidden layer for k's row

$v_k$: output of hidden layer for k's row

## Deep Neural Network

> deep Neural Network is not a type of NN, but a concept
>
> It has more layers, and more unit in each layers

But, why 'Deep' preform better?

### Approximation Ability

A single layer NN with huge amount of hidden unit and an activation function can represent any function.
$$
\hat{F}(x)=\sum^N_{i=1}w_i\sigma(v_ix)\\
\hat{F}(x)-F(x)< \epsilon
$$
But, this is not practical, because we need:

- exponential d hidden unit
- get parameter of network, W and b
- how to build?

## Multilayer Perceptron

$$
h^{j}(o^{j-1})=\sigma(W^{j}o^{j-1}), \ for\ j=2,...,L
$$

​																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																							
