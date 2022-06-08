Prediction function
$$
\mathscr{X} \rightarrow \mathscr{A}\\
\mathsf{x} \mapsto f(\mathsf{x})
$$
Loss Function
$$
\ell: \mathscr{A} \times \mathscr{Y} \to \mathsf{R}\\
(\mathsf{a}, \mathsf{y}) \mapsto \ell(\mathsf{a}, \mathsf{y})
$$
Data generating distribution
$$
P_{\mathscr{X} \times \mathscr{Y}}
$$
Risk(expected loss of f over $ P_{\mathscr{X}\times\mathscr{Y}} $)
$$
\mathsf{R}(f)=\mathbb{E}_{\mathsf{x}\times\mathsf{y} \sim P_{\mathscr{X}\times\mathscr{Y}}}[\ell(f(\mathsf{x}),\mathsf{y})]
$$
Bayes Prediction Function (Target Function)
$$
f^*\in \arg\min\limits_{f} \mathsf{R}(f)
$$
Empirical Risk
$$
\mathsf{\hat{R}_n}(f)=\frac{1}{n}\sum^n_{i=1}\ell(f(\mathsf{x_i}), \mathsf{y_i})
$$
Relation between Empirical Risk and Risk
$$
\mathsf{R}(f)=\lim_{n \rightarrow \infty}\mathsf{\hat{R}_n}(f)
$$
Empirical Risk Minimizer in $\mathscr{F}$
$$
\hat{f}_n \in \arg\min_{f\in\mathcal{F}} \mathsf{\hat{R}_n}(f)\\
$$
$\mathscr{F}:$ Space defined for reduce overfitting risk

Approximation Error
$$
\mathsf{R(f_\mathscr{F}) - R({f}^*)}
$$
Estimation Error
$$
\mathsf{R({\hat{f}_n}) - R(f_\mathscr{F})}
$$
Optimization Error(Actual risk)
$$
\mathsf{R({\tilde{f}_n}) - R({\hat{f}_n})}
$$
Excess Risk
$$
\underbrace{\mathsf{R({\tilde{f}_n}) - R({\hat{f}_n})}}_{Optimization Error}
+
\underbrace{\mathsf{R({\hat{f}_n}) - R(f_\mathscr{F})}}_{Estimation Error}
 + 
\underbrace{\mathsf{R(f_\mathscr{F}) - R({f}^*)}}_{Approximation Error}
=
\mathsf{R({\tilde{f}_n}) - R({f}^*)}
$$
![ML](https://raw.githubusercontent.com/Losiyu/image-bed/master/ML.png?token=ANF6TD3VPGJXT7PLO4XKEG3CBMXUS)

For example

$f^*: y = e^x$ : Bayes, how data generate. We cannot get this because it is **fact(unlimited data)**

> Approximate error

$f_\mathcal{F}: y = w_1^2x + w_2x + w_3$: This is the best function we can get in our hypo space. we cannot get this because it is base on hypo space and **fact(unlimited data)**

> Estimation Error

$\hat{f}_n: w_4x + w_5$: This is the best function we can get base on our data(sampling(1... n)). Although it is base on the data we have, we cannot get this because we may choose a wrong function or hyperparameter

> Optimization Error, can be negative

$\widetilde{f}: w_6x+w_7$:  This is the best function we get

**Why Optimization Error can be negative**

$\hat{R}$: is empirical risk base on data, but $R$ is expectation base on fact
