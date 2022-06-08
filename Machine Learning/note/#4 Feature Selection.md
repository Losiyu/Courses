Sequence Model
$$
\mathcal{F}_1\subset\mathcal{F}_2 \subset\mathcal{F}_n\subset\ldots\subset\mathcal{F}\\
\begin{align*}
&\mathcal{F}: Linear Function using all feature\\
&\mathcal{F}_d: Linear function using less than d feature
\end{align*}
$$
Greedy Selection: (Forward)

1. For feature i not in S (features set)
2. compute score of model $\alpha_i$
3. if it improve the current best score $\arg\max_i\alpha_i$
4. $S \leftarrow S \cup j$

Complexity Penalty
$$
score(S)=train\_loss(S) + \lambda|S|\\
S: \mathsf{set\,of\,features}
$$

General approach to feature selection

-  Define a score that balances training error and complexity
- Find the subset of features that maximizes the score
