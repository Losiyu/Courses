### Feature Extractor

Raw input: X

Feature vector: $\mathbb{R}^d$
$$
X \xrightarrow{x} Feature\ extraction \xrightarrow{\phi(x)}\mathbb{R}^d
$$

>  For linear models, to grow the hypothesis spaces, we must add features

### Issue 1: Non-monotonicity

**Problem:** body temperature, Both high and low are bad

[1, x]

**solution:** add more

 $[1, (x - 37)^2] \rightarrow [1, x,x^2] $

### Issue 2: Interactions

**Problem:** two feature are not independent

$\phi(x) = [weight\ w(h), height\ h(x)]$

**Solution 1:** include all second order features:

$[1, w,h,w^2, h^2, wh]$

**Solution 2:** replace with single feature

$\phi(x) = [weight\ w(h)]$

### Issue 3: Big Hypothesis spaces

**Problem:** Overfitting, computing cost

**Solution1:** regularization

**Solution2:** Kernel