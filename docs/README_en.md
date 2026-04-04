# Soft-Actor-Critic

**This was AI-translated, so it might not be accurate.**

This is the implementation code for the [SAC with automatic temperature adjustment](https://arxiv.org/pdf/1812.05905) version.

It utilizes the standard Twin-Q model (using the minimum of two Q-values for the Actor).

---

### reference

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

> [!NOTE]
> If you want to implement directly from the skeleton, see [this link](https://github.com/Y0rFa1se/Soft-Actor-Critic/tree/dev).


## How to Run

### Execute Training Code
```bash
uv run main.py
```

### Execute Test Code
```bash
uv run test.py
```

Parameters that have solved the continuous lunar lander ([200 points or more](https://gymnasium.farama.org/environments/box2d/lunar_lander/#rewards)) are included for verification.

![sample_gif](/imgs/lunarlander_sample.gif)

## Review

### Objective function

$$
J_Q(w_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}[\frac{1}{2}(Q_{w_i}(s, a) - \hat{Q}(s, a))^2]
$$
$$
\text{where } \hat{Q}(s, a) := r + \gamma(\min\limits_{j = 1, 2} Q_{\bar{w}_j}(s', a') - \alpha\log\pi_\theta(a'|s')),
$$
$$
\text{with } (a', \log\pi_\theta(a'|s')) \sim \pi_\theta.
$$
$$
J_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}}[\alpha\log\pi_\theta(a|s) - \min\limits_{j = 1, 2}Q_{w_j}(s, a)]
$$
$$
\text{with } a \sim \pi_\theta.
$$
$$
\text{Let } \zeta := \log\alpha.
$$
$$
J(\zeta) = \mathbb{E}_{s \sim \mathcal{D}}[-\zeta(\log\pi_\theta(a|s) + \bar{\mathcal{H}})]
$$
$$
\text{with } a \sim \pi_\theta.
$$

### Gradient

$$
\nabla_{w_i} J_Q(w_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q_{w_i}(s, a) - \hat{Q}(s, a) \right) \nabla_{w_i} Q_{w_i}(s, a) \right]
$$
$$
\nabla_\theta J_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \left( \alpha \log \pi_\theta(a|s) - \min_{j=1,2} Q_{w_j}(s, a) \right) + \nabla_\theta (\alpha \log \pi_\theta(a|s)) \right]
$$
$$
\nabla_\zeta J(\zeta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ - (\log \pi_\theta(a|s) + \bar{\mathcal{H}}) \right]
$$

> [!Important]
> Since we use PyTorch's auto-grad, we only need to ensure the functions are tractable.
> Additionally, to keep the alpha value always positive, we actually train $\zeta$ and use $\alpha = \exp(\zeta)$ in practice.

One part that could be problematic is sampling $(a,\log\pi_\theta​)$ from $\pi_\theta$​.
To prevent the gradient chain from breaking due to randomness, the reparameterization trick is used.

In the actual code implementation, the policy network is set to receive the state and output $\mu,\log\sigma$. That is, the action distribution at a given state is assumed to follow a Normal distribution.

Then, using the reparameterization trick, we sample $z = \mu + \sigma\epsilon$ and transform it into an action in the range $(−1,1)$ using $\tanh(z)$.

> [!note]
> The reason for setting it as $\log\sigma$ instead of $\sigma$ is that the $\sigma$ value cannot be negative and significantly influences the distribution.
> During use, it is calculated as $\sigma = \exp(\log\sigma)$.

> [!caution]
> Since the action is pulled through $\tanh$, a Jacobian correction of the probability density is required.
> The formula is as follows:

$$
\log \pi_\theta(a|s) = \log \mu(z|s) - \sum_{i=1}^{D} \log (1 - \tanh^2(z_i))
$$

### Update

$$
w_i := w_i - \lambda_Q\nabla_{w_i}J_{Q}(w_i)
$$
$$
\theta := \theta - \lambda_\theta\nabla_\theta J_\pi(\theta)
$$
$$
\zeta := \zeta - \lambda_\zeta\nabla_\zeta J_\zeta(\zeta)
$$
$$
\alpha := \exp(\zeta)
$$
$$
\bar{w_i} := \tau w_i + (1 - \tau)\bar{w_i}
$$
