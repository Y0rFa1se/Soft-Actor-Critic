# Soft-Actor-Critic

### [🇰🇷 한국어](/README.md) | [🇺🇸 English](/docs/README_en.md)

[SAC with automatic temperature adjustment](https://arxiv.org/pdf/1812.05905) 버전 구현 코드입니다.

가장 표준적인 Twin-Q model을 이용하고 있습니다. (두 Q중 작은것을 Actor로 사용)

---

### reference

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

> [!NOTE]
> skeleton부터 직접 구현하려면 [이거](https://github.com/Y0rFa1se/Soft-Actor-Critic/tree/skeleton)


## 실행방법

### 학습 코드 실행
```bash
uv run main.py
```

### 테스트 코드 실행
```bash
uv run test.py
```

현재 continuous lunar lander를 학습해 solve([200점 이상](https://gymnasium.farama.org/environments/box2d/lunar_lander/#rewards))한parameter를 같이 두었으니 확인해 볼 수 있다.

![sample_gif](/imgs/lunarlander_sample.gif)

## Review

### Objective function

$$
J_Q(w_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}[\frac{1}{2}(Q_{w_i}(s, a) - \hat{Q}(s, a))^2]
$$
$$
$$
$$
\text{where } \hat{Q}(s, a) := r + \gamma(\min\limits_{j = 1, 2} Q_{\bar{w}_j}(s', a') - \alpha\log\pi_\theta(a'|s')),
$$
$$
\text{with }(a', \log\pi_\theta(a'|s')) \sim \pi_\theta.
$$
$$
J_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}}[\alpha\log\pi_\theta(a|s) - \min\limits_{j = 1, 2}Q_{w_j}(s, a)]
$$
$$
$$
$$
\text{with } a \sim \pi_\theta.
$$
$$
$$
$$
\text{Let } \zeta := \log\alpha.
$$
$$
$$
$$
J(\zeta) = \mathbb{E}_{s \sim \mathcal{D}}[-\zeta(\log\pi_\theta(a|s) + \bar{\mathcal{H}})]
$$
$$
$$
$$
\text{with } a \sim \pi_\theta.
$$

### Gradient

$$
\nabla_{w_i} J_Q(w_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q_{w_i}(s, a) - \hat{Q}(s, a) \right) \nabla_{w_i} Q_{w_i}(s, a) \right]
$$
$$
$$
$$
\nabla_\theta J_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \left( \alpha \log \pi_\theta(a|s) - \min_{j=1,2} Q_{w_j}(s, a) \right) + \nabla_\theta (\alpha \log \pi_\theta(a|s)) \right]
$$
$$
$$
$$
\nabla_\zeta J(\zeta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \left[ - (\log \pi_\theta(a|s) + \bar{\mathcal{H}}) \right]
$$

> [!Important]
> pytorch의 auto grad를 사용할 것이기 때문에 tractable함만 알고 있으면 된다.
> 또한 alpha 값을 항상 양수로 유지하기 위해 실제로는 $\zeta$를 학습하고 $\alpha = e^\zeta$로 구해 사용한다.

다만 문제가 될만한 부분은 $\pi_\theta$에서 $(a, \log\pi_\theta)$를 샘플링 하는 부분인데,
이는 randomness에 의해 gradient chain이 끊어지지 않도록 reparametrization trick을 사용한다.

실제 코드 구현에서는 policy network를 state만들 받아서 $\mu, \log\sigma$를 출력하도록 설정해 두었다. 즉 state에서의 action의 분포가 normal을 따른다고 가정하였다.

이후 reparametrization trick을 이용해 $z = \mu + \sigma\epsilon$ 으로 샘플링하고 $\tanh(z)$로 $(-1,1)$의 값으로 바꾸어 action으로 사용한다.

> [!note]
> $\sigma$가 아닌 $\log\sigma$로 설정한 이유는 $\sigma$ 값은 음수가 될 수 없고 분포에 큰 영향을 주기 때문이다.
> 이후 사용시에는 $\sigma = \exp(\log\sigma)$를 계산하여 사용한다.

> [!caution]
> $\tanh$를 거쳐 action을 뽑기 때문에 확률밀도의 Jacobian 보정이 필요하다.
> 수식은 아래와 같다.

> [!note]
> 수정내역) policy network에서 $\mu, \log\sigma$ 를 뽑아 샘플링 하지 않고 $\mu$ 를 그대로 $\tanh$ 함수를 거쳐 사용하는 act 함수 추가.
> 테스트 환경에서는 노이즈를 추가한 액션을 뽑을 필요가 없기 때문

$$
\log \pi_\theta(a|s) = \log \mu(z|s) - \sum_{i=1}^{D} \log (1 - \tanh^2(z_i))
$$

### Update

$$
w_i := w_i - \lambda_Q\nabla_{w_i}J_{Q}(w_i)
$$
$$
$$
$$
\theta := \theta - \lambda_\theta\nabla_\theta J_\pi(\theta)
$$
$$
$$
$$
\zeta := \zeta - \lambda_\zeta\nabla_\zeta J_\zeta(\zeta)
$$
$$
$$
$$
\alpha := \exp(\zeta)
$$
$$
$$
$$
\bar{w_i} := \tau w_i + (1 - \tau)\bar{w_i}
$$
