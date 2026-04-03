import torch
import torch.nn.functional as F


def _q_loss(
    q_network, policy_network, target_q_network, log_alpha, s, a, r, s_next, done, gamma
):
    with torch.no_grad():
        a_next, log_prob_next = policy_network(s_next)
        q1_next, q2_next = target_q_network(s_next, a_next)
        q_next = torch.min(q1_next, q2_next)

        target_q = r + (1 - done) * gamma * (
            q_next - log_alpha.exp().detach() * log_prob_next
        )

    q1, q2 = q_network(s, a)

    loss_q1 = F.mse_loss(q1, target_q.detach())
    loss_q2 = F.mse_loss(q2, target_q.detach())
    loss = loss_q1 + loss_q2
    return loss


def _policy_loss(q_network, log_alpha, s, a, log_prob):
    q1, q2 = q_network(s, a)
    q = torch.min(q1, q2)

    loss = (log_alpha.exp().detach() * log_prob - q).mean()
    return loss


def _log_alpha_loss(log_alpha, a, log_prob, target_entropy):
    loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
    return loss


def losses(agent, buffer_samples, gamma):
    s, a, r, s_next, done = buffer_samples
    q_loss = _q_loss(
        agent.q_network,
        agent.policy_network,
        agent.target_q_network,
        agent.log_alpha,
        s,
        a,
        r,
        s_next,
        done,
        gamma,
    )

    a, log_prob = agent._sample_action(s)
    policy_loss = _policy_loss(agent.q_network, agent.log_alpha, s, a, log_prob)
    log_alpha_loss = _log_alpha_loss(
        agent.log_alpha, a, log_prob, agent.target_entropy
    )

    return q_loss, policy_loss, log_alpha_loss
