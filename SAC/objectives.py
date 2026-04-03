import torch
import torch.nn.functional as F


def loss_q(agent, buffer_samples):
    q_network, target_q_network, log_alpha, gamma = (
        agent.q_network,
        agent.target_q_network,
        agent.log_alpha,
        agent.config.gamma,
    )
    s, a, r, s_next, done = buffer_samples
    with torch.no_grad():
        a_next, log_prob_next = agent._sample_action(s_next)
        q1_next, q2_next = target_q_network(s_next, a_next)
        q_next = torch.min(q1_next, q2_next)

        target_q = r + (1 - done) * gamma * (
            q_next - log_alpha.exp().detach() * log_prob_next
        )

    q1, q2 = q_network(s, a)

    loss_q1 = F.mse_loss(q1, target_q.detach().view_as(q1))
    loss_q2 = F.mse_loss(q2, target_q.detach().view_as(q2))
    loss = loss_q1 + loss_q2
    return loss


def loss_policy(agent, buffer_samples):
    q_network, log_alpha = agent.q_network, agent.log_alpha
    s, a, r, s_next, done = buffer_samples
    a, log_prob = agent._sample_action(s)
    q1, q2 = q_network(s, a)
    q = torch.min(q1, q2)

    loss = (log_alpha.exp().detach() * log_prob - q).mean()
    return loss


def loss_log_alpha(agent, buffer_samples):
    log_alpha, target_entropy = agent.log_alpha, agent.target_entropy
    s, a, r, s_next, done = buffer_samples
    a, log_prob = agent._sample_action(s)
    loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
    return loss
