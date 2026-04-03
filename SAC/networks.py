import torch
import torch.nn as nn


class _QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(_QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        q_value = self.network(x)
        return q_value


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TwinQNetwork, self).__init__()
        self.q_network1 = _QNetwork(state_dim, action_dim)
        self.q_network2 = _QNetwork(state_dim, action_dim)

    def forward(self, s, a):
        q1 = self.q_network1(s, a)
        q2 = self.q_network2(s, a)
        return q1, q2


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, s):
        x = self.fc_network(s)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std
