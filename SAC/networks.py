import torch
import torch.nn as nn


class _QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(_QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, a):
        return self.network(torch.cat([s, a], dim=-1))


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TwinQNetwork, self).__init__()
        self.q1 = _QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = _QNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, s):
        x = self.fc(s)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std
