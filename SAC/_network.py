import torch.nn as nn


class _QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(_QNetwork, self).__init__()
        pass

    def forward(self, s, a):
        pass


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
        pass

    def forward(self, s):
        pass
