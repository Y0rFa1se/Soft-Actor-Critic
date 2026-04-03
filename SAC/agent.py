import pathlib

import torch
import torch.optim as optim

from .networks import PolicyNetwork, TwinQNetwork
from .objectives import losses


class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        tau=0.005,
        lr=1e-4,
        target_entropy=None,
        device="cpu",
    ):
        self.tau = tau
        self.lr = lr
        self.device = device

        self.q_network = TwinQNetwork(state_dim, action_dim).to(self.device)
        self.policy_network = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network = TwinQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(action_dim)
        )
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

    def save(self, path, filename="agent.pth", overwrite=False):
        filepath = pathlib.Path(path) / filename

        if not overwrite and filepath.exists():
            raise FileExistsError(
                "File already exists. Use overwrite=True to replace it."
            )

        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "policy_network": self.policy_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "target_entropy": self.target_entropy,
                "log_alpha": self.log_alpha,
                "q_optimizer": self.q_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, path, filename="agent.pth"):
        filepath = pathlib.Path(path) / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist.")

        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])

        self.target_entropy = checkpoint["target_entropy"]
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])

        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])

    def _sample_action(self, state):
        mu, log_std = self.policy_network(state)
        std = log_std.exp()
        normal_dist = torch.distributions.Normal(mu, std)

        action = normal_dist.rsample()
        action = torch.tanh(action)

        log_prob = normal_dist.log_prob(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def play(self, state):
        with torch.no_grad():
            action, _ = self._sample_action(state)

        return action.cpu().numpy()

    def update(self, buffer_samples, gamma):
        q_loss, policy_loss, log_alpha_loss = losses(self, buffer_samples, gamma)

        self.q_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.log_alpha_optimizer.zero_grad()

        q_loss.backward()
        policy_loss.backward()
        log_alpha_loss.backward()

        self.q_optimizer.step()
        self.policy_optimizer.step()
        self.log_alpha_optimizer.step()

        for param, target_param in zip(
            self.q_network.parameters(), self.target_q_network.parameters()
        ):
            target_param.data.lerp_(param.data, self.tau)
