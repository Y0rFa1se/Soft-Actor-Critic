from ._network import TwinQNetwork, PolicyNetwork

import torch
import pathlib


class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        tau=0.005,
        lr=1e-4,
        target_entropy=None,
        device="cpu",
    ):
        self.gamma = gamma
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

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.lr
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

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
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
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
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        
    def play(self, state):
        pass

    def update(self, batch):
        pass
