import numpy as np
import pytorch_lightning as L
import torch

from .networks import PolicyNetwork, TwinQNetwork
from .objectives import loss_log_alpha, loss_policy, loss_q


class Agent(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        self.q_network = TwinQNetwork(config.state_dim, config.action_dim)
        self.policy_network = PolicyNetwork(config.state_dim, config.action_dim)
        self.target_q_network = TwinQNetwork(config.state_dim, config.action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.target_entropy = (
            config.target_entropy
            if config.target_entropy is not None
            else -float(config.action_dim)
        )
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))

        self.automatic_optimization = False

    def _sample_action(self, s):
        mu, log_std = self.policy_network(s)
        std = log_std.exp()
        normal_dist = torch.distributions.Normal(mu, std)

        z = normal_dist.rsample()
        a = torch.tanh(z)

        log_prob = normal_dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return a, log_prob

    def training_step(self, batch, batch_idx):
        q_opt, policy_opt, log_alpha_opt = self.optimizers()
        buffer_samples = batch

        q_loss = loss_q(self, buffer_samples)
        q_opt.zero_grad()
        self.manual_backward(q_loss)
        q_opt.step()

        policy_loss = loss_policy(self, buffer_samples)
        policy_opt.zero_grad()
        self.manual_backward(policy_loss)
        policy_opt.step()

        log_alpha_loss = loss_log_alpha(self, buffer_samples)
        log_alpha_opt.zero_grad()
        self.manual_backward(log_alpha_loss)
        log_alpha_opt.step()

        self._soft_update_target_q()

        self.log_dict(
            {
                "train/q_loss": q_loss,
                "train/policy_loss": policy_loss,
                "train/alpha_loss": log_alpha_loss,
                "train/alpha": self.log_alpha.exp(),
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        opt_q = torch.optim.Adam(self.q_network.parameters(), lr=self.config.q_lr)
        opt_p = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.config.policy_lr
        )
        opt_a = torch.optim.Adam([self.log_alpha], lr=self.config.log_alpha_lr)
        return opt_q, opt_p, opt_a

    def _soft_update_target_q(self):
        for param, target_param in zip(
            self.q_network.parameters(), self.target_q_network.parameters()
        ):
            target_param.data.lerp_(param.data, self.config.tau)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action, _ = self._sample_action(state)

        return action.squeeze(0).cpu().numpy()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        reward = self.trainer.datamodule.step(self)
        self.log("train/step_reward", reward, prog_bar=True)
