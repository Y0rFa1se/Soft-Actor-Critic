import gymnasium as gym
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset


class Dataset(IterableDataset):
    def __init__(self, buffer, batch_size):
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            if len(self.buffer) >= self.batch_size:
                yield self.buffer.sample(self.batch_size)
            else:
                import time

                time.sleep(0.1)


class DataModule(L.LightningDataModule):
    def __init__(self, env_id, buffer, batch_size, warmup_steps=1000):
        super().__init__()
        self.env = gym.make(env_id)
        self.buffer = buffer
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.state, _ = self.env.reset()

    def setup(self, stage=None):
        for _ in range(self.warmup_steps):
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.add(
                self.state, action, reward, next_state, terminated or truncated
            )
            self.state = next_state
            if terminated or truncated:
                self.state, _ = self.env.reset()

    def train_dataloader(self):
        return DataLoader(Dataset(self.buffer, self.batch_size), batch_size=None)

    def step(self, agent):
        action = agent(self.state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        self.buffer.add(self.state, action, reward, next_state, terminated or truncated)
        self.state = next_state

        if terminated or truncated:
            self.state, _ = self.env.reset()
        return reward
