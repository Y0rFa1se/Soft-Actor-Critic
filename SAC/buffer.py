from pathlib import Path
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.states = np.empty((max_size, state_dim), dtype=np.float32)
        self.actions = np.empty((max_size, action_dim), dtype=np.float32)
        self.rewards = np.empty((max_size, 1), dtype=np.float32)
        self.next_states = np.empty((max_size, state_dim), dtype=np.float32)
        self.dones = np.empty((max_size, 1), dtype=np.float32)

    def __len__(self):
        return self.size

    def clear(self):
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(
            self.size, batch_size, replace=False if self.size >= batch_size else True
        )

        return (
            torch.from_numpy(self.states[idxs]),
            torch.from_numpy(self.actions[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            torch.from_numpy(self.next_states[idxs]),
            torch.from_numpy(self.dones[idxs]),
        )

    def save(self, path: str|Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            states=self.states[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            next_states=self.next_states[: self.size],
            dones=self.dones[: self.size],
            ptr=self.ptr,
            size=self.size,
        )

    def load(self, path: str|Path):
        path = Path(path)
        data = np.load(path)

        self.states[: data["states"].shape[0]] = data["states"]
        self.actions[: data["actions"].shape[0]] = data["actions"]
        self.rewards[: data["rewards"].shape[0]] = data["rewards"]
        self.next_states[: data["next_states"].shape[0]] = data["next_states"]
        self.dones[: data["dones"].shape[0]] = data["dones"]
        self.ptr = data["ptr"]
        self.size = data["size"]
