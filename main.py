from SAC import Agent, ReplayBuffer


def main():
    agent = Agent(state_dim=1, action_dim=1)
    buffer = ReplayBuffer(max_size=1000)