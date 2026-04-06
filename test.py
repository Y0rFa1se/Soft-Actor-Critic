import gymnasium as gym
import hydra
from omegaconf import DictConfig

from SAC import Agent


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    env = gym.make(cfg.env_id, render_mode="human")
    checkpoint = "parameters/ll_cont_solved_params.ckpt"
    agent = Agent.load_from_checkpoint(checkpoint, weights_only=False)
    agent.eval()

    num_episode = 10
    for episode in range(num_episode):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            total_reward += reward

        print(f"EPISODE {episode + 1}: Total Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
