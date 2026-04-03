import hydra
import pytorch_lightning as L
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from SAC import Agent, DataModule, ReplayBuffer


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    L.seed_everything(cfg.train.seed)

    agent = Agent(cfg.agent)
    buffer = ReplayBuffer(
        cfg.agent.state_dim, cfg.agent.action_dim, cfg.buffer.max_size
    )

    dm = DataModule(cfg.env_id, buffer, cfg.train.batch_size)

    wandb_logger = WandbLogger(project=cfg.wandb.project, log_model="all")

    trainer = L.Trainer(
        **cfg.trainer,
        logger=wandb_logger,
    )

    trainer.fit(agent, datamodule=dm)


if __name__ == "__main__":
    main()
