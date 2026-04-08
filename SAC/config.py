import hydra
import pytorch_lightning as L
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from .agent import Agent
from .datamodule import DataModule
from .buffer import ReplayBuffer

def _agent(cfg, MODE):
    if cfg.runner.agent_checkpoint:
        agent = Agent.load_from_checkpoint(
            cfg.runner.agent_checkpoint, weights_only=False
        )
    else:
        agent = Agent(cfg.agent, MODE)

    return agent


def _buffer(cfg):
    buffer = ReplayBuffer(
        cfg.agent.state_dim, cfg.agent.action_dim, cfg.buffer.max_size
    )

    if cfg.runner.buffer_load:
        buffer.load(cfg.runner.buffer_load)

    return buffer


def _datamodule(cfg, buffer):
    dm = DataModule(cfg.env_id, buffer, cfg.runner.batch_size, cfg.runner.warmup_steps)

    return dm


def _trainer(cfg, agent):
    wandb_logger = WandbLogger(project=cfg.wandb.project, log_model="all")

    trainer = L.Trainer(
        **cfg.trainer,
        logger=wandb_logger,
    )

    return trainer
