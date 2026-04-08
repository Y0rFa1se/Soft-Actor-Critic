import pytorch_lightning as L

from .config import _agent, _buffer, _datamodule, _trainer


def train(cfg, MODE):
    L.seed_everything(cfg.runner.seed if "seed" in cfg.runner else None)

    agent = _agent(cfg, "train")
    buffer = _buffer(cfg)
    dm = _datamodule(cfg, buffer)
    trainer = _trainer(cfg, agent)

    trainer.fit(agent, datamodule=dm)

    if cfg.runner.buffer_save:
        buffer.save(cfg.runner.buffer_save)

    return agent, buffer, dm, trainer
