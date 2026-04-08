import pytorch_lightning as L

from .config import _agent, _buffer, _datamodule, _trainer

def test(cfg):
    L.seed_everything(cfg.runner.seed if "seed" in cfg.runner else None)

    agent = _agent(cfg, "test")
    buffer = _buffer(cfg)
    dm = _datamodule(cfg, buffer)
    trainer = _trainer(cfg, agent)

    trainer.test(agent, datamodule=dm)
