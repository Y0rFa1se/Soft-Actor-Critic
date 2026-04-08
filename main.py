import hydra
import pytorch_lightning as L
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from SAC import train, test


MODE = "train"

@hydra.main(config_path="configs", config_name=MODE, version_base="1.3")
def main(cfg: DictConfig):
    if MODE == "train":
        agent, buffer, dm, trainer = train(cfg)
    elif MODE == "test":
        test(cfg)


if __name__ == "__main__":
    main()
