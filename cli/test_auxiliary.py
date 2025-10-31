import os 
import sys

from omegaconf import OmegaConf

import glob

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

#from pytorch_lightning.callbacks import Callback

from model.models import ContrastiveModel
#from model.utils import PeriodicCallback

from data.data import ContrastiveDataset


seed_everything(42)



def main(path) -> None:

    cfg = OmegaConf.load(os.path.join(path, "hparams.yaml"))

    seed_everything(cfg.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    dataset = ContrastiveDataset(cfg.dataset.dimA, 
                                 cfg.dataset.dimZ, 
                                 cfg.dataset.dimX,
                                 cfg.dataset.dimY, 
                                 device=device,
                                 n=cfg.dataset.n,
                                 hidden_dim=cfg.dataset.hidden_dim,
                                 n_layers=cfg.dataset.n_layers,
                                 n_contrastive_pairs=cfg.dataset.n_contrastive_pairs,)
    
    dataset.generate_mixing_funcs()
    dataset.load_mixing_funcs(os.path.join(path, "mixing_funcs.pth"))
    
    dataset.sample(cfg.dataset.gamma_train)
    data_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    
    logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    logger.log_hyperparams(cfg)

    save_path = os.path.join(logger.log_dir, "mixing_funcs.pth")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.save_mixing_funcs(save_path)

    
    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        # callbacks=[PeriodicCallback(frequency=cfg.trainer.test_freq)]
        )

    last_checkpoint = glob.glob(os.path.join(path, "checkpoints","*.ckpt"))[-1]
    model = ContrastiveModel.load_from_checkpoint(last_checkpoint,
                                                  input_dim=cfg.dataset.dimX, 
                                                  hidden_dim_encoder=cfg.model.encoder.hidden_dim, 
                                                  num_layers_encoder=cfg.model.encoder.num_layers,
                                                  hidden_dim_head=cfg.model.head.hidden_dim,
                                                  num_layers_head=cfg.model.head.num_layers,
                                                  latent_dim=cfg.dataset.dimZ,
                                                  auxiliary_dim=cfg.dataset.dimA,
                                                  lambda_contrastive=cfg.loss.l,
                                                  lr=cfg.optimizer.lr)

    test_result = trainer.test(model, data_loader, verbose=False)
    result = {"test": test_result}
    print(result)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python test_auxiliary.py /path/to/config.yaml")
    #     sys.exit(1)

    # config_path = sys.argv[1]
    config_path = "logs/auxiliary_contrastive3/version_3"
    
    main(config_path)


