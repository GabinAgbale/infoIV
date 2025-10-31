import os

import hydra
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from model.models import iVAE
from data.imca_data import IMCADataset

@hydra.main(config_path="../conf", config_name="config_ivae_imca", version_base=None)
def main(cfg: DictConfig) -> None:
    CHECKPOINT_PATH = cfg.checkpoint_path
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dataset = IMCADataset(
        dimA=cfg.dataset.dimA,
        dimZ=cfg.dataset.dimZ,
        dimX=cfg.dataset.dimX,
        dimY=cfg.dataset.dimY,
        device=device,
        n=cfg.dataset.n,
        causal_effect=cfg.dataset.causal_effect,
        hidden_dim=cfg.dataset.hidden_dim,
        n_layers=cfg.dataset.n_layers,
        confounding_strength=cfg.dataset.confounding_strength,
        indep_latents=cfg.dataset.indep_latents,
    )

    dataset.generate_mixing_funcs()
    if cfg.dataset.path is not None:
        print(f"Loading mixing functions from {cfg.dataset.path}")
        dataset.load_mixing_funcs(cfg.dataset.path)

    dataset.sample()
    train_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True)

    val_dataset = dataset.return_clone()
    val_dataset.n = cfg.dataset.n_val
    val_dataset.sample()
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    test_dataset = dataset.return_clone()
    test_dataset.sample()
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    save_path = os.path.join(logger.log_dir, "mixing_funcs.pth")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.save_mixing_funcs(save_path)

    logger.log_hyperparams(cfg)

    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
    )

    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"ivae_{cfg.expe_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = iVAE.load_from_checkpoint(pretrained_filename)
    else:
        model = iVAE(
            auxiliary_dim=cfg.dataset.dimA,
            input_dim=cfg.dataset.dimX,
            hidden_dims=cfg.model.hidden_dim,
            latent_dim=cfg.dataset.dimZ,
            num_layers=cfg.model.num_layers,
            lr=cfg.optimizer.lr,
            optimizer=cfg.optimizer.name,
            lr_scheduler=cfg.optimizer.scheduler,
            weight_decay=cfg.optimizer.weight_decay,
            activation=cfg.model.activation,
            n_train=len(dataset),
            n_val=len(val_dataset),
            a=cfg.loss.a,
            b=cfg.loss.b,
            c=cfg.loss.c,
            d=cfg.loss.d,
            slope=cfg.model.slope,
            dropout_rate=cfg.model.dropout_rate,
        )
        trainer.fit(model, train_loader, val_loader)

    val_result = trainer.validate(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)

    mcc = test_result[0].get('test_mcc', None)
    torch.save(mcc, os.path.join(logger.log_dir, "test_mcc.pth"))


    result = {"test": test_result, "val": val_result}
    print(result)

if __name__ == "__main__":
    main()