import os 

import hydra 
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dsprites_data import dSpritesLoader, dSpritesFullDataset, dSpritesConditionalLoader
from data.utils import generate_full_rank_matrix
from data.target_fn import Linear_mixing, posX_scale_target
from model.image_model import ImageNCE

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger




@hydra.main(config_path="../conf", config_name="config_nce_dsprites", version_base=None)
def main(cfg: DictConfig) -> None:


    CHECKPOINT_PATH = cfg.checkpoint_path
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dsprites_path = 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    # Load dSprites data
    loader = dSpritesLoader(dsprites_path, device=device)
    # Use conditional loader to filter by shape=1
    conditions = {'shape': 1}
    confounder_name = 'posY'
    cond_loader = dSpritesConditionalLoader(loader, conditions=conditions)
    
    confounding_fn = lambda x: x - 0.5
    instrument_mixing = Linear_mixing(generate_full_rank_matrix(5, 8), device=device)
    target_fn = posX_scale_target

    # Create dataset
    train_dataset = dSpritesFullDataset(
        loader=cond_loader,
        confounder_name=confounder_name,
        instrument_mixing=instrument_mixing,
        target_fn=target_fn,
        confounding_fn=confounding_fn,
        device=device,
        test=False,
    )
    train_dataset.sample(cfg.n_train)

    val_dataset = dSpritesFullDataset(
        loader=cond_loader,
        confounder_name=confounder_name,
        instrument_mixing=instrument_mixing,
        target_fn=target_fn,
        confounding_fn=confounding_fn,
        device=device,
        test=False,
    )
    val_dataset.sample(cfg.n_val)

    test_dataset = dSpritesFullDataset(
        loader=cond_loader,
        confounder_name=confounder_name,
        instrument_mixing=instrument_mixing,
        target_fn=target_fn,
        device=device,
        test=True,
    )
    test_dataset.sample(cfg.n_val)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.num_workers)

    logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    logger.log_hyperparams(cfg)

    save_path = os.path.join(logger.log_dir, "mixing_funcs.pth")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    #train_dataset.save_mixing_funcs(save_path)

    model = ImageNCE(
        auxiliary_dim=8,
        latent_dim=6,
        input_channels=1,
        hidden_dims=cfg.model.hidden_dims,
        lambda_recon=cfg.loss.lambda_recon,
        temperature=cfg.loss.temperature,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,  
        optimizer=cfg.optimizer.name,
        lr_scheduler=cfg.lr_scheduler.name,
    )

    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=[device.index] if device.type == "cuda" else None,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()

