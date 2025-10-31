import os 

import hydra 
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from model.models import BaseVAE
from data.data import RepExSCMDataset


@hydra.main(config_path="../conf", config_name="config_vae", version_base=None)
def main(cfg: DictConfig) -> None:


    CHECKPOINT_PATH = cfg.checkpoint_path
    seed_everything(cfg.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset = RepExSCMDataset(cfg.dataset.dimA, 
                              cfg.dataset.dimZ, 
                              cfg.dataset.dimX,
                              cfg.dataset.dimY, 
                              device=device,
                              n=cfg.dataset.n,
                              hidden_dim=cfg.dataset.hidden_dim,
                              n_layers=cfg.dataset.n_layers,
                              alpha=cfg.dataset.alpha,
                              causal_effect=cfg.dataset.causal_effect,
                              noise_distribution=cfg.dataset.noise_distribution,)
    
    dataset.generate_mixing_funcs()
    if cfg.dataset.path is not None:
        print(f"Loading mixing functions from {cfg.dataset.path}")
        dataset.load_mixing_funcs(cfg.dataset.path)
    
    dataset.sample(cfg.dataset.gamma_train)
    train_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True)

    val_dataset = dataset.return_clone()
    val_dataset.sample(cfg.dataset.gamma_train)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    
    test_dataset = dataset.return_clone()
    test_dataset.sample(cfg.dataset.gamma_test)
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

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"rep4ex_{cfg.expe_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = BaseVAE.load_from_checkpoint(pretrained_filename)
    else:
        model = BaseVAE(input_dim=cfg.dataset.dimX, 
                        hidden_dim=cfg.model.hidden_dim,
                        latent_dim=cfg.dataset.dimZ,
                        lr=cfg.optimizer.lr,
                        lambda_recon=cfg.loss.l,
                        optimizer=cfg.optimizer.name,
                        lr_scheduler=cfg.optimizer.scheduler,
                        weight_decay=cfg.optimizer.weight_decay,
        )
                    
        trainer.fit(model, train_loader, val_loader)

    val_result = trainer.validate(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    
    r2_score = test_result[0].get('test_r2', None)  
    torch.save(r2_score, os.path.join(logger.log_dir, "test_r2_score.pth"))

    result = {"test": test_result, "val": val_result}
    print(result)


if __name__ == "__main__":
    main()

