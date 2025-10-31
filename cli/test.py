import os
import glob
import hydra 
from omegaconf import DictConfig


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger 

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything

from model.models import ContrastiveClassifierModel, InfoNCEModel, AdditiveMLP
from data.data import RepExSCMDataset, ControlVariableDataset


@hydra.main(config_path="../conf", config_name="config_iv", version_base=None)
def main(cfg: DictConfig) -> None:

    seed_everything(cfg.seed)

    # load latest checkpoint
    EXPE_PATH = cfg.expe_path
    ckpt_path = glob.glob(os.path.join(EXPE_PATH, "checkpoints", "*.ckpt"))[-1]
    if not os.path.isfile(ckpt_path) and cfg.type != "oracle":
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")

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
    path_mixing = os.path.join(EXPE_PATH, "mixing_funcs.pth")
    dataset.load_mixing_funcs(path_mixing)
    dataset.sample(cfg.dataset.shifts[-1])

    match cfg.type:
        case "nce":
            print("Loading InfoNCE model...")
            latent_model = InfoNCEModel.load_from_checkpoint(ckpt_path,
                                                            input_dim=cfg.dataset.dimX, 
                                                            hidden_dim=cfg.model.hidden_dim, 
                                                            num_layers=cfg.model.num_layers,
                                                            latent_dim=cfg.dataset.dimZ,
                                                            lr=cfg.optimizer.lr,)
            latent_model = latent_model.to(device)
            model.freeze()
            with torch.no_grad():
                Z_pred = model.encoder(dataset.X)
        
        case "aux":
            print("Loading ContrastiveClassifierModel...")
            model = ContrastiveClassifierModel.load_from_checkpoint(ckpt_path,
                                                                    input_dim=cfg.dataset.dimX, 
                                                                    hidden_dim_encoder=cfg.model.encoder.hidden_dim, 
                                                                    num_layers_encoder=cfg.model.encoder.num_layers,
                                                                    hidden_dim_head=cfg.model.head.hidden_dim,
                                                                    num_layers_head=cfg.model.head.num_layers,
                                                                    latent_dim=cfg.dataset.dimZ,
                                                                    auxiliary_dim=cfg.dataset.dimA,
                                                                    lambda_recon=cfg.loss.l,
                                                                    lr=cfg.optimizer.lr,)
            

    val_cv_dataset = ControlVariableDataset(Z=Z_pred, A=dataset.A, Y=dataset.Y, device=device)
    val_cv_dataset.solve_linear()
    val_cv_dataset.compute_control_variable()
    val_cv_loader = DataLoader(val_cv_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)

    logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    trainer_cv = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        )
    
    causal_model = AdditiveMLP(input_dim=cfg.dataset.dimZ, 
                               hidden_dim=cfg.model.hidden_dim, 
                               output_dim=cfg.dataset.dimY,
                               num_layers=cfg.model.num_layers,
                               lr=cfg.optimizer.lr)
    
    val_result = trainer_cv.validate(causal_model, val_cv_loader, verbose=False)


if __name__ == "__main__":
    main()

