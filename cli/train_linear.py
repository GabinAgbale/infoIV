
import os
import glob
import hydra 
import copy
from omegaconf import DictConfig



import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from data.data import RepExSCMDataset, ZYDataset
from model.models import InfoNCEModel, ContrastiveClassifierModel, LinearRegression

from utils.utils import log_mse_vs_shift_plot, log_dim1_plot, log_pred_gt_plot, log_pred_gt_multiplot

@hydra.main(config_path="../conf", config_name="config_iv", version_base=None)
def main(cfg: DictConfig) -> None:

    seed_everything(cfg.seed)

    device = torch.device(cfg.device)

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

    path_mixing = os.path.join(cfg.expe_path, "mixing_funcs.pth")
    if not os.path.exists(path_mixing):
        raise FileNotFoundError(f"Mixing functions path {path_mixing} does not exist.")
    
    print(f"Loading mixing functions from {path_mixing}")
    dataset.load_mixing_funcs(path_mixing)
    dataset.sample(cfg.dataset.gamma_train)

    val_dataset = dataset.return_clone()
    val_dataset.sample(cfg.dataset.gamma_test)

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

    # load latest checkpoint
    ckpt_path = glob.glob(os.path.join(cfg.expe_path, "checkpoints", "*.ckpt"))[-1]
    if not os.path.isfile(ckpt_path) and cfg.type != "oracle":
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")
    

    # Load adequate model and generate latents
    match cfg.type:
        case "nce":
            print("Loading InfoNCE model...")
            model = InfoNCEModel.load_from_checkpoint(ckpt_path,
                                                      auxiliary_dim=cfg.dataset.dimA,
                                                      input_dim=cfg.dataset.dimX, 
                                                      hidden_dim=cfg.model.hidden_dim, 
                                                      num_layers=cfg.model.num_layers,
                                                      latent_dim=cfg.dataset.dimZ,
                                                      lr=cfg.optimizer.lr,)
            model = model.to(device)
            model.freeze()
            with torch.no_grad():
                Z_pred = model.encoder(dataset.X)
                Z_pred_val = model.encoder(val_dataset.X)
        
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
            
            model = model.to(device)
            model.freeze()
            with torch.no_grad():
                Z_pred = model.encoder(dataset.X)
                Z_pred_val = model.encoder(val_dataset.X)
                
        case "oracle":
            Z_pred = dataset.Z
            Z_pred_val = val_dataset.Z
    
    lin_reg = LinearRegression(input_dim=cfg.dataset.dimZ, 
                               output_dim=cfg.dataset.dimY,
                               lambda_lasso=cfg.loss.lambda_lasso,
                               lr=cfg.optimizer.lr,)

    trainer_cv = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        )
    

    train_effect_dataset = ZYDataset(Z=Z_pred, Y=dataset.Y)
    train_dataloader = DataLoader(train_effect_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)

    val_effect_dataset = ZYDataset(Z=Z_pred_val, Y=val_dataset.Y)
    val_dataloader = DataLoader(val_effect_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    
    trainer_cv = pl.Trainer(default_root_dir=cfg.trainer.root_dir,
                            accelerator=device.type,
                            devices=cfg.trainer.devices,
                            max_epochs=cfg.trainer.max_epochs,
                            logger=logger,)
    
    trainer_cv.fit(lin_reg, train_dataloader, val_dataloader)

    print("Linear Causal Effect fitting done.")
    causal_model_copy = copy.deepcopy(lin_reg)
    pred_gt_list = []

    print("Performing Validation on Distribution Shift:")
    mses = []
    for shift in cfg.dataset.shifts:
        print(f"Shift: {shift}")
        shifted_dataset = dataset.return_clone()
        shifted_dataset.sample(cfg.dataset.gamma_train + shift)
        
        if cfg.type != "oracle":
            Z_pred_shift = model.encoder(shifted_dataset.X)
        else:
            Z_pred_shift = shifted_dataset.Z

        lin_reg_device = lin_reg.to(device)
        with torch.no_grad():
            y_pred = lin_reg_device(Z_pred_shift)
            shifted_val_result = mse_loss(y_pred, shifted_dataset.Y)
            mses.append(shifted_val_result.item())

        print(f"Shift {shift} Validation Result: {shifted_val_result}")

        # pred on shifted dataset
        with torch.no_grad():
            Y_pred = causal_model_copy.to(device)(shifted_dataset.Z)
            pred_gt_list.append((Y_pred.detach().cpu(), shifted_dataset.Y))
        
    log_mse_vs_shift_plot(cfg.dataset.shifts, mses, logger=trainer_cv.logger, global_step=trainer_cv.current_epoch)

    log_pred_gt_multiplot(pred_gt_list,logger=trainer_cv.logger, global_step=trainer_cv.current_epoch, tag="pred vs g-t shift")

if __name__ == "__main__":
    main()
