import os 
import glob
import copy

import hydra 
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from model.models import InfoNCEModel, AdditiveMLP, ContrastiveClassifierModel, LinearControlFunction
from data.data import RepExSCMDataset, ControlVariableDatasetNCE

from utils.utils import log_mse_vs_shift_plot, log_dim1_plot,  log_pred_gt_multiplot, linear_regression


@hydra.main(config_path="../conf", config_name="config_iv", version_base=None)
def main(cfg: DictConfig) -> None:

    if not os.path.exists(cfg.expe_path):
        raise FileNotFoundError(f"IV experiment requires a valid experiment path, {cfg.expe_path} does not exist.")

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
    
    match cfg.type:
        case "nce":
            print("Loading InfoNCE model...")
            model = InfoNCEModel.load_from_checkpoint(ckpt_path,
                                                      input_dim=cfg.dataset.dimX, 
                                                      auxiliary_dim=cfg.dataset.dimA,
                                                      hidden_dim=cfg.model.hidden_dim, 
                                                      num_layers=cfg.model.num_layers,
                                                      latent_dim=cfg.dataset.dimZ,
                                                      lr=cfg.optimizer.lr,)
            model = model.to(device)
            model.freeze()
            with torch.no_grad():
                Z_pred = model.encoder(dataset.X)
                Z_pred_val = model.encoder(val_dataset.X)
        
                if cfg.iv_type == "cf":
                    Z_aux_pred = model.W(dataset.A)
                    Z_aux_pred_val = model.W(val_dataset.A)
                
        case "oracle":
            Z_pred = dataset.Z
            Z_aux_pred = dataset.A @ dataset._M0.T
            Z_pred_val = val_dataset.Z
            Z_aux_pred_val = val_dataset.A @ dataset._M0.T

        
        case _:
            raise ValueError(f"Unknown type {cfg.type}. Supported types are 'nce' or 'oracle'.")

    train_cv_dataset = ControlVariableDatasetNCE(Z=Z_pred, Z_aux=Z_aux_pred, Y=dataset.Y, device=device)
    train_cv_dataset.compute_control_variable()
    train_cv_loader = DataLoader(train_cv_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)


    val_cv_dataset = ControlVariableDatasetNCE(Z=Z_pred_val, Z_aux=Z_aux_pred_val, Y=val_dataset.Y, device=device)
    val_cv_dataset.compute_control_variable()
    val_cv_loader = DataLoader(val_cv_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    
    # Train additive MLP on V and phi(X)
    match cfg.dataset.causal_effect:
        case "nonlinear":
            causal_model = AdditiveMLP(input_dim=cfg.dataset.dimZ, 
                                       hidden_dim=cfg.model.hidden_dim, 
                                       output_dim=cfg.dataset.dimY,
                                       num_layers=cfg.model.num_layers,
                                       lr=cfg.optimizer.lr)
        
        case "linear":
            causal_model = LinearControlFunction(input_dim=cfg.dataset.dimZ,
                                                 output_dim=cfg.dataset.dimY,
                                                 lr=cfg.optimizer.lr )

    
    trainer_cv = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        )
    
 
    trainer_cv.fit(causal_model, train_cv_loader, val_cv_loader)
    val_result = trainer_cv.validate(causal_model, val_cv_loader, verbose=False)
    train_result = trainer_cv.test(causal_model, train_cv_loader, verbose=False)

    print("IV regression training done.")
    result = {"train": train_result, "val": val_result}

    
    print("Perform Validation on Distribution Shift:")
    mses = []
    
    causal_model_copy = copy.deepcopy(causal_model).to(device)
    pred_gt_list = []

    for shift in cfg.dataset.shifts:
        print(f"Shift: {shift}")
        shifted_dataset = dataset.return_clone()
        shifted_dataset.sample(cfg.dataset.gamma_train + shift)
        
        match cfg.type:
            case "nce":
                Z_pred_shift = model.encoder(shifted_dataset.X)
                Z_aux_pred_shift = model.W(shifted_dataset.A)
            case "oracle":
                Z_pred_shift = shifted_dataset.Z
                Z_aux_pred_shift = shifted_dataset.A @ shifted_dataset._M0.T

       
        ### Computing MSE loss on shifted dataset ###
        shifted_cv_dataset = ControlVariableDatasetNCE(Z=Z_pred_shift, Z_aux=Z_aux_pred_shift, Y=shifted_dataset.Y, device=device)
        # Avoiding data leakage, W and b shall not be recomputed from the shifted dataset

        shifted_cv_dataset.compute_control_variable()
        shifted_loader = DataLoader(shifted_cv_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
        
        shifted_val_result = trainer_cv.validate(causal_model, shifted_loader, verbose=False)
        print(f"Shifted Validation Result: {shifted_val_result}")
        mses.append(shifted_val_result[0]["val_mse"])

            # pred on shifted dataset
        with torch.no_grad():
            Y_pred = causal_model_copy(shifted_cv_dataset.Z, shifted_cv_dataset.V)
            pred_gt_list.append((Y_pred.detach().cpu(), shifted_cv_dataset.Y))
            

    log_pred_gt_multiplot(pred_gt_list,logger=trainer_cv.logger, global_step=trainer_cv.current_epoch, tag="pred vs g-t shift")

    log_mse_vs_shift_plot(cfg.dataset.shifts, mses, logger=trainer_cv.logger, global_step=trainer_cv.current_epoch)


    if cfg.dataset.dimA == 1:
        log_dim1_plot(dataset=dataset,
                      latent_model=model,
                      causal_model=causal_model.to(device),
                      logger=trainer_cv.logger, 
                      global_step=trainer_cv.current_epoch)

    print(result)

if __name__ == "__main__":
    main()

