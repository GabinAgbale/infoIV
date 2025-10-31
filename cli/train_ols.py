import os 
import copy
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

import hydra 
from omegaconf import DictConfig

from data.data import RepExSCMDataset
from model.models import OLSModel, LinearRegression

from utils.utils import log_mse_vs_shift_plot, log_dim1_plot, log_pred_gt_multiplot



@hydra.main(config_path="../conf", config_name="config_ols", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to train the OLS model.
    """
    
    if not os.path.exists(cfg.expe_path):
        raise FileNotFoundError(f"IV experiment require a valid experiment path, {cfg.expe_path} does not exist.")


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
                              noise_distribution=cfg.dataset.noise_distribution,
                              confounding_strength=cfg.dataset.confounding_strength,
                              )
    
    dataset.generate_mixing_funcs()

    path_mixing = os.path.join(cfg.expe_path, "mixing_funcs.pth")
    if not os.path.exists(path_mixing):
        raise FileNotFoundError(f"Mixing functions path {path_mixing} does not exist.")
        
    print(f"Loading mixing functions from {path_mixing}")
    dataset.load_mixing_funcs(path_mixing)
    dataset.sample(cfg.dataset.gamma_train)
    train_dataloader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True)

    val_dataset = dataset.return_clone()
    val_dataset.sample(cfg.dataset.gamma_test)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        )

    save_path = os.path.join(logger.log_dir, "mixing_funcs.pth")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.save_mixing_funcs(save_path)
    
    logger.log_hyperparams(cfg)

    valid_types = {"observed": dataset.dimX, 
                   "latent": dataset.dimZ, 
                   "2s": dataset.dimZ,
                   "instrument": dataset.dimA}
    
    
    model = OLSModel(input_dim=valid_types[cfg.type],
                     output_dim=dataset.dimY,
                     hidden_dim=cfg.model.hidden_dim,
                     num_layers=cfg.model.num_layers,
                     lr=cfg.optimizer.lr,
                     type=cfg.type)

    trainer.fit(model, train_dataloader, val_dataloader)

    val_results = trainer.validate(model, val_dataloader, verbose=False)
    print(f"Validation results: {val_results}")

    causal_model_copy = copy.deepcopy(model)
    pred_gt_list = []
    mses = []

    print("Performing Validation on Distribution Shift:")
    for shift in cfg.dataset.shifts:
        print(f"Shift: {shift}")
        shifted_dataset = dataset.return_clone()
        shifted_dataset.sample(cfg.dataset.gamma_train + shift)

        shifted_loader = DataLoader(shifted_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
        
        shifted_val_result = trainer.validate(model, shifted_loader, verbose=False)
        print(f"Shifted Validation Result: {shifted_val_result}")
        mses.append(shifted_val_result[0]['val_mse'])

         # pred on shifted dataset
        with torch.no_grad():
            match cfg.type:
                case "observed":
                    input_data = shifted_dataset.X
                case "latent":
                    input_data = shifted_dataset.Z
                case "instrument":
                    input_data = shifted_dataset.A

            Y_pred = causal_model_copy.to(device)(input_data)
            pred_gt_list.append((Y_pred.detach().cpu(), shifted_dataset.Y))
    
    torch.save(torch.Tensor(mses), os.path.join(logger.log_dir, "mses.pth"))
    log_pred_gt_multiplot(pred_gt_list,logger=trainer.logger, global_step=trainer.current_epoch, tag="pred vs g-t shift")


    log_mse_vs_shift_plot(cfg.dataset.shifts, mses, logger=trainer.logger, global_step=trainer.current_epoch)

if __name__ == "__main__":
    main()