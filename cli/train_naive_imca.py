import os 

import hydra 
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from model.models import NeuralFitter
from data.imca_data import IMCADataset
from data.data import SimpleDataset

from utils.utils import save_causal_effect_plot



@hydra.main(config_path="../conf", config_name="config_imca_full", version_base=None)
def main(cfg: DictConfig) -> None:

    CHECKPOINT_PATH = cfg.checkpoint_path
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")


    base_logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    dataset = IMCADataset(dimA=cfg.dataset.dimA, 
                          dimZ=cfg.dataset.dimZ, 
                          dimX=cfg.dataset.dimX,
                          dimY=cfg.dataset.dimY, 
                          device=device,
                          n=cfg.dataset.n,
                          causal_effect=cfg.dataset.causal_effect,
                          hidden_dim=cfg.dataset.hidden_dim,
                          n_layers=cfg.dataset.n_layers,
                          confounding_strength=cfg.dataset.confounding_strength,
                          indep_latents=cfg.dataset.indep_latents,)
    
    dataset.generate_mixing_funcs()
    if cfg.dataset.path is not None:
        print(f"Loading mixing functions from {cfg.dataset.path}")
        dataset.load_mixing_funcs(cfg.dataset.path)
    
    dataset.sample()

    val_dataset = dataset.return_clone()
    val_dataset.n = cfg.dataset.n_val
    val_dataset.sample()

    test_dataset = dataset.return_clone()
    test_dataset.sample()
  
    causal_effect_train_dataset = SimpleDataset(input_key="Z",
                                                 output_key="Y",
                                                 input_data=dataset.Z, 
                                                 output_data=dataset.Y)
    
    causal_effect_val_dataset = SimpleDataset(input_key="Z",
                                               output_key="Y",
                                               input_data=val_dataset.Z,
                                               output_data=val_dataset.Y)
    
    causal_effect_dataloader = DataLoader(causal_effect_train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    causal_effect_dataloader_val = DataLoader(causal_effect_val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    model = NeuralFitter(input_key="Z",
                         output_key="Y",
                         input_dim=cfg.dataset.dimZ,
                         output_dim=cfg.dataset.dimY,
                         hidden_dim=cfg.model.step3.hidden_dim,
                         num_layers=cfg.model.step3.num_layers,
                         lr=cfg.optimizer.step3.lr,
                         weight_decay=cfg.optimizer.step3.weight_decay,
                         activation=cfg.model.step3.activation,
                         lr_scheduler=cfg.optimizer.step3.scheduler,
    )

    logger_step3 = TensorBoardLogger(
        save_dir=base_logger.log_dir,
        name="step3",
    )

    trainer_step3 = pl.Trainer(
        accelerator=device.type,
        devices=[device.index] if device.type == "cuda" else None,
        max_epochs=cfg.trainer.step3.max_epochs,
        logger=logger_step3,
    )

    print("Training causal effect estimator on estimated latents...")
    trainer_step3.fit(model, causal_effect_dataloader, causal_effect_dataloader_val)
    val_result = trainer_step3.validate(model, causal_effect_dataloader_val, verbose=False)
    print({"val": val_result})

    print("Evaluate performances on causal effect estimation...")

    y_ce = test_dataset.l(test_dataset.Z).cpu()
    with torch.no_grad():
        y_pred = model(test_dataset.Z).cpu()
        mse = torch.mean((y_pred - y_ce)**2).item()
        print({"mse": mse})
        val_result[0]["mse"] = mse

    torch.save(mse, base_logger.log_dir + "/mse.pt")

    save_causal_effect_plot(y_pred.detach(), y_ce.detach(), base_logger)

if __name__ == "__main__":
    main()