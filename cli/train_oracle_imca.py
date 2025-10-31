import os 

import hydra 
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from model.models import InfoNCEModel, NeuralFitter
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
    train_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True)

    val_dataset = dataset.return_clone()
    val_dataset.n = cfg.dataset.n_val
    val_dataset.sample()
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    
    test_dataset = dataset.return_clone()
    test_dataset.sample()
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    save_path = os.path.join(base_logger.log_dir, "mixing_funcs.pth")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.save_mixing_funcs(save_path)
    
    ####  2nd STEP: TRAIN NEURAL FITTER TO LEARN ESTIMATED FEATURES BASED ON INSTRUMENT A ####
    
    latent_instrument_dataset_train = SimpleDataset(input_key="A",
                                                    output_key="Z",
                                                    input_data=dataset.A.cpu(),
                                                    output_data=dataset.Z.cpu())
    
    latent_instrument_dataset_val = SimpleDataset(input_key="A",
                                                  output_key="Z",
                                                  input_data=val_dataset.A.cpu(),
                                                  output_data=val_dataset.Z.cpu())
    
    li_train_loader = DataLoader(latent_instrument_dataset_train, batch_size=cfg.dataset.batch_size, shuffle=True)
    li_val_loader = DataLoader(latent_instrument_dataset_val, batch_size=cfg.dataset.batch_size, shuffle=False)

    model_step2 = NeuralFitter(input_key="A",
                         output_key="Z",
                         input_dim=cfg.dataset.dimA,
                         output_dim=cfg.dataset.dimZ,
                         hidden_dim=cfg.model.step2.hidden_dim,
                         num_layers=cfg.model.step2.num_layers,
                         lr=cfg.optimizer.step2.lr,
                         weight_decay=cfg.optimizer.step2.weight_decay,
                         activation=cfg.model.step2.activation,
                         optimizer=cfg.optimizer.step2.name,
                         lr_scheduler=cfg.optimizer.step2.scheduler,
                         dropout=cfg.model.step2.dropout_rate,
                         slope=cfg.model.step2.slope,
                         )
    
    logger_step2 = TensorBoardLogger(
        save_dir=base_logger.log_dir,
        name="step2",
    )
    trainer_step2 = pl.Trainer(
        accelerator=device.type,
        devices=[device.index] if device.type == "cuda" else None,
        max_epochs=cfg.trainer.step2.max_epochs,
        logger=logger_step2,
    )


    print("Training neural fitter to learn estimated features based on instrument A...")

    trainer_step2.fit(model_step2, li_train_loader, li_val_loader)
    val_result = trainer_step2.validate(model_step2, li_val_loader, verbose=False)

    ####  3rd STEP: ESTIMATE CAUSAL EFFECT ON ESTIMATED LATENTS ####

    with torch.no_grad():
        z_pred = model_step2(val_dataset.A).cpu()
    causal_effect_dataset = SimpleDataset(input_key="Z",
                                          output_key="Y",
                                          input_data=z_pred.cpu(),
                                          output_data=val_dataset.Y.cpu())
    
    with torch.no_grad():
        z_pred_val = model_step2(test_dataset.A).cpu()
    causal_effect_dataset_val = SimpleDataset(input_key="Z",
                                              output_key="Y",
                                              input_data=z_pred_val.cpu(),
                                              output_data=test_dataset.Y.cpu())
    
    causal_effect_dataloader = DataLoader(causal_effect_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    causal_effect_dataloader_val = DataLoader(causal_effect_dataset_val, batch_size=cfg.dataset.batch_size, shuffle=False)

    model_step3 = NeuralFitter(input_key="Z",
                               output_key="Y",
                               input_dim=cfg.dataset.dimZ,
                               output_dim=cfg.dataset.dimY,
                               hidden_dim=cfg.model.step3.hidden_dim,
                               num_layers=cfg.model.step3.num_layers,
                               lr=cfg.optimizer.step3.lr,
                               weight_decay=cfg.optimizer.step3.weight_decay,
                               activation=cfg.model.step3.activation,
                               lr_scheduler=cfg.optimizer.step3.scheduler,
                               optimizer=cfg.optimizer.step3.name,
                               dropout=cfg.model.step3.dropout_rate,
                               slope=cfg.model.step3.slope,
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
    trainer_step3.fit(model_step3, causal_effect_dataloader, causal_effect_dataloader_val)
    val_result = trainer_step3.validate(model_step3, causal_effect_dataloader_val, verbose=False)
    print({"val": val_result})

    print("Evaluate performances on causal effect estimation...")

    y_ce = test_dataset.l(test_dataset.Z).cpu()
    with torch.no_grad():
        z_pred = model_step2(test_dataset.A).cpu()
        y_pred = model_step3(z_pred).cpu()
        mse = torch.mean((y_pred - y_ce)**2).item()
        print({"mse": mse})
        val_result[0]["mse"] = mse

    torch.save(mse, base_logger.log_dir + "/mse.pt")

    save_causal_effect_plot(y_pred.detach(), y_ce.detach(), base_logger)

if __name__ == "__main__":
    main()