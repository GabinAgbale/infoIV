import os
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from model.models import InfoNCEModel, NeuralFitter
from data.imca_data import IMCADataset
from data.data import SimpleDataset

def run_all_checkpoints(main_folder, cfg):
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    base_logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    for recon_folder in sorted(os.listdir(main_folder)):
        recon_path = os.path.join(main_folder, recon_folder)
        if not os.path.isdir(recon_path):
            continue
         # create logger 
        recon_logger = TensorBoardLogger(
            save_dir=base_logger.log_dir,
            name=f"{recon_folder}",
            log_graph=True,
        )
        for version_folder in sorted(os.listdir(recon_path)):
            version_path = os.path.join(recon_path, version_folder)
            checkpoint_dir = os.path.join(version_path, "checkpoints")

            version_logger = TensorBoardLogger(
                save_dir=recon_logger.log_dir,
                name=f"{version_folder}",
                log_graph=True,
            )
            # load model 
            if not os.path.isdir(checkpoint_dir):
                continue
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if not ckpt_files:
                continue
            ckpt_file = os.path.join(checkpoint_dir, ckpt_files[0])
            print(f"Processing checkpoint: {ckpt_file}")
            model = InfoNCEModel.load_from_checkpoint(ckpt_file,
                                                      input_dim=cfg.dataset.dimX, 
                                                        auxiliary_dim=cfg.dataset.dimA,
                                                        hidden_dims=cfg.model.step1.hidden_dim, 
                                                        num_layers=cfg.model.step1.num_layers,
                                                        latent_dim=cfg.dataset.dimZ,
                                                        lr=cfg.optimizer.step1.lr,
                                                        lambda_recon=cfg.loss.l,
                                                        temperature=cfg.loss.temperature,
                                                        optimizer=cfg.optimizer.step1.name,
                                                        lr_scheduler=cfg.optimizer.step1.scheduler,
                                                        weight_decay=cfg.optimizer.step1.weight_decay,
                                                        activation=cfg.model.step1.activation,
                                                        dropout_rate=cfg.model.step1.dropout_rate,
                                                        slope=cfg.model.step1.slope,
                                                      )

            # load dataset
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
            path_mixing_funcs = os.path.join(version_path, "mixing_funcs.pth")
            dataset.generate_mixing_funcs()
            dataset.load_mixing_funcs(path_mixing_funcs)

            val_dataset = dataset.return_clone()
            val_dataset.n = cfg.dataset.n_val
            val_dataset.sample()

            ####  2nd STEP: TRAIN NEURAL FITTER TO LEARN ESTIMATED FEATURES BASED ON INSTRUMENT A ####
    
            # create dataset
            train_dataset_step2 = dataset.return_clone()
            train_dataset_step2.sample()

            with torch.no_grad():
                z_pred = model.encoder(train_dataset_step2.X.to(device)).cpu()

            latent_instrument_dataset_train = SimpleDataset(input_key="A",
                                                    output_key="Z",
                                                    input_data=train_dataset_step2.A.cpu(),
                                                    output_data=z_pred.cpu())
            
            with torch.no_grad():
                z_pred_val = model.encoder(val_dataset.X.to(device)).cpu()

            latent_instrument_dataset_val = SimpleDataset(input_key="A",
                                                        output_key="Z",
                                                        input_data=val_dataset.A.cpu(),
                                                        output_data=z_pred_val.cpu())
            
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
                save_dir=version_logger.log_dir,
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
            train_dataset_step3 = dataset.return_clone()
            train_dataset_step3.sample()

            
            with torch.no_grad():
                z_pred = model_step2(train_dataset_step3.A).cpu()
            causal_effect_dataset = SimpleDataset(input_key="Z",
                                                output_key="Y",
                                                input_data=z_pred.cpu(),
                                                output_data=train_dataset_step3.Y.cpu())
            
            with torch.no_grad():
                z_pred_val = model_step2(val_dataset.A).cpu()
            causal_effect_dataset_val = SimpleDataset(input_key="Z",
                                                    output_key="Y",
                                                    input_data=z_pred_val.cpu(),
                                                    output_data=val_dataset.Y.cpu())

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
                save_dir=version_logger.log_dir,
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

            y_ce = dataset.l(val_dataset.Z).cpu()
            with torch.no_grad():
                z_pred = model_step2(val_dataset.A).cpu()
                y_pred = model_step3(z_pred).cpu()
                mse = torch.mean((y_pred - y_ce)**2).item()
                print({"mse": mse})
                val_result[0]["mse"] = mse

            torch.save(mse, version_logger.log_dir + "/mse.pt")
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_folder', type=str, required=True, help='Path to main log folder')
    # Add other config arguments as needed
    args = parser.parse_args()
    cfg = OmegaConf.load("conf/config_imca_full.yaml")
    run_all_checkpoints(args.main_folder, cfg)
