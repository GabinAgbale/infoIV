import os 

import hydra 
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from data.data import SimpleDataset
from data.dsprites_data import dSpritesLoader, dSpritesFullDataset, dSpritesConditionalLoader
from data.utils import generate_full_rank_matrix
from data.target_fn import Linear_mixing, posX_scale_target
from model.image_model import ImageNCE
from model.models import NeuralFitter

from utils.utils import save_causal_effect_plot

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger




@hydra.main(config_path="../conf", config_name="config_dsprites_full", version_base=None)
def main(cfg: DictConfig) -> None:


    CHECKPOINT_PATH = cfg.checkpoint_path
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dsprites_path = 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    # Load dSprites data
    loader = dSpritesLoader(dsprites_path, device="cpu")
    # Use conditional loader to filter by shape=1
    conditions = {'shape': 1}
    confounder_name = 'posY'
    cond_loader = dSpritesConditionalLoader(loader, conditions=conditions)
    
    confounding_fn = lambda x: x - 0.5
    instrument_mixing = Linear_mixing(generate_full_rank_matrix(5, 8), device="cpu")
    target_fn = posX_scale_target

    # Create dataset
    train_dataset = dSpritesFullDataset(
        loader=cond_loader,
        confounder_name=confounder_name,
        instrument_mixing=instrument_mixing,
        target_fn=target_fn,
        confounding_fn=confounding_fn,
        device="cpu",
        test=False,
        confounding_strength=cfg.dataset.confounding_strength,
    )
    train_dataset.sample(cfg.n_train)

    val_dataset = dSpritesFullDataset(
        loader=cond_loader,
        confounder_name=confounder_name,
        instrument_mixing=instrument_mixing,
        target_fn=target_fn,
        confounding_fn=confounding_fn,
        device="cpu",
        test=False,
        confounding_strength=cfg.dataset.confounding_strength,
    )
    val_dataset.sample(cfg.n_val)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.num_workers)

    base_logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    base_logger.log_hyperparams(cfg)

    save_path = os.path.join(base_logger.log_dir, "mixing_funcs.pth")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #### 1st STEP : Train ImageNCE to estimate latent Z ####

    logger_step1 = TensorBoardLogger(
        save_dir=base_logger.log_dir,
        name="step1",
    )
    model = ImageNCE(
        auxiliary_dim=8,
        latent_dim=6,
        input_channels=1,
        hidden_dims=cfg.model.step1.hidden_dims,
        lambda_recon=cfg.loss.lambda_recon,
        temperature=cfg.loss.temperature,
        learning_rate=cfg.optimizer.step1.lr,
        weight_decay=cfg.optimizer.step1.weight_decay,
        optimizer=cfg.optimizer.step1.name,
        lr_scheduler=cfg.optimizer.step1.scheduler,
        activation=cfg.model.step1.activation,
        slope=cfg.model.step1.slope,
        dropout_rate=cfg.model.step1.dropout_rate,
    )

    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=[device.index] if device.type == "cuda" else None,
        max_epochs=cfg.trainer.step1.max_epochs,
        logger=logger_step1,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    ####  2nd STEP: TRAIN NEURAL FITTER TO LEARN ESTIMATED FEATURES BASED ON INSTRUMENT A ####
    dataset_step2 = dSpritesFullDataset(
        loader=cond_loader,
        confounder_name=confounder_name,
        instrument_mixing=instrument_mixing,
        target_fn=target_fn,
        confounding_fn=confounding_fn,
        device="cpu",
        test=False,
        confounding_strength=cfg.dataset.confounding_strength,
    )
    dataset_step2.sample(cfg.n_train)

    with torch.no_grad():
        Z_estimated = model.encoder(dataset_step2.X.unsqueeze(1).to("cpu"))
        Z_val = model.encoder(val_dataset.X.unsqueeze(1).to("cpu"))
    train_dataset_step2 = SimpleDataset(input_key='A', 
                                        output_key='Z',
                                        input_data=dataset_step2.A,
                                        output_data=Z_estimated,)
    train_dataloader_step2 = DataLoader(train_dataset_step2, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=cfg.num_workers)

    val_dataset_step2 = SimpleDataset(input_key='A',
                                      output_key='Z',
                                      input_data=val_dataset.A,
                                      output_data=Z_val,)
    val_dataloader_step2 = DataLoader(val_dataset_step2, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model_step2 = NeuralFitter(input_key="A",
                               output_key="Z",
                               input_dim=8,
                               output_dim=6,
                               hidden_dim=cfg.model.step2.hidden_dims,
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
    trainer_step2.fit(model_step2, train_dataloader_step2, val_dataloader_step2)
    val_result = trainer_step2.validate(model_step2, val_dataloader_step2, verbose=False)
    print({"val": val_result})

    ####  3rd STEP: ESTIMATE CAUSAL EFFECT ON ESTIMATED LATENTS ####
    dataset_step3 = dSpritesFullDataset(
        loader=cond_loader,
        confounder_name=confounder_name,
        instrument_mixing=instrument_mixing,
        target_fn=target_fn,
        confounding_fn=confounding_fn,
        device="cpu",
        test=False,
        confounding_strength=cfg.dataset.confounding_strength,
    )
    dataset_step3.sample(cfg.n_train)

    with torch.no_grad():
        Z_train = model_step2(dataset_step3.A.to("cpu"))
        Z_val = model_step2(val_dataset.A.to("cpu"))
    train_dataset_step3 = SimpleDataset(input_key="Z",
                                        output_key="Y",
                                        input_data=Z_train.cpu(),
                                        output_data=dataset_step3.Y.cpu())
    train_dataloader_step3 = DataLoader(train_dataset_step3, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=cfg.num_workers)

    val_dataset_step3 = SimpleDataset(input_key="Z",
                                      output_key="Y",
                                      input_data=Z_val.cpu(),
                                      output_data=val_dataset.Y.cpu())
    val_dataloader_step3 = DataLoader(val_dataset_step3, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=cfg.num_workers)


    model_step3 = NeuralFitter(input_key="Z",
                               output_key="Y",
                               input_dim=6,
                               output_dim=1,
                               hidden_dim=cfg.model.step3.hidden_dims,
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
    trainer_step3.fit(model_step3, train_dataloader_step3, val_dataloader_step3)
    val_result = trainer_step3.validate(model_step3, val_dataloader_step3, verbose=False)
    print({"val": val_result})

    print("Evaluate performances on causal effect estimation...")

    oos_dataset = dSpritesFullDataset(
        loader=cond_loader,
        confounder_name=confounder_name,
        instrument_mixing=instrument_mixing,
        target_fn=target_fn,
        confounding_fn=confounding_fn,
        device="cpu",
        test=True,
        confounding_strength=cfg.dataset.confounding_strength,
    )
    oos_dataset.sample(cfg.n_val)
    y_ce = oos_dataset.Y.unsqueeze(1)
    with torch.no_grad():
        z_pred = model_step2(oos_dataset.A)
        y_pred = model_step3(z_pred)
        mse = torch.mean((y_pred - y_ce)**2).item()
        print({"mse": mse})
        val_result[0]["mse"] = mse

    torch.save(mse, base_logger.log_dir + "/mse.pt")

    save_causal_effect_plot(y_pred.detach(), y_ce.detach(), base_logger)

if __name__ == "__main__":
    main()

