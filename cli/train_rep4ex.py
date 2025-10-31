import os 

import hydra 
from omegaconf import DictConfig

import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import linear_regression
from model.models import MMRae, InfoNCEModel, AdditiveMLP, NeuralFitter

from data.data import RepExSCMDataset, SimpleDataset, DoubleDataset


from utils.utils import save_causal_effect_plot


seed_everything(42)


@hydra.main(config_path="../conf", config_name="config_iv", version_base=None)
def main(cfg: DictConfig) -> None:


    CHECKPOINT_PATH = cfg.checkpoint_path
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
                              noise_distribution="uniform",)
    dataset.generate_mixing_funcs()
    
    dataset.sample(cfg.dataset.gamma_train)
    train_loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    
    val_dataset = dataset.return_clone()
    val_dataset.sample(cfg.dataset.gamma_train)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
    
    test_dataset = dataset.return_clone()
    test_dataset.sample(cfg.dataset.gamma_test)
  
    logger = TensorBoardLogger(
        save_dir=cfg.trainer.root_dir,
        name=cfg.expe_name,
        log_graph=True,
    )

    logger.log_hyperparams(cfg)

    save_path = os.path.join(logger.log_dir, "mixing_funcs.pth")
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset.save_mixing_funcs(save_path)

    
    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.root_dir,
        accelerator=device.type,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.step1.max_epochs,
        logger=logger,
        # callbacks=[PeriodicCallback(frequency=cfg.trainer.test_freq)]
        )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"rep4ex_{cfg.expe_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        match cfg.type:
           case "mmr":
               model = MMRae.load_from_checkpoint(pretrained_filename)
           case "nce":
               model = InfoNCEModel.load_from_checkpoint(pretrained_filename)
           case "oracle":
                pass 
           case _:
                raise ValueError("Unknown model type.")
    
    else:
        match cfg.type:
           case "nce":
               model = InfoNCEModel(input_dim=cfg.dataset.dimX, 
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
                             compute_r2=True,
                             )
               trainer.fit(model, train_loader, val_loader)
               val_result = trainer.validate(model, val_loader, verbose=False)
               r2_score = val_result[0].get('test_r2', None)  
               torch.save(r2_score, os.path.join(logger.log_dir, "val_r2.pth"))


               print("Autoencoder training done.")
               result = {"val": val_result}
               print(result)


           case "mmr":
                model = MMRae(input_dim=cfg.dataset.dimX, 
                                    hidden_dim=cfg.model.step1.hidden_dim, 
                                    latent_dim=cfg.dataset.dimZ,
                                    lambda_recon=cfg.loss.l,
                                    num_layers=cfg.model.step1.num_layers,
                                    lr=cfg.optimizer.step1.lr,)

                trainer.fit(model, train_loader, val_loader)

                val_result = trainer.validate(model, val_loader, verbose=False)

                print("Autoencoder training done.")
                result = {"val": val_result}
                r2_score = val_result[0].get('test_r2', None)  
                torch.save(r2_score, os.path.join(logger.log_dir, "val_r2.pth"))
                print(result)


    if cfg.type == "oracle":
        Z_pred_train = dataset.Z
    else:
        with torch.no_grad():
            model = model.to(device)
            model.freeze()
            Z_pred_train = model.encoder(dataset.X)

    #### 2nd step fit neuralfitter to learn E[Z|A] ####
    W, b = linear_regression(dataset.A, Z_pred_train)


    #### 3rd step: fit additive MLP to learn causal effect on estimated latents ####
    step3_train_dataset = dataset.return_clone()
    step3_train_dataset.sample(cfg.dataset.gamma_train)

    if cfg.type == "oracle":
        Z_pred = step3_train_dataset.Z
        V_pred = step3_train_dataset.V
        Z_val = val_dataset.Z
        V_val = val_dataset.V
    else:
        with torch.no_grad():
            Z_pred = model.encoder(step3_train_dataset.X)
            V_pred = Z_pred - step3_train_dataset.A @ W - b

            Z_val = model.encoder(val_dataset.X)
            V_val = Z_val - val_dataset.A @ W - b

    double_dataset_train = DoubleDataset(input1_key="Z",
                                   input2_key="V",
                                   output_key="Y",
                                   input1_data=Z_pred.cpu(),
                                   input2_data=V_pred.cpu(),
                                   output_data=step3_train_dataset.Y.cpu())
    

    double_dataset_train_loader = DataLoader(double_dataset_train, batch_size=cfg.dataset.batch_size, shuffle=True)

    double_dataset_val_dataset = DoubleDataset(input1_key="Z",
                                    input2_key="V",
                                    output_key="Y",
                                    input1_data=Z_val.cpu(),
                                    input2_data=V_val.cpu(),
                                    output_data=val_dataset.Y.cpu())

    double_dataset_val_loader = DataLoader(double_dataset_val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    model_step3 = AdditiveMLP(input_dim=cfg.dataset.dimZ,
                                output_dim=1,
                                hidden_dim=cfg.model.step3.hidden_dim,
                                num_layers=cfg.model.step3.num_layers,
                                lr=cfg.optimizer.step3.lr,
                                weight_decay=cfg.optimizer.step3.weight_decay,
                                activation=cfg.model.step3.activation,
                                optimizer=cfg.optimizer.step3.name,
                                lr_scheduler=cfg.optimizer.step3.scheduler,
                                dropout_rate=cfg.model.step3.dropout_rate,
                                slope=cfg.model.step3.slope,
                                )
    
    logger_step3 = TensorBoardLogger(
        save_dir=logger.log_dir,
        name="step3",
        log_graph=True,
    )

    trainer_step3 = pl.Trainer(
        accelerator=device.type,
        devices=[device.index] if device.type == "cuda" else None,
        max_epochs=cfg.trainer.step3.max_epochs,
        logger=logger_step3,
    )

    print("Training additive MLP to learn causal effect on estimated latents...")
    trainer_step3.fit(model_step3, double_dataset_train_loader, double_dataset_val_loader)
    val_result = trainer_step3.validate(model_step3, double_dataset_val_loader, verbose=False)


    print("Additive MLP training done.")
    result = {"val": val_result}
    print(result)

    print("Perfom evaluation on interventioned data...")
    # sample intervention data
    do_A = 2 *torch.rand(len(test_dataset), cfg.dataset.dimA, device=device) * cfg.dataset.gamma_test - cfg.dataset.gamma_test

    mses_intervention = []
    
    Y_do_true = val_dataset.compute_intervention(do_A).cpu()
    Y_preds = []
    with torch.no_grad():
        for i, a in enumerate(do_A):        
            
            if cfg.type == "oracle":
                Z = val_dataset.Z
                V = val_dataset.V

                input1 = val_dataset._M0.to(device) @ a + V
                y_preds = model_step3.causal_effect(input1.cpu()) - (model_step3.causal_effect(Z.cpu()) - val_dataset.Y.cpu())

            else:
                Z = model.encoder(val_dataset.X)
                V = Z - val_dataset.A @ W - b
                input1 = a @ W + b + V

                y_preds = model_step3.causal_effect(input1.cpu()) - (model_step3.causal_effect(Z.cpu()) - val_dataset.Y.cpu())

            y_pred = y_preds.mean()
            mse = (y_pred - Y_do_true[i])**2

            Y_preds.append(y_pred.item())
            
            mses_intervention.append(mse.item())
        
    save_causal_effect_plot(torch.Tensor(Y_preds).unsqueeze(1), Y_do_true, logger )
    print("Mean squared errors for interventioned data:", np.mean(mses_intervention))
    torch.save(np.mean(mses_intervention), os.path.join(logger.log_dir, "test_mse_intervention.pth"))

if __name__ == "__main__":
    main()
