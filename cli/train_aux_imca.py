import os 

import hydra 
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


from model.models import ContrastiveClassifierModel
#from model.utils import PeriodicCallback

from data.imca_data import IMCADataset, ContrastiveIMCADataset


@hydra.main(config_path="../conf", config_name="config_aux_imca", version_base=None)
def main(cfg: DictConfig) -> None:


    CHECKPOINT_PATH = cfg.checkpoint_path
    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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
    contrastive_dataset = ContrastiveIMCADataset(dataset, n_contrastive_pairs=cfg.dataset.n_contrastive_pairs, seed=cfg.seed)
    train_loader = DataLoader(contrastive_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)

    val_dataset = dataset.return_clone()
    val_dataset.n = cfg.dataset.n_val
    val_dataset.sample()
    contrastive_val_dataset = ContrastiveIMCADataset(val_dataset, n_contrastive_pairs=cfg.dataset.n_contrastive_pairs, seed=cfg.seed)
    val_loader = DataLoader(contrastive_val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    test_dataset = dataset.return_clone()
    test_dataset.n = cfg.dataset.n_val
    test_dataset.sample()
    contrastive_test_dataset = ContrastiveIMCADataset(test_dataset, n_contrastive_pairs=cfg.dataset.n_contrastive_pairs, seed=cfg.seed)
    test_loader = DataLoader(contrastive_test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)
   
   
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
        model = ContrastiveClassifierModel.load_from_checkpoint(pretrained_filename)
    else:
        model = ContrastiveClassifierModel(input_dim=cfg.dataset.dimX, 
                                           hidden_dims_encoder=cfg.model.encoder.hidden_dims,
                                           num_layers_encoder=cfg.model.encoder.num_layers,
                                           hidden_dim_head=cfg.model.head.hidden_dim,
                                           num_layers_head=cfg.model.head.num_layers,
                                           latent_dim=cfg.dataset.dimZ,
                                           auxiliary_dim=cfg.dataset.dimA,
                                           lambda_recon=cfg.loss.l,
                                           lr=cfg.optimizer.lr,
                                           optimizer=cfg.optimizer.name,
                                           lr_scheduler=cfg.optimizer.scheduler,
                                           weight_decay=cfg.optimizer.weight_decay,
                                           activation=cfg.model.encoder.activation,
                                           dropout_rate=cfg.model.encoder.dropout_rate,)

        trainer.fit(model, train_loader, val_loader)

    val_result = trainer.validate(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)

    mcc = test_result[0].get('test_mcc', None)
    torch.save(mcc, os.path.join(logger.log_dir, "test_mcc.pth"))

    result = {"val": val_result}
    print(result)


if __name__ == "__main__":
    main()

