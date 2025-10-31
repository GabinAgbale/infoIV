import io 
import PIL.Image as Image
import random 
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model import utils
from model.components import ConvEncoder, ConvDecoder

from metrics.mcc import mean_corr_coef_np

class ImageNCE(pl.LightningModule):
    def __init__(self, 
                 auxiliary_dim: int,
                 latent_dim: int = 64,
                 hidden_dims: list = [32, 64, 128, 256],
                 learning_rate: float = 1e-3,
                 input_channels: int = 1,
                 activation: str = 'leaky_relu',
                 weight_decay: float = 1e-5,
                 optimizer: str = "adam",
                 lr_scheduler: str = "none",
                 lambda_recon: float = 0.,
                 temperature: float = 0.7,
                 log_every_n_epochs: int = 10,
                 dropout_rate: float = 0.1,
                 slope: float = 0.2):
        super(ImageNCE, self).__init__()

        self.encoder = ConvEncoder(in_channels=input_channels, 
                                   hidden_dims=hidden_dims, 
                                   latent_dim=latent_dim, 
                                   activation=activation,
                                   dropout_rate=dropout_rate,
                                   slope=slope)

        self.decoder = ConvDecoder(out_channels=input_channels, 
                                   hidden_dims=hidden_dims[::-1], 
                                   latent_dim=latent_dim, 
                                   activation=activation,
                                   dropout_rate=dropout_rate,
                                   slope=slope,
                                   feature_shape=self.encoder.feature_shape)

        self.W = nn.Linear(auxiliary_dim, latent_dim, bias=False)


        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.lambda_recon = lambda_recon
        self.temperature = temperature

        self.log_every_n_epochs = log_every_n_epochs

    def forward(self, x, aux):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        z_aux = self.W(aux)
        return x_recon, z, z_aux


    def _get_loss(self, batch, validation=False):
        x, aux, z_gt = batch['X'], batch['A'], batch['Z']
        x_recon, z, z_aux  = self.forward(x, aux)
        nceloss = utils.infoNCEloss(z, z_aux, self.temperature)
        reconstruction_loss = F.mse_loss(x_recon, x, reduction='mean')
        if validation:
            mcc = mean_corr_coef_np(z.detach().cpu().numpy(), z_gt.detach().cpu().numpy(), method="rdc")
            return reconstruction_loss, nceloss, mcc
        return reconstruction_loss, nceloss


    def training_step(self, batch, batch_idx):
        rec_loss, nceloss = self._get_loss(batch)
        loss = self.lambda_recon * rec_loss + nceloss
        self.log('train_reconstruction_loss', rec_loss, prog_bar=True, on_epoch=True)
        self.log('train_nceloss', nceloss, prog_bar=True, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        rec_loss, nceloss, mcc = self._get_loss(batch, validation=True)
        loss = self.lambda_recon * rec_loss + nceloss
        
        if batch_idx == 0:
            self.example_batch = batch

        # Log learning rate if using a scheduler
        if self.trainer is not None and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, prog_bar=True, on_epoch=True)
        self.log('val_reconstruction_loss', rec_loss, prog_bar=True, on_epoch=True)
        self.log('val_nceloss', nceloss, prog_bar=True, on_epoch=True)
        self.log('val_mcc', mcc, prog_bar=True, on_epoch=True)
        return loss
    

    def configure_optimizers(self):
        match self.optimizer:
            case "adam":
                optimizer = torch.optim.Adam(self.parameters(), 
                                             lr=self.lr, 
                                             betas=(0.9, 0.999), 
                                             weight_decay=self.weight_decay)
            case "sgd":
                optimizer = torch.optim.SGD(self.parameters(),
                                            lr=self.lr, 
                                            weight_decay=self.weight_decay)
            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        match self.lr_scheduler:
            case "none":
                return optimizer
            
            case "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                       mode='min', 
                                                                       factor=0.5, 
                                                                       patience=10)
        
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    

    def on_validation_epoch_end(self):
        if self.current_epoch % self.log_every_n_epochs != 0:
            return
        
        batch = getattr(self, "example_batch", None)
        if batch is None:
            return

        x, aux, z_gt = batch['X'], batch['A'], batch['Z']
        x_recon, _, _ = self.forward(x, aux)

        # pick a random index
        idx = random.randint(0, x.size(0) - 1)
        img_true = x[idx].detach().cpu()
        img_recon = x_recon[idx].detach().cpu()
        z_val = z_gt[idx].detach().cpu().numpy()

        # plot with matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].imshow(img_true.squeeze(), cmap="gray")
        axes[0].set_title(f"GT (Z={z_val})")
        axes[0].axis("off")
        axes[1].imshow(img_recon.squeeze(), cmap="gray")
        axes[1].axis("off")

        # convert to tensor for logging
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = torchvision.transforms.ToTensor()(img)
        plt.close(fig)

        if hasattr(self.logger.experiment, "add_image"):
            self.logger.experiment.add_image("random_reconstruction", img_tensor, self.current_epoch)