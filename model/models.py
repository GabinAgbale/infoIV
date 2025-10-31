import torch 
from torch import nn 
from torch.nn import functional as F


import pytorch_lightning as pl

from model.utils import MMRLoss
from model import utils
from model.components import Encoder, Decoder, LogisticRegression

from metrics import mcc 

class MMRae(pl.LightningModule):
    """
    Base Autoencoder model with MMR loss for invariance learning.

    Args:
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 latent_dim: int, 
                 lambda_recon: float = 0.1,
                 num_layers: int = 3,
                 lr: float = 1e-3,
                 sigma_kernel: float = 1.0):
        """
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layers.
            latent_dim (int): Dimension of the latent space.
            kernel (kernel.Kernel): Kernel function used for MMR loss.
            lambda_recon (float): Weight for the reconstruction term in the total loss.
            num_layers (int): Number of layers in the encoder and decoder (same).
            lr (float): Learning rate for the optimizer.
            sigma_kernel (float): Bandwidth for the Gaussian kernel used in MMR loss.
        """
        super(MMRae, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_layers)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        kernel = utils.GaussianKernel(sigma=sigma_kernel)
        self.mmr_loss_fn = MMRLoss(kernel)

        self.lambda_recon = lambda_recon

        self.lr = lr

    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer
    
    def _get_loss(self, batch, validation=False):
        x, a = batch['X'], batch['A']
        x_recon, z = self.forward(x)
        reconstruction_loss = F.mse_loss(x_recon, x, reduction='mean')
        invariance_loss = self.mmr_loss_fn(z, a)
        if validation:
            return reconstruction_loss, invariance_loss, utils.get_r2_score(z, batch['Z'])
        return reconstruction_loss, invariance_loss


    def training_step(self, batch, batch_idx):
        rec_loss, inv_loss = self._get_loss(batch)
        loss = self.lambda_recon * rec_loss + inv_loss
        self.log('train_invariance_loss', inv_loss, on_epoch=True)
        self.log('reconstruction_loss', rec_loss, prog_bar=True, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        rec_loss, inv_loss, r2 = self._get_loss(batch, True)
        loss = self.lambda_recon * rec_loss + inv_loss
        self.log('val_invariance_loss', inv_loss, prog_bar=True, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_r2', r2, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        rec_loss, inv_loss, r2 = self._get_loss(batch, True)
        loss = self.lambda_recon * rec_loss + inv_loss
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.log('test_r2', r2, prog_bar=True, on_epoch=True)
        return loss


class BaseAE(pl.LightningModule):
    """
    Base Autoencoder model.

    Args:
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 latent_dim: int, 
                 num_layers: int = 3,
                 lr: float = 1e-3):
        """
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layers.
            latent_dim (int): Dimension of the latent space.
            kernel (kernel.Kernel): Kernel function used for MMR loss.
            lambda_recon (float): Weight for the reconstruction term in the total loss.
            num_layers (int): Number of layers in the encoder and decoder (same).
            lr (float): Learning rate for the optimizer.
        """
        super(BaseAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_layers)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.lr = lr

    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer
    
    def _get_loss(self, batch, validation=False):
        x = batch['X']
        x_recon, z = self.forward(x)
        reconstruction_loss = F.mse_loss(x_recon, x, reduction='mean')
        if validation:
            return reconstruction_loss, utils.get_r2_score(z, batch['Z'])
        return reconstruction_loss


    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('reconstuction_loss', loss, prog_bar=True, on_epoch=True)   
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        rec_loss, r2 = self._get_loss(batch, True)
        self.log('val_loss',rec_loss, prog_bar=True, on_epoch=True)
        self.log('val_r2', r2, prog_bar=True, on_epoch=True)
        return rec_loss
    

    def test_step(self, batch, batch_idx):
        rec_loss, r2 = self._get_loss(batch, True)
        self.log('test_loss',rec_loss, prog_bar=True, on_epoch=True)
        self.log('test_r2', r2, prog_bar=True, on_epoch=True)
        return rec_loss



class BaseVAE(pl.LightningModule):
    """
    Base Variational Autoencoder model.
    """

    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 latent_dim: int, 
                 num_layers: int = 3,
                 lr: float = 1e-3,
                 lambda_recon: float = 1.0,
                 optimizer: str = "adam",
                 lr_scheduler: str = "none",
                 weight_decay: float = 0.0):
        """
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layers.
            latent_dim (int): Dimension of the latent space.
            num_layers (int): Number of layers in the encoder and decoder (same).
            lr (float): Learning rate for the optimizer.
            lambda_recon (float): Weight for the reconstruction term in the total loss.
            optimizer (str): Name of the optimizer to use.
            lr_scheduler (str): Name of the learning rate scheduler to use.
            weight_decay (float): Weight decay for the optimizer.
        """
        super(BaseVAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim * 2, num_layers) 
        # outputs both mu and logvar concatenated
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_layers)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.lr = lr
        self.lambda_recon = lambda_recon
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar


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
                                                                       patience=10,)
        # return [optimizer], [scheduler]
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    

    def _get_loss(self, batch, validation=False):
        x = batch['X']
        x_recon, z, mu, logvar = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss =  self.lambda_recon * recon_loss + kl_loss

        if validation:
            return loss, recon_loss, kl_loss, utils.get_r2_score(mu, batch['Z'])
        return loss


    def training_step(self, batch, batch_idx):

        loss = self._get_loss(batch)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    # def on_train_epoch_end(self):
    #     sch = self.lr_schedulers()
    #     sch.step(self.trainer.callback_metrics["train_loss"])

    def validation_step(self, batch, batch_idx):
        loss, rec_loss, kl_loss, r2 = self._get_loss(batch, True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_recon', rec_loss, prog_bar=True, on_epoch=True)
        self.log('val_kl', kl_loss, prog_bar=True, on_epoch=True)
        self.log('val_r2', r2, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, rec_loss, kl_loss, r2 = self._get_loss(batch, True)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.log('test_recon', rec_loss, prog_bar=True, on_epoch=True)
        self.log('test_kl', kl_loss, prog_bar=True, on_epoch=True)
        self.log('test_r2', r2, prog_bar=True, on_epoch=True)
        return loss
    

class iVAE(pl.LightningModule):
    """
    iVAE model as introduced in "Variational Autoencoders and Nonlinear ICA: A Unifying Framework" (Khemakhem et al., 2020).
    """

    def __init__(self, 
                 auxiliary_dim: int,
                 input_dim: int, 
                 hidden_dims: list, 
                 latent_dim: int, 
                 num_layers: int = 3,
                 lr: float = 1e-3,
                 optimizer: str = "adam",
                 lr_scheduler: str = "none",
                 weight_decay: float = 0.0,
                 activation: str = "lrelu",
                 n_train: int = 10000,
                 n_val: int = 1000,
                 a: float = 1,
                 b: float = 1,
                 c: float = 1,
                 d: float = 1,
                 slope: float = 0.1,
                 dropout_rate: float = 0.0,):
        """
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (list): Dimensions of the hidden layers.
            latent_dim (int): Dimension of the latent space.
            num_layers (int): Number of layers in the encoder and decoder (same).
            lr (float): Learning rate for the optimizer.
            optimizer (str): Name of the optimizer to use.
            lr_scheduler (str): Name of the learning rate scheduler to use.
            weight_decay (float): Weight decay for the optimizer.
            activation (str): Activation function to use in the prior network. either "lrelu", "tanh", or "xtanh"
            n_train (int): Number of training samples, used for the KL divergence computation.
            n_val (int): Number of validation samples, used for the KL divergence computation.
            a,b,c,d (float): coefficients for the ELBO loss terms.
            slope (float): slope for the leaky relu or xtanh activation functions.
            dropout_rate (float): dropout rate for the encoder network.
        """
        super().__init__()

        self.prior = utils.trainable_mlp(auxiliary_dim, hidden_dims, latent_dim, num_layers, activation=activation, slope=slope)

        self.encoder = Encoder(input_dim + auxiliary_dim, hidden_dims, latent_dim * 2, num_layers, activation=activation, slope=slope, dropout_rate=dropout_rate) 
        # outputs both mu and logvar concatenated

        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, num_layers, activation=activation, slope=slope, dropout_rate=dropout_rate)
        self.decoder_var = .1 * torch.ones(1)
        
        
        self.auxiliary_dim = auxiliary_dim  
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.n_train = n_train
        self.n_val = n_val

        self.a, self.b, self.c, self.d = a, b, c, d

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar.exp()

    def reparameterize(self, mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def forward(self, x, a):
        v_aux = self.prior(a).exp()
        xa = torch.cat([x, a], dim=1)
        mu, v = self.encode(xa)
        z = self.reparameterize(mu, v)
        x_recon = self.decoder(z)
        return x_recon, mu, v, z, v_aux


    def _get_loss(self, batch, validation=False):
        if validation:
            N = self.n_val
        else:
            N = self.n_train

        x, a = batch['X'], batch['A']
        f, g, v, z, l = self.forward(x, a)

        M, d_latent = z.size()

        norm = torch.log(torch.Tensor([M * N]).to(z.device))

        logpx = utils.log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = utils.log_normal(z, g, v).sum(dim=-1)
        logps_cu = utils.log_normal(z, None, l).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = utils.log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - norm
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False)  - norm
                   ).sum(dim=-1)

        elbo = -(self.a * logpx - self.b * (logqs_cux - logqs) - self.c * (logqs - logqs_i) - self.d * (logqs_i - logps_cu)).mean()
        if validation:
            return elbo, mcc.mean_corr_coef_np(z.detach().cpu().numpy(), batch['Z'].detach().cpu().numpy(), method='rdc')
        return elbo


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
                                                                       patience=10,)
        # return [optimizer], [scheduler]
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
    

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, corr = self._get_loss(batch, True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_mcc', corr, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, corr = self._get_loss(batch, True)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.log('test_mcc', corr, prog_bar=True, on_epoch=True)
        return loss

    
class AdditiveMLP(pl.LightningModule):
    """
    Additive MLP, estimating y from latent Z and controle variable V.

    E[Y | Z, V] = f(Z) + g(V)

    Trained using MSE loss.

    Args:
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 3,
                 lr: float = 1e-3,
                 activation: str = "lrelu",
                 lr_scheduler: str = "none",
                 optimizer: str = "adam",
                 weight_decay: float = 0.0,
                 slope: float = 0.1,
                 dropout_rate: float = 0.0):
        """
        Args:
            input_dim (int): Dimension of Z and V
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of Y.
            activation (str): Activation function to use.
            lr_scheduler (str): Learning rate scheduler to use, default is "none". also: "plateau"
            optimizer (str): Optimizer to use, default is "adam". also: "sgd"
            weight_decay (float): Weight decay for the optimizer, default is 0.0.
            slope (float): slope for the leaky relu activation function, default is 0.1.
            dropout_rate (float): dropout rate for the encoder network, default is 0.0.
        """
        super(AdditiveMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.causal_effect = utils.trainable_mlp(input_dim, hidden_dim, output_dim, num_layers, activation=activation, slope=slope, dropout=dropout_rate)
        self.control_function = utils.trainable_mlp(input_dim, hidden_dim, output_dim, num_layers, activation=activation, dropout=dropout_rate)
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def forward(self, z, v):
        f_z = self.causal_effect(z)
        g_v = self.control_function(v)
        return f_z + g_v
    
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
                                                                       patience=10,)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        y, z, v = batch['Y'], batch['Z'], batch['V']
        y_pred = self.forward(z, v)
        loss = F.mse_loss(y_pred, y)
        self.log('train_mse', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, z, v = batch['Y'], batch['Z'], batch['V']
        y_pred = self.forward(z, v)
        loss = F.mse_loss(y_pred, y)
        self.log('val_mse', loss, prog_bar=True, on_epoch=True)

        return loss        
    
    def test_step(self, batch, batch_idx):
        y, z, v = batch['Y'], batch['Z'], batch['V']
        y_pred = self.forward(z, v)
        loss = F.mse_loss(y_pred, y)
        self.log('test_mse', loss, prog_bar=True)
        return loss
    


class ContrastiveClassifierModel(pl.LightningModule):
    """
    Architecture : Encoder + Logistic Head 

    Args:
        input_dim (int): Dimension of the input data.
        auxiliary_dim (int): Dimension of the auxiliary data.
        hidden_dim_encoder (int): Dimension of the encoder hidden layers.
        num_layers_encoder (int): Number of encoder hidden layers.
        hidden_dim_head (int): Dimension of the logistic head hidden layers.
        num_layers_head (int): Number of head hidden layers.
        latent_dim (int): Dimension of the latent space.
        lr (float): Learning rate for the optimizer.
        lambda_recon (float): Weight for the reconstruction term in the total loss.
        activaton (str): Activation function to use, default is "lrelu"
    """

    def __init__(self, 
                 input_dim: int,
                 auxiliary_dim: int,
                 hidden_dims_encoder: int = 32,
                 num_layers_encoder: int = 3,
                 hidden_dim_head: int = 32,
                 num_layers_head: int = 3,
                 latent_dim: int = 32,
                 lr: float = 1e-3,
                 lambda_recon: float = 1,
                 optimizer: str = "adam",
                 lr_scheduler: str = "none",
                 weight_decay: float = 0.0,
                 activation: str = "leaky_relu",
                 slope: float = 0.1,
                 dropout_rate: float = 0.1) -> None:
        super(ContrastiveClassifierModel, self).__init__()

 
        self.input_dim = input_dim
        self.auxiliary_dim = auxiliary_dim
        self.hidden_dims_encoder = hidden_dims_encoder
        self.num_layers_encoder = num_layers_encoder
        self.hidden_dim_head = hidden_dim_head
        self.num_layers_head = num_layers_head
        self.latent_dim = latent_dim
        self.lr = lr
        self.lambda_recon = lambda_recon
        
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.activation = activation

        self.slope = slope 

        self.encoder = Encoder(input_dim=input_dim, 
                               hidden_dims=hidden_dims_encoder, 
                               latent_dim=latent_dim, 
                               num_layers=num_layers_encoder,
                               activation=activation,
                               slope=slope,
                               dropout_rate=dropout_rate)
        
        self.decoder = Decoder(latent_dim=latent_dim,
                               hidden_dims=hidden_dims_encoder, 
                               output_dim=input_dim, 
                               num_layers=num_layers_encoder,
                               activation=activation,
                               slope=slope,
                               dropout_rate=dropout_rate)
        
        self.head = LogisticRegression(input_dim=latent_dim+auxiliary_dim, 
                                       num_layers=num_layers_head, 
                                       hidden_dim=hidden_dim_head)



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
                                                                       patience=10,)
        
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

  
      
    def forward(self, x, aux):
        z = self.encoder(x)
        x_recon = self.decoder(z)

        z_aux = torch.cat([z, aux], dim=1)
        logits = self.head(z_aux)

        return x_recon, z, logits
    
    
    def _get_loss(self, batch, validation=False):
        x, aux, z_true, c = batch['X'], batch['A'], batch['Z'], batch['C']
        x_recon, z_pred, logits = self.forward(x, aux)
        reconstruction_loss = F.mse_loss(x_recon, x, reduction='mean')
        contrastive_loss = F.cross_entropy(logits, c)
        
        if validation:
            corr = mcc.mean_corr_coef_np(z_pred.detach().cpu().numpy(), z_true.detach().cpu().numpy(), method="rdc")
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == c).float().mean()
            return reconstruction_loss, contrastive_loss, utils.get_r2_score(z_pred, z_true), accuracy, corr
        return reconstruction_loss, contrastive_loss
    
    def training_step(self, batch, batch_idx):
        rec_loss, contrastive_loss = self._get_loss(batch)
        loss = self.lambda_recon * rec_loss +  contrastive_loss
        self.log('train_reconstruction_loss', rec_loss, prog_bar=True, on_epoch=True)
        self.log('train_contrastive_loss', contrastive_loss, prog_bar=True, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        rec_loss, contrastive_loss, r2, accuracy, corr = self._get_loss(batch, validation=True)
        loss = self.lambda_recon * rec_loss + contrastive_loss
        self.log('test_reconstruction_loss', rec_loss, prog_bar=True)
        self.log('test_contrastive_loss', contrastive_loss, prog_bar=True)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_r2', r2, prog_bar=True, on_epoch=True)
        self.log('test_accuracy', accuracy, prog_bar=True, on_epoch=True)
        self.log('test_mcc', corr, prog_bar=True, on_epoch=True)
        return loss
     
    def validation_step(self, batch, batch_idx):
        rec_loss, contrastive_loss, r2, accuracy, corr = self._get_loss(batch, validation=True)
        loss = self.lambda_recon * rec_loss +  contrastive_loss
        self.log('val_reconstruction_loss', rec_loss, prog_bar=True, on_epoch=True)
        self.log('val_contrastive_loss', contrastive_loss, prog_bar=True, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_r2', r2, prog_bar=True, on_epoch=True)
        self.log('val_mcc', corr, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', accuracy, prog_bar=True, on_epoch=True)
    
        return loss



class ContrastiveOracle(pl.LightningModule):
    
    """
    Oracle Model performing contrastive learning on ground truth latent
    
    Architecture: Logistic Head 

    Args:
        latent_dim (int): Dimension of the latent space.
        auxiliary_dim (int): Dimension of the auxiliary data.
        hidden_dim_head (int): Dimension of the logistic head hidden layers.
        num_layers_head (int): Number of head hidden layers.
        lr (float): Learning rate for the optimizer.
    """

    def __init__(self, 
                 auxiliary_dim: int,
                 latent_dim: int,
                 hidden_dim_head: int = 32,
                 num_layers_head: int = 3,
                 lr: float = 1e-3):
        super(ContrastiveOracle, self).__init__()

        self.head = LogisticRegression(input_dim=latent_dim+auxiliary_dim,
                                       num_layers=num_layers_head, 
                                       hidden_dim=hidden_dim_head)
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer
    
    def forward(self, z, aux):
        z_aux = torch.cat([z, aux], dim=1)
        logits = self.head(z_aux)
        return logits
    
    def _get_loss(self, batch):
        aux, z_true, c = batch['A'], batch['Z'], batch['C']
        logits = self.forward(z_true, aux)
        contrastive_loss = F.cross_entropy(logits, c)
        return contrastive_loss

    def training_step(self, batch, batch_idx):
        contrastive_loss = self._get_loss(batch)
        self.log('train_contrastive_loss', contrastive_loss, prog_bar=True, on_epoch=True)
        return contrastive_loss

    def validation_step(self, batch, batch_idx):
        contrastive_loss = self._get_loss(batch)
        self.log('val_contrastive_loss', contrastive_loss, prog_bar=True, on_epoch=True)
        
        aux, z_true, c = batch['A'], batch['Z'], batch['C']
        logits = self.forward(z_true, aux)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == c).float().mean()
        self.log('val_accuracy', accuracy, prog_bar=True, on_epoch=True)
        return contrastive_loss

    def test_step(self, batch, batch_idx):
        aux, z_true, c = batch['A'], batch['Z'], batch['C']
        logits = self.forward(z_true, aux)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == c).float().mean()
        self.log('test_accuracy', accuracy, prog_bar=True)
        return accuracy
    

class InfoNCEModel(pl.LightningModule):
    """
    InfoNCE model.
    Architecture: Encoder + Decoder 
    Trained on reconstruction loss and InfoNCE loss.
    InfoNCE loss is computed on the latent space and auxiliary data.
    
    Args:
        input_dim (int): Dimension of the input data.
        auxiliary_dim (int): Dimension of the auxiliary data.
        hidden_dim(int): Dimension of the encoder/decoder hidden layers.
        num_layers (int): Number of encoder/decoder hidden layers.
        latent_dim (int): Dimension of the latent space.
        lr (float): Learning rate for the optimizer.
        lambda_recon (float): Weight for the reconstruction term in the total loss.
        temperature (float): Temperature parameter for the InfoNCE loss.
        optimizer (str): Optimizer to use, default is "adam". also: "sgd"
        weight_decay (float): Weight decay for the optimizer, default is 0.0.
        lr_scheduler (str): Learning rate scheduler to use, default is "none". also: "plateau"
        activation (str): Activation function to use, default is "relu". also: "gelu"
    """
    
    def __init__(self, 
                 input_dim: int,
                 auxiliary_dim: int,
                 latent_dim: int,
                 hidden_dims: list[int] = [32, 64, 128],
                 num_layers: int = 3,
                 lr: float = 1e-3,
                 lambda_recon: float = 1,
                 temperature: float = 0.1,
                 optimizer: str = "adam",
                 weight_decay: float = 0.0,
                 lr_scheduler: str = "none",
                 activation: str = "lrelu",
                 compute_r2: bool = False,
                 dropout_rate: float = 0.0,
                 slope: float = 0.1):
        super(InfoNCEModel, self).__init__()
        
        self.encoder = Encoder(input_dim=input_dim, 
                               hidden_dims=hidden_dims, 
                               latent_dim=latent_dim, 
                               num_layers=num_layers,
                               activation=activation,
                               dropout_rate=dropout_rate,
                               slope=slope)
        
        self.decoder = Decoder(latent_dim=latent_dim,
                               hidden_dims=hidden_dims[::-1], 
                               output_dim=input_dim, 
                               num_layers=num_layers,
                               activation=activation,
                               dropout_rate=dropout_rate,
                               slope=slope)
        
        self.W = nn.Linear(auxiliary_dim, latent_dim, bias=False)
    
        self.temperature = temperature
        self.lambda_recon = lambda_recon
        
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler

        self.compute_r2 = compute_r2

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
                                                                       patience=10,)
        
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


    def forward(self, x, aux):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        z_aux = self.W(aux)
        return x_recon, z, z_aux
    

    def _get_loss(self, batch, validation=False):
        x, aux = batch['X'], batch['A']
        x_recon, z, z_aux  = self.forward(x, aux)
        nceloss = utils.infoNCEloss(z, z_aux, self.temperature)
        reconstruction_loss = F.mse_loss(x_recon, x, reduction='mean')
        if validation:
            z_gt = batch['Z']
            if self.compute_r2:
                return reconstruction_loss, nceloss, utils.get_r2_score(z, z_gt), mcc.mean_corr_coef_np(z.detach().cpu().numpy(), z_gt.detach().cpu().numpy(), method="rdc")
            return reconstruction_loss, nceloss, mcc.mean_corr_coef_np(z.detach().cpu().numpy(), z_gt.detach().cpu().numpy(), method="rdc")
        return reconstruction_loss, nceloss
    

    def training_step(self, batch, batch_idx):
        rec_loss, nceloss = self._get_loss(batch)
        loss = self.lambda_recon * rec_loss + nceloss
        self.log('train_reconstruction_loss', rec_loss, prog_bar=True, on_epoch=True)
        self.log('train_nceloss', nceloss, prog_bar=True, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        if self.compute_r2:
            rec_loss, nceloss, r2_score, mcc_coeff = self._get_loss(batch, validation=True)
        else:
            rec_loss, nceloss, mcc_coeff = self._get_loss(batch, validation=True)
            r2_score = torch.tensor(float('nan'))
        loss = self.lambda_recon * rec_loss + nceloss
        # Log learning rate if using a scheduler
        if self.trainer is not None and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, prog_bar=True, on_epoch=True)
        self.log('val_reconstruction_loss', rec_loss, prog_bar=True, on_epoch=True)
        self.log('val_mcc', mcc_coeff, prog_bar=True, on_epoch=True)
        self.log('val_nceloss', nceloss, prog_bar=True, on_epoch=True)
        self.log('val_r2', r2_score, prog_bar=True, on_epoch=True)
        return loss
    

    def test_step(self, batch, batch_idx):
        if self.compute_r2:
            rec_loss, nceloss, r2_score, mcc_coeff = self._get_loss(batch, validation=True)
        else:
            rec_loss, nceloss, mcc_coeff = self._get_loss(batch, validation=True)
            r2_score = torch.tensor(float('nan'))
        loss = self.lambda_recon * rec_loss + nceloss
        self.log('test_reconstruction_loss', rec_loss, prog_bar=True)
        self.log('test_nceloss', nceloss, prog_bar=True)
        self.log('test_r2', r2_score, prog_bar=True, on_epoch=True)
        self.log('test_mcc', mcc_coeff, prog_bar=True, on_epoch=True)
        return loss
    

class OLSModel(pl.LightningModule):
    """
    Ordinary Least Squares model.
    
    Args:
        input_dim (int): Dimension of the input data.
        output_dim (int): Dimension of the output data.
        hidden_dim (int): Dimension of the hidden layers.
        num_layers (int): Number of hidden layers.
        lr (float): Learning rate for the optimizer.
        type (str): Type of the model, either "observed" or "latent".
    """
    
    def __init__(self, input_dim: int, type: str, output_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1, lr: float = 1e-3):
        super(OLSModel, self).__init__()
        self.mlp = utils.trainable_mlp(input_dim, hidden_dim, output_dim, n_layers=num_layers)
        self.lr = lr

        valid_types = {"observed": "X", "latent": "Z", "2s": "Z_aux", "instrument": "A"}
        if type not in valid_types:
            raise ValueError(f"Invalid type: {type}")
        self.input_key = valid_types[type]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer
    
    def forward(self, x):
        return self.mlp(x)
    
    def _get_loss(self, batch):
        x, y = batch[self.input_key], batch['Y']
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train_mse', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('val_mse', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('test_mse', loss, prog_bar=True, on_epoch=True)
        return loss

class LinearControlFunction(pl.LightningModule):
    """
    Linear Instrumental Variable model for linear regression with two inputs: 
    
    input: Z, V (both of dimZ)
    output: Y (dimY)

    Args:
        input_dim (int): Dimension of the input data (dimZ). 
        output_dim (int): Dimension of the output data.
    """


    def __init__(self, input_dim: int, output_dim: int = 1, lr: float = 0.01):
        super(LinearControlFunction, self).__init__()
        self.causal_effect = nn.Linear(input_dim, output_dim)
        self.control_function = nn.Linear(input_dim, output_dim)
        self.lr = lr
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer
    
    def forward(self, z, v):
        return self.causal_effect(z) + self.control_function(v)
    
    def _get_loss(self, batch):
        z, v, y = batch['Z'], batch['V'], batch['Y']
        y_pred = self.forward(z, v)
        loss = F.mse_loss(y_pred, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train_mse', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('val_mse', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y, z, v = batch['Y'], batch['Z'], batch['V']
        y_pred = self.forward(z, v)
        loss = F.mse_loss(y_pred, y)
        self.log('test_mse', loss, prog_bar=True)
        return loss
    

class LinearRegression(pl.LightningModule):
    """
    Linear Instrumental Variable model for linear regression with two inputs: 
    
    input: Z, V (both of dimZ)
    output: Y (dimY)

    Args:
        input_dim (int): Dimension of the input data (dimZ). 
        output_dim (int): Dimension of the output data.
        lambda_lasso (float): Regularization strength for Lasso regression.
        lr (float): Learning rate for the optimizer.
    """


    def __init__(self, input_dim: int, lambda_lasso: float = 0.1, output_dim: int = 1, lr: float = 0.01):
        super(LinearRegression, self).__init__()
        self.causal_effect = nn.Linear(input_dim, output_dim)
        self.lr = lr
        self.lambda_lasso = lambda_lasso 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer
    
    def forward(self, z):
        return self.causal_effect(z)
    
    def _get_loss(self, batch):
        z, y = batch['Z_aux'], batch['Y']
        y_pred = self.forward(z)
        loss = F.mse_loss(y_pred, y) + self.lambda_lasso * (torch.norm(self.causal_effect.weight, p=1))
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train_mse', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('val_mse', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        z, y = batch['Z'], batch['Y']
        y_pred = self.forward(z)
        loss = F.mse_loss(y_pred, y)
        self.log('test_mse', loss, prog_bar=True)
        return loss
    

class NeuralFitter(pl.LightningModule):
    """
    Simple Neural Fitter for regression tasks, with mse loss.

    Args:
        input_dim (int): Dimension of the input data. 
        output_dim (int): Dimension of the output data.
        hidden_dim (int): Dimension of the hidden layers.
        num_layers (int): Number of hidden layers.
        lr (float): Learning rate for the optimizer.
        optimizer (str): Optimizer to use, default is "adam". also: "sgd"
        weight_decay (float): Weight decay for the optimizer, default is 0.0.
        lr_scheduler (str): Learning rate scheduler to use, default is "none". also: "plateau"
        activation (str): Activation function to use, default is "lrelu". also: "tanh"
    """

    def __init__(self, input_key: str,
                 output_key: str,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: list[int],
                 num_layers: int,
                 optimizer: str = "adam",
                 weight_decay: float = 0.0,
                 lr: float = 0.01,
                 lr_scheduler: str = "none",
                 activation: str = "leaky_relu",
                 dropout: float = 0.0,
                 slope: float = 0.1):
        super(NeuralFitter, self).__init__()
        
        self.model = utils.trainable_mlp(input_dim, 
                                         hidden_dims=hidden_dim, 
                                         output_dim=output_dim, 
                                         n_layers=num_layers, 
                                         activation=activation,
                                         dropout=dropout,
                                         slope=slope)

        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr = lr

        self.input_key = input_key
        self.output_key = output_key

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
                                                                       patience=10,)
        
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": f"train_mse_{self.output_key}"}

    
    def forward(self, z):
        return self.model(z)
    
    def _get_loss(self, batch):
        y, x = batch[self.output_key], batch[self.input_key]
        z_pred = self.forward(x)
        loss = F.mse_loss(z_pred, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log(f'train_mse_{self.output_key}', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log(f'val_mse_{self.output_key}', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log(f'test_mse_{self.output_key}', loss, prog_bar=True)
        return loss