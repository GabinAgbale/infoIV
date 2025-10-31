import torch 
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score

from model import components

import pytorch_lightning as pl 

import matplotlib.pyplot as plt

class GenerateCallback(pl.Callback):

    def __init__(self, A_logs, Y_logs, Y_preds, every_n_epochs=50):
        super().__init__()
        self.A_logs = A_logs 
        self.Y_logs = Y_logs
        self.Y_preds = Y_preds
        self.every_n_epochs = every_n_epochs 

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            A = torch.cat(self.log_A, dim=0).numpy().flatten()
            Y = torch.cat(self.log_Y, dim=0).numpy().flatten()

            fig, ax = plt.subplots()
            ax.scatter(A, Y, alpha=0.5)
            ax.set_xlabel("A")
            ax.set_ylabel("Y")
            ax.set_title("Y vs A (Test Set)")

        self.logger.experiment.add_figure("Y_vs_A", fig, global_step=self.current_epoch)

        plt.close(fig)
    
    
def trainable_mlp(input_dim, hidden_dims, output_dim, n_layers, activation="tanh", slope=0.1, dropout=0.0):
    """
    Create a trainable MLP model.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    hidden_dims : [int]
        Dimensions of the hidden layers.
    output_dim : int
        Dimension of the output data.
    n_layers : int
        Number of hidden layers.

    slope: : float
        Slope for leaky relu or xtanh activation functions.

    dropout: float
        Dropout rate between layers.
    Returns
    -------
    model: nn.Sequential
        The MLP model.
    """
    layers = []

    assert len(hidden_dims) == n_layers \
        , "hidden_dims must have length equal to n_layers"

    match activation:
        case "tanh":
            activation_fn = nn.Tanh()
        case "leaky_relu":
            activation_fn = nn.LeakyReLU(slope)
        case "xtanh":
            activation_fn = components.xTanh(slope)
        case "relu":
            activation_fn = nn.ReLU()
        case _:
            raise ValueError(f"Unknown activation function: {activation}")

    for i in range(n_layers):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
        else:
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

        layers.append(activation_fn)

    layers.append(nn.Linear(hidden_dims[-1], output_dim))

    model = nn.Sequential(*layers)
    
    return model

def logreg_mlp(input_dim, hidden_dim, n_layers):
    """
    Create a logistic regression MLP model.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    hidden_dim : int
        Dimension of the hidden layers.
    n_layers : int
        Number of hidden layers.

    Returns
    -------
    model: nn.Sequential
        The MLP model.
    """
    layers = []
    
    for i in range(n_layers):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dim))
        else:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        layers.append(torch.nn.ReLU())
    
    layers.append(nn.Linear(hidden_dim, 2))
    model = nn.Sequential(*layers)
    
    return model

def get_r2_score(z_pred, z_true):
    with torch.no_grad():
        z_pred_aug = torch.cat([z_pred, torch.ones_like(z_pred[:, :1])], dim=1)
        W_aug = torch.linalg.pinv(z_pred_aug) @ z_true
        W = W_aug[:-1]
        alpha = W_aug[-1]
        z_reg = z_pred @ W + alpha

    return r2_score(z_true, z_reg).item()


def infoNCEloss(x, y, temperature):
    cos_sim = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)
    exp_sim = torch.exp(cos_sim / temperature)
    pos = exp_sim.diag()
    total = exp_sim.sum(dim=1) - pos
    loss = -torch.log(pos / total)
    return loss.mean()

def _check_inputs(size, mu, v):
    """helper function to ensure inputs are compatible"""
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
        if v is None:
            v = torch.Tensor([1])
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))


def log_normal(x, mu=None, v=None, broadcast_size=False):
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v)
    else:
        mu, v = _check_inputs(x.size(), mu, v)
    assert mu.shape == v.shape
    return -0.5 * (torch.log(2 * torch.Tensor([torch.pi])).to(x.device) + v.log() + (x - mu).pow(2).div(v))


class GaussianKernel:

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, A):
        x_norm = (A ** 2).sum(dim=1).view(-1, 1)  # shape: (n, 1)
        dist_sq = x_norm + x_norm.T - 2.0 * A @ A.T  # shape: (n, n)

        # Apply Gaussian kernel
        return torch.exp(-dist_sq / (2 * self.sigma ** 2))



class MMRLoss(nn.Module):
    def __init__(self, kernel, lambda_reg: float = 1.0, eps: float = 1e-6, shuffled: bool = True):
        """
        Args:
            kernel: an object implementing either kernel(A, A2) or kernel(A)
                        (e.g. kernel.GaussianKernel()).
            lambda_reg: multiplier for the returned penalty.
            eps: ridge added to A^T A for stability.
            shuffled: whether to use the shuffled independent-copy estimator.
        """
        super().__init__()
        self.kernel = kernel 
        self.eps = eps
        self.shuffled = shuffled

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, d_z]
            a: [B, d_a]
        Returns:
            scalar tensor: lambda_reg * MMR penalty
        """
        B = z.shape[0]
        device = z.device
        dtype = z.dtype
        if B == 0:
            return torch.tensor(0., device=device, dtype=dtype)

        
        K = self.kernel(a)   # [B,B]

        # Augment A with intercept column
        ones = torch.ones(B, 1, device=device, dtype=dtype)
        a_aug = torch.cat([a, ones], dim=1)  # [B, d_a+1]

        # Compute empirical least squares W_aug = (A^T A + eps I)^{-1} A^T Z (no grad)
        with torch.no_grad():
            AtA = a_aug.T @ a_aug                      # [d_a+1, d_a+1]
            ridge = self.eps * torch.eye(AtA.size(0), device=device, dtype=dtype)
            inv = torch.linalg.inv(AtA + ridge)
            W_aug = inv @ (a_aug.T @ z)                # [d_a+1, d_z]

        # split W and intercept
        W = W_aug[:-1, :]    # [d_a, d_z]
        alpha = W_aug[-1:, :]  # [1, d_z]

        # predicted z and residuals psi
        z_pred = a @ W + alpha   # [B, d_z]
        psi = z - z_pred         # [B, d_z]

        loss = psi.T @ K @ psi 
        return loss[0, 0]
