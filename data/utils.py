import matplotlib.pyplot as plt
import numpy as np 

import torch 
from torch import nn
import torch.nn.functional as F


class Sin(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return torch.sin(x)
  

class Abs(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return torch.abs(x)


def weights_init_uniform(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(-1.0, 1.0)
            m.bias.data.fill_(0)


def generate_full_rank_matrix(rows, cols, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    assert rows > 0 and cols > 0, "Matrix dimensions must be positive"
    min_dim = min(rows, cols)

    # in 1D case we just return a random nonzero matrix
    if min_dim == 1:
        M = torch.randn(rows, cols)
        # try to avoid almost all 0 case
        while torch.allclose(M, torch.zeros_like(M), atol=1e-2):
            M = torch.randn(rows, cols)
        return M

    A = torch.randn(rows, min_dim)
    B = torch.randn(cols, min_dim)

    # get Q element of QR decomposition
    Q1, _ = torch.linalg.qr(A)
    Q2, _ = torch.linalg.qr(B)

    full_rank_matrix = Q1 @ Q2.T

    return full_rank_matrix


def construct_invertible_mixing(input_dim, hidden_dim, output_dim, n_layers):
    """
    Create an (approximately) invertible mixing network based on an MLP.

    Parameters
    ----------
    input_dim : int
        Dimension of the input data.
    hidden_dim : int
        Dimension of the hidden layers.
    output_dim : int
        Dimension of the output data.
    n_layers : int
        Number of hidden layers.

    Returns
    -------
    mixing_net: nn.Sequential
        The mixing network.

    """
    layers = []
    
    for i in range(n_layers):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dim))
        else:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        layers.append(torch.nn.LeakyReLU(negative_slope=0.2))
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    for layer in layers:
        if isinstance(layer, nn.Linear):
            layer.weight.data = generate_full_rank_matrix(layer.in_features, layer.out_features).T
            layer.bias.data.fill_(0)
    
    mixing_net = nn.Sequential(*layers)
    
    # fix parameters
    for p in mixing_net.parameters():
        p.requires_grad = False
    
    return mixing_net


def generate_derangement(n):
    """
    Generate a derangement of index up to n with no fixed points.
    """
    assert n > 1, "n must be > 1"
    while True:
        perm = np.random.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm


def plot_images(images, title=None):
    n = images.shape[0]
    ncols = int(n ** 0.5)
    nrows = int((n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if title:
        fig.suptitle(title)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i].cpu().numpy() if isinstance(images[i], torch.Tensor) else images[i]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.show()

def plot_density(images, title=None):
    if isinstance(images, torch.Tensor):
        arr = images.cpu().numpy()
    else:
        arr = images
    mean_img = arr.mean(axis=0)
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    if title:
        plt.title(title)
    plt.show()

def sample_covariance_matrix(dim, device='cpu'):
    A = torch.rand(dim, dim, device=device)
    cov = A @ A.T
    return cov + torch.eye(dim, device=device) 


# Utility to generate causal effect and confounding functions
def generate_causal_effect(dimZ, dimY, causal_effect):

    if causal_effect == "linear":
        l = nn.Linear(dimZ, dimY)
        weights = generate_full_rank_matrix(dimY, dimZ)
        l.weight = nn.Parameter(weights)

        h = nn.Linear(dimZ, dimY)
        h_weights = generate_full_rank_matrix(dimY, dimZ)
        h.weight = nn.Parameter(h_weights)
    
    elif causal_effect == "nonlinear":
        l = nn.Sequential(
            nn.Linear(dimZ, 16),
            nn.Tanh(),
            nn.Linear(16, dimY),
        )
        # l.apply(weights_init_uniform)

        h = nn.Sequential(
            nn.Linear(dimZ, 16),
            nn.Tanh(),
            nn.Linear(16, dimY),
        )
        # h.apply(weights_init_uniform)
    elif causal_effect == "sin":
        l = nn.Sequential(
            nn.Linear(dimZ, 16),
            Sin(),
            nn.Linear(16, dimY),
        )
        l.apply(weights_init_uniform)

        h = nn.Sequential(
            nn.Linear(dimZ, 16),
            nn.Tanh(),
            nn.Linear(16, dimY),
        )
        h.apply(weights_init_uniform)
    elif causal_effect == "abs":
        l = nn.Sequential(
            nn.Linear(dimZ, 16),
            Abs(),
            nn.Linear(16, dimY),
        )
        l.apply(weights_init_uniform)

        h = nn.Sequential(
            nn.Linear(dimZ, 16),
            nn.Tanh(),
            nn.Linear(16, dimY),
        )
        h.apply(weights_init_uniform)
    else:
        raise ValueError("wrong causal_effect.")
    return l, h