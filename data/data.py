from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset

from data import utils

from utils.utils import linear_regression


class SimpleDataset(Dataset):
    def __init__(self, input_key, output_key, input_data, output_data):
        
        assert len(input_data) == len(output_data), "Input and output data must have the same length."

        self.input_key = input_key
        self.output_key = output_key
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return {self.input_key: self.input_data[idx], self.output_key: self.output_data[idx]}


class DoubleDataset(Dataset):
    def __init__(self, input1_key, input2_key, output_key, input1_data, input2_data, output_data):
        
        assert len(input1_data) == len(output_data), "Input1 and output data must have the same length."
        assert len(input2_data) == len(output_data), "Input2 and output data must have the same length."

        self.input1_key = input1_key
        self.input2_key = input2_key
        self.output_key = output_key
        self.input1_data = input1_data
        self.input2_data = input2_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input1_data)

    def __getitem__(self, idx):
        return {self.input1_key: self.input1_data[idx], 
                self.input2_key: self.input2_data[idx], 
                self.output_key: self.output_data[idx]}


class SCMDataset(ABC, Dataset):
    
    @abstractmethod
    def sample(self):
        pass


class RepExSCMDataset(SCMDataset, Dataset):
    
    """
    Sample according to SCM:
        A -> Z -> X
        Z -> Y
        Z <-- C --> Y (unobserved confounder)

    with: 
        A ~ U(-gamma, gamma) or N(0, I)
        Z = M0 A + V,  V ~ N(0, E)
        X = g(Z)
        Y = l(Z) + U, U = h(V) + N(0, I)    

    Args:  
        dimA (int): Dimension of the anchor variable A.
        dimZ (int): Dimension of the latent variable Z.
        dimX (int): Dimension of the observed variable X.
        dimY (int): Dimension of the target variable Y.
        cond_thresh_ratio (float): Relative threshold for the inversitibility of the mixing network. (lower -> more invertible)
        n (int, optional): Number of samples to generate. Default is 1000.  
        hidden_dim (int, optional): Dimension of the hidden layers in the mixing network. Default is 16.
        n_layers (int, optional): Number of layers in the mixing network. Default is 2.
        device (str, optional): Device to use for the tensors. Default is 'cpu'.
        alpha (float, optional): Linear dependance parameter of Z on A. Default is 1.
        causal_effect (str, optional): "linear" or "nonlinear". Type of causal effect to model. Default is "linear". 
        noise_distribution (str, optional): Distribution of the noise term in the SCM. Default is "uniform".
        sigma_noise (float, optional): Standard deviation of the noise term in the SCM. In gaussian case only. Default is 1.
        confounding_strength (float, optional): Scaling factor for the confounding effect. Default is 1.0.
        noise_indep (bool, optional): If True, the components of V are mutually independent. Default is True.
    """

    def __init__(self, 
                 dimA, dimZ, dimX, dimY, 
                 hidden_dim=16, n_layers=2, alpha=1., 
                 cond_thresh_ratio=0.25,device='cpu', 
                 n=1000, causal_effect="linear", 
                 noise_distribution="uniform",
                 sigma_noise=.1,
                 confounding_strength=1.0,
                 noise_indep=True,
                 ):
        
        super().__init__()

        self.n = n
        self.dimA = dimA
        self.dimZ = dimZ
        self.dimX = dimX
        self.dimY = dimY
        self.cond_thresh_ratio = cond_thresh_ratio
        self.device = device
        self.alpha = alpha
        self.sigma_noise = sigma_noise
        self.confounding_strength = confounding_strength

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers    
        self.h = None
        self.l = None 

        self.noise_indep = noise_indep

        self.noise_distribution = noise_distribution
        if self.noise_distribution not in ["uniform", "gaussian"]:
            raise ValueError("noise_distribution must be either 'uniform' or 'gaussian'.")

        self.causal_effect = causal_effect
        
        a,e = torch.rand(self.dimZ).view(-1, 1), torch.rand(self.dimZ)
        self._E = a @ a.T + torch.diag(e)


    def generate_mixing_funcs(self) -> None:
        self._M0 = utils.generate_full_rank_matrix(self.dimZ, self.dimA)  # Ensure full rank
        
        self.l, self.h = utils.generate_causal_effect(self.dimZ, self.dimY, self.causal_effect)
        
        self.g0 = utils.construct_invertible_mixing(input_dim=self.dimZ,
                                                    hidden_dim=self.hidden_dim,
                                                    output_dim=self.dimX,
                                                    n_layers=self.n_layers)


    def sample(self, gamma) -> None:
        """
        Returns:
            A: (n, dimA) torch.Tensor
            Z: (n, dimZ) torch.Tensor
            X: (n, dimZ) torch.Tensor
            Y: (n, dimY) torch.Tensor
        """
        assert self._M0 is not None, "Mixing matrix M0 not generated. Call generate_mixing_funcs() first."

        match self.noise_distribution:
            case "uniform":
                A = 2 * torch.rand(self.n, self.dimA) * gamma - gamma
                
            case "gaussian":
                A = MultivariateNormal(torch.zeros(self.dimA), torch.eye(self.dimA)).sample([self.n]) + gamma

            case _:
                raise ValueError("noise_distribution must be either 'uniform' or 'gaussian'.")
        
        match self.noise_indep:
            case True:
                V = MultivariateNormal(torch.zeros(self.dimZ), self.sigma_noise * torch.eye(self.dimZ)).sample([self.n])
            case False:
                V = MultivariateNormal(torch.zeros(self.dimZ), self.sigma_noise * self._E).sample([self.n])

        U = self.confounding_strength * self.h(V) + MultivariateNormal(torch.zeros(self.dimY), 0.01 * torch.eye(self.dimY)).sample([self.n])
        
        Z =  A @ self._M0.T + self.alpha * V
        X = self.g0(Z)
        Y = self.l(Z) + U

        self.V = V.to(self.device)
        self.A = A.to(self.device)
        self.Z = Z.to(self.device).detach()
        self.X = X.to(self.device).detach()
        self.Y = Y.to(self.device).detach()
    
    
    def compute_intervention(self, do_A):
        V = self.V.clone() 
        Z = do_A @ self._M0.T.to(self.device) + self.alpha * V

        Y = self.l(Z.cpu()) + self.confounding_strength * self.h(V.cpu())

        return Y.to(self.device).detach()


    def __len__(self):
        if self.A is None:
            return 0
        return self.n
    
    
    def __getitem__(self, idx):
        if self.A is None:
            raise ValueError("No data generated yet. Call `.sample(gamma)` first.")
        return {
            'A': self.A[idx],
            'Z': self.Z[idx],
            'X': self.X[idx],
            'Y': self.Y[idx]
        }
    
    def return_clone(self):
        """Returns a clone of the dataset with same mixing functions."""

        clone = RepExSCMDataset(dimA=self.dimA, 
                                dimZ=self.dimZ, 
                                dimX=self.dimX, 
                                dimY=self.dimY,
                                cond_thresh_ratio=self.cond_thresh_ratio,
                                device=self.device,
                                noise_distribution=self.noise_distribution,)
        clone._M0 = self._M0.clone()
        clone.h = self.h
        clone.l = self.l
        clone.g0 = self.g0
        return clone
    
    def save_mixing_funcs(self, path):
        """Saves the mixing functions to a file."""
        torch.save({
            'M0': self._M0,
            'h': self.h.state_dict(),
            'l': self.l.state_dict(),
            'g0': self.g0.state_dict()
        }, path)


    def load_mixing_funcs(self, path):
        """Loads the mixing functions from a file."""
        checkpoint = torch.load(path)
        self._M0 = checkpoint['M0']
        self.h.load_state_dict(checkpoint['h'])
        self.l.load_state_dict(checkpoint['l'])
        self.g0.load_state_dict(checkpoint['g0'])
    
    def set_M0(self, M0):
        """
        Sets the mixing matrix M0 to a provided matrix.
        
        Args:
            M0 (torch.Tensor): The mixing matrix to set.
        """
        tensor_M0 = torch.tensor(M0, dtype=torch.float32)
        if tensor_M0.shape != (self.dimZ, self.dimA):
            raise ValueError(f"M0 must have shape ({self.dimZ}, {self.dimA}), got {tensor_M0.shape}.")
        
        self._M0 = tensor_M0
    

class ControlVariableDataset(Dataset):
    """
    Dataset containing target variable Y, estimated latent variable Z 
    and estimated Control Variable V.
    
    """
    def __init__(self, Z, Y, A, X, device):

        self.device = device
        self.Z = Z.to(self.device)
        self.Y = Y.to(self.device)
        self.V = None # Control variable V

        self.A = A.to(self.device)
        self.X = X.to(self.device)

        self.W = None  # Linear regression weights
        self.b = None  # Linear regression bias

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, idx):
        if self.V is None:
            raise ValueError("Control variable V not computed. Call `.compute_control_variable()` first.")

        return {
            'Z': self.Z[idx],
            'Y': self.Y[idx],
            'V': self.V[idx],
            'X': self.X[idx],
        }
    
    def solve_linear(self):
        with torch.no_grad():
            self.W, self.b = linear_regression(self.A, self.Z)

        
    def assign_linear_weights(self, W, b):
        """
        Assigns the linear regression weights W and bias b.
        """
        if W.shape[0] != self.A.shape[1] or b.shape[0] != self.Z.shape[1]:
            raise ValueError("Weights W and bias b must match the dimensions of A and Z respectively.")
        
        self.W = W.to(self.device)
        self.b = b.to(self.device)

    def compute_control_variable(self):
        """
        Computes the control variable V = Z - (A @ W + b)
        where W and b are obtained from linear regression of A on Z.
        """
        if self.W is None or self.b is None:
            raise ValueError("Weights W and bias b not computed. Call `.solve_linear(A)` first.")

        with torch.no_grad():
            self.V = self.Z - (self.A @ self.W + self.b)
            self.V = self.V.to(self.device)

    def save(self, path):
        torch.save({
            'Z': self.Z,
            'Y': self.Y,
            'V': self.V
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Z = checkpoint['Z']
        self.Y = checkpoint['Y']
        self.V = checkpoint['V']
    
    def to(self, device):
        self.Z = self.Z.to(device)
        self.Y = self.Y.to(device)
        self.V = self.V.to(device)
        return self


class ContrastiveDataset(RepExSCMDataset):
    """
    Extends RepExSCMDataset to include contrastive samples by swapping A.
    C = 1 for original samples, 0 for swapped contrastive samples.
    """
    def __init__(self, dimA, dimZ, dimX, dimY, n_contrastive_pairs=10, 
                 hidden_dim=16, n_layers=2, alpha=1., cond_thresh_ratio=0.25, 
                 device='cpu', n=1000, causal_effect="linear", noise_distribution="uniform"):

        super().__init__(dimA=dimA,
                         dimZ=dimZ,
                         dimX=dimX,
                         dimY=dimY,
                         hidden_dim=hidden_dim,
                         n_layers=n_layers,
                         alpha=alpha,
                         cond_thresh_ratio=cond_thresh_ratio,
                         device=device,
                         n=n,
                         causal_effect=causal_effect,
                         noise_distribution=noise_distribution)

        self.n_contrastive_pairs = n_contrastive_pairs

    def _generate_contrastive_data(self):
        """Call after sampling to create contrastive views of A."""
        self.Z = torch.cat([self.Z] * (self.n_contrastive_pairs + 1), dim=0)
        self.X = torch.cat([self.X] * (self.n_contrastive_pairs + 1), dim=0)
        self.Y = torch.cat([self.Y] * (self.n_contrastive_pairs + 1), dim=0)

        perms = [utils.generate_derangement(self.n) for _ in range(self.n_contrastive_pairs)]
        swapped_As = [self.A[perm] for perm in perms]

        self.A = torch.cat([self.A] + swapped_As, dim=0)
        self.C = torch.cat([torch.ones(self.n)] + [torch.zeros(self.n)] * self.n_contrastive_pairs, dim=0)

    def sample(self, gamma=1.0) -> None:
        """Override to also generate contrastive samples after sampling."""
        assert self._M0 is not None, "Mixing matrix M0 not generated. Call generate_mixing_funcs() first."
        super().sample(gamma)
        self._generate_contrastive_data()

    def __getitem__(self, idx):
        return {
            'A': self.A[idx],
            'Z': self.Z[idx],
            'X': self.X[idx],
            'Y': self.Y[idx],
            'C': self.C[idx],
        }
    
    def __len__(self):
        return self.n * (self.n_contrastive_pairs + 1)

    def return_clone(self):
        """Returns a full clone of the contrastive dataset, preserving the contrastive structure."""
        clone = ContrastiveDataset(dimA=self.dimA,
                                   dimZ=self.dimZ,
                                   dimX=self.dimX,
                                   dimY=self.dimY,
                                   n_contrastive_pairs=self.n_contrastive_pairs,
                                   hidden_dim=self.hidden_dim,
                                   n_layers=self.n_layers,
                                   alpha=self.alpha,
                                   cond_thresh_ratio=self.cond_thresh_ratio,
                                   device=self.device,
                                   n=self.n)
        clone._M0 = self._M0.clone()
        clone.h = self.h
        clone.l = self.l
        clone.g0 = self.g0
        clone.sample(gamma=1.0)  # Re-sample and regenerate contrastive data
        return clone

    def load_mixing_funcs(self, path):
        """Loads the mixing functions from a file."""
        checkpoint = torch.load(path)
        self._M0 = checkpoint['M0']
        self.h.load_state_dict(checkpoint['h'])
        self.l.load_state_dict(checkpoint['l'])
        self.g0.load_state_dict(checkpoint['g0'])


class ZYDataset(Dataset):
    """
    Dataset for 2nd step in IV method, to fit causal effect model on estimated latent
    """
    def __init__(self, Z, Y):
        self.Z = Z
        self.Y = Y

    def __getitem__(self, idx):
        return {
            'Z': self.Z[idx],
            'Y': self.Y[idx],
        }
    
    def __len__(self):
        return len(self.Z)
    

class IVDataset(Dataset):
    """
    Dataset containing target variable Y, estimated latent variable Z,
    estimated auxiliary latent prediction Z_aux, and estimated Control Variable V.

    Args:
        Z (torch.Tensor): Estimated latent variable Z.
        Z_aux (torch.Tensor): Estimated auxiliary latent prediction Z_aux.
        Y (torch.Tensor): Target variable Y.
        device (str): Device to use for the tensors.
    
    """
    def __init__(self, Z, Z_aux, Y, device):
        self.device = device

        self.Z_aux = Z_aux.to(self.device)
        self.Z = Z.to(self.device)
        self.Y = Y.to(self.device)
        self.Z_aux = self.Z_aux.to(self.device)
        self.V = None

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, idx):
        if self.V is None:
            raise ValueError("Control variable V not computed. Call `.compute_control_variable()` first.")

        return {
            'Z': self.Z[idx],
            'Z_aux': self.Z_aux[idx],
            'Y': self.Y[idx],
            'V': self.V[idx]
        }

    def compute_control_variable(self):
        """
        Computes the control variable V = Z - Z_aux
        """
        with torch.no_grad():
            self.V = self.Z - self.Z_aux
            self.V = self.V.to(self.device)
        
    def save(self, path):
        torch.save({
            'Z': self.Z,
            'Y': self.Y,
            'V': self.V,
            'Z_aux': self.Z_aux
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.Z_aux = checkpoint['Z_aux']
        self.Z = checkpoint['Z']
        self.Y = checkpoint['Y']
        self.V = checkpoint['V']
    
    def to(self, device):
        self.Z_aux = self.Z_aux.to(device)
        self.Z = self.Z.to(device)
        self.Y = self.Y.to(device)
        self.V = self.V.to(device)
        return self
    
