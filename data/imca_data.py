import torch 
from torch import nn 
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal

from data import utils 

class IMCADataset(Dataset):

    def __init__(self,
                 dimA, dimZ, dimX, dimY,
                 n=10000,hidden_dim=16, 
                 n_layers=2, device='cpu', 
                 causal_effect="linear",
                 confounding_strength=1.,
                 indep_latents=False,):
        
        super().__init__()
        
        self.n = n
        self.dimA = dimA
        self.dimZ = dimZ
        self.dimX = dimX
        self.dimY = dimY
        self.device = device
        self.causal_effect = causal_effect
        self.confounding_strength = confounding_strength
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.indep_latents = indep_latents

    def generate_mixing_funcs(self) -> None:
        
        # mixing function from Z to X
        self.g0 = utils.construct_invertible_mixing(input_dim=self.dimZ,
                                                    hidden_dim=self.hidden_dim,
                                                    output_dim=self.dimX,
                                                    n_layers=self.n_layers)

        # causal effect from Z to Y
        self.l, _ = utils.generate_causal_effect(self.dimZ, self.dimY, self.causal_effect)
        

        # generate mean function 
        self.mean_func = utils.construct_invertible_mixing(input_dim=self.dimA,
                                                           hidden_dim=self.hidden_dim,
                                                           output_dim=self.dimZ,
                                                           n_layers=1)
        
        self.var_func = utils.construct_invertible_mixing(input_dim=self.dimA,
                                                          hidden_dim=self.hidden_dim,
                                                          output_dim=self.dimZ,
                                                          n_layers=1)
    
        
        # generate base covariance matrix mixing Z components
        match self.indep_latents:
            case True:
                self.base_cov = torch.eye(self.dimZ)
            case False:
                self.base_cov = utils.sample_covariance_matrix(self.dimZ)

        # generate linear mixing of epsilon to obtain residual noise in Y, no intercept to ensure zero-mean
        self.noise_linear_mixing = nn.Linear(self.dimZ, self.dimY, bias=False)



    def sample(self) -> None:

        assert self.g0 is not None and self.l is not None, \
            ValueError("Mixing functions not initialized. Call generate_mixing_funcs() first.")
        
        # sample instrument A
        self.A = MultivariateNormal(torch.zeros(self.dimA), 
                                    torch.eye(self.dimA)).sample((self.n,))
        
        means = self.mean_func(self.A)
        vars = torch.exp(self.var_func(self.A))  # add exp to ensure positivity

        
        # generate epsilon, independent of A, mixing components of Z
        self.epsilon = MultivariateNormal(torch.zeros(self.dimZ), 0.01*self.base_cov).sample((self.n,))

        #generate Z
        self.Z = means + vars * self.epsilon

        # generate X
        self.X = self.g0(self.Z)

        # generate Y
        with torch.no_grad():
            endogen_noise = self.confounding_strength * self.noise_linear_mixing(self.epsilon)
        
            noise = 0.01 * torch.randn(self.n, self.dimY)
            self.Y = self.l(self.Z) + endogen_noise + noise
    

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

        clone = IMCADataset(self.dimA, self.dimZ, self.dimX, self.dimY,
                           n=self.n, hidden_dim=self.hidden_dim,
                           n_layers=self.n_layers, device=self.device,
                           causal_effect=self.causal_effect,
                           confounding_strength=self.confounding_strength,
                           indep_latents=self.indep_latents)
        
        clone.g0 = self.g0
        clone.l = self.l
        clone.mean_func = self.mean_func
        clone.var_func = self.var_func
        clone.base_cov = self.base_cov
        clone.noise_linear_mixing = self.noise_linear_mixing

        return clone
    
    def load_mixing_funcs(self, path):
        """Loads the mixing functions from a file."""
        checkpoint = torch.load(path, weights_only=False)
        self.g0 = checkpoint['g0']
        self.l = checkpoint['l']
        self.mean_func = checkpoint['mean_func']
        self.var_func = checkpoint['var_func']
        self.base_cov = checkpoint['base_cov']
    
    def save_mixing_funcs(self, path):
        torch.save({
            'g0': self.g0,
            'l': self.l,
            'mean_func': self.mean_func,
            'var_func': self.var_func,
            'base_cov': self.base_cov
        }, path)
        print(f"Saved mixing functions to {path}")
        return None

import torch
from torch.utils.data import Dataset

class ContrastiveIMCADataset(Dataset):
    """
    Takes samples from IMCADataset and creates contrastive pairs by swapping A,
    and swaps X, Z, Y accordingly. C = 1 for original samples, 0 for swapped contrastive samples.
    """
    def __init__(self, imca_dataset, seed, n_contrastive_pairs=10):
        self.n = len(imca_dataset.A)
        self.n_contrastive_pairs = n_contrastive_pairs

        # Store original samples
        self.A = imca_dataset.A
        self.Z = imca_dataset.Z
        self.X = imca_dataset.X
        self.Y = imca_dataset.Y

        torch.manual_seed(seed)

        # Generate contrastive samples by swapping A, Z, X, Y
        self.Z = torch.cat([self.Z] * (self.n_contrastive_pairs + 1), dim=0)
        self.X = torch.cat([self.X] * (self.n_contrastive_pairs + 1), dim=0)
        self.Y = torch.cat([self.Y] * (self.n_contrastive_pairs + 1), dim=0)

        perms = [utils.generate_derangement(self.n) for _ in range(self.n_contrastive_pairs)]
        swapped_As = [self.A[perm] for perm in perms]

        self.A = torch.cat([self.A] + swapped_As, dim=0)
        self.C = torch.cat([torch.ones(self.n, dtype=int)] + [torch.zeros(self.n, dtype=int)] * self.n_contrastive_pairs, dim=0)

    def __len__(self):
        return self.n * (self.n_contrastive_pairs + 1)

    def __getitem__(self, idx):
        return {
            'A': self.A[idx],
            'Z': self.Z[idx],
            'X': self.X[idx],
            'Y': self.Y[idx],
            'C': self.C[idx],
        }