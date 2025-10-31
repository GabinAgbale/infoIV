import numpy as np
import torch
from torch.utils.data import Dataset

import time

# Global dictionary mapping latent variable names to their indices
LATENT_IDX = {
    'color': 0,
    'shape': 1,
    'scale': 2,
    'orientation': 3,
    'posX': 4,
    'posY': 5
}


class dSpritesLoader:
    def __init__(self, path, device="cpu"):
        data = np.load(path, allow_pickle=True, encoding='latin1')
        imgs = data['imgs']
        latents_values = data['latents_values']
        latents_classes = data['latents_classes']
        self.metadata = data['metadata'][()]
        self.latent_names = self.metadata['latents_names']

        self.imgs = torch.from_numpy(imgs).float()
        self.latents_values = torch.from_numpy(latents_values).float()
        self.latents_classes = torch.from_numpy(latents_classes)
        
        self.device = device
        self.imgs = self.imgs.to(device)
        self.latents_values = self.latents_values.to(device)
        self.latents_classes = self.latents_classes.to(device)
        self.latent_names = self.metadata['latents_names']

class dSpritesConditionalLoader:
    def __init__(self, loader: dSpritesLoader,  
                 conditions: dict = None):
        
        self.loader = loader
        
        if conditions is not None:
            mask = self._select_latents_idx(conditions)
            self.imgs = loader.imgs[mask]
            self.latents_values = loader.latents_values[mask]
        else:
            self.imgs = loader.imgs
            self.latents_values = loader.latents_values
        
        self.latents_names = loader.latent_names


    def _select_latents_idx(self, conditions):
        full_latents = self.loader.latents_values

        mask = torch.ones(len(full_latents), dtype=torch.bool, device=self.loader.device)
        for name, cond in conditions.items():
            i = self.loader.latent_names.index(name)
            values = full_latents[:, i]
            if isinstance(cond, tuple) and len(cond) == 2:
                mask &= (values >= cond[0]) & (values <= cond[1])
            else:
                mask &= (values == cond)
        return mask


class dSpritesFullDataset(Dataset):
    
    """

    1. choose a confounding variable U (e.g. "shape")          
    2. sample Q 
    3. compute I as a deterministic (invertible) function of Q independent of U 
    4. compute Z = Q + noise
    5. extract X from Z (dSprites image)
    6. compute Y = f(Z_U) + l(U) + noise (target function)

    Args:
    loader: dSpritesLoader
    confounder_name: name of the confounding variable (e.g. "shape")
    target_fn: function to compute target Y from latents Z
    instrument_mixing: function to compute instrument I from Q
    confounding_fn: function to compute confounding effect from U
    device: device to load data on

    Outputs:
    X: images,
    A: instrument
    Z: ground-truth latents
    Y: target 
    """
    

    def __init__(self, 
                 loader: 'dSpritesLoader | dSpritesConditionalLoader',
                 confounder_name: str,
                 instrument_mixing: callable,
                 target_fn: callable,
                 confounding_fn: None = None,
                 device="cpu",
                 seed: int = 0,
                 test: bool = False,
                 confounding_strength: float = 1.0):
        super().__init__()

        torch.manual_seed(seed)

        self.test = test
        self.loader = loader

        assert confounder_name in self.loader.latents_names, \
            ValueError(f"Confounder {confounder_name} not in latent names.")
        self.confounder_name = confounder_name
    
        self.target_fn = target_fn
        self.instrument_mixing = instrument_mixing
        self.confounding_fn = confounding_fn

        self.X = None
        self.Z = None
        self.A = None
        self.Y = None

        self.device = device

        self.confounding_strength = confounding_strength

        # Compute possible values for each latent
        self.uniques = [torch.unique(self.loader.latents_values[:, i]).to(self.device) for i in range(self.loader.latents_values.shape[1])]
        steps = [(u[1] - u[0])/2 if len(u) > 1 else u[0]/2 for u in self.uniques]
        self.ranges = [(u[0]-s, u[-1]+s) for u, s in zip(self.uniques, steps)]

    def sample(self, n):
        t1 = time.time()
        # Sample intermediate Q uniformly in ranges
        Q = torch.empty((n, self.loader.latents_values.shape[1]), device=self.device)
        for i, (low, high) in enumerate(self.ranges):
            Q[:, i] = torch.empty(n, device=self.device).uniform_(low, high)

        # compute Z from Q as the closest latent value for each component
        Z = torch.empty_like(Q,device=self.device)
        for i in range(Q.shape[1]):
            vals = self.uniques[i]
            # For each sample, find the closest value in vals
            idx = torch.argmin(torch.abs(vals.unsqueeze(0) - Q[:, i].unsqueeze(1)), dim=1)
            Z[:, i] = vals[idx]

        self.Z = Z 
        self.X = self._get_images_from_latents(self.Z)

        # exclude confounder from instrument
        mask = [i for i in range(Q.shape[1]) if i != LATENT_IDX[self.confounder_name]]
        self.A = self.instrument_mixing(Q[:, mask])

        if not self.test:
            self.Y = self.target_fn(Z).unsqueeze(1) + self.confounding_strength * self.confounding_fn(Z[:, LATENT_IDX[self.confounder_name]].unsqueeze(1)) + 0.01 * torch.randn(n, 1,device=self.device)
        else:  
            self.Y = self.target_fn(Z)
            
        
        print(f"Sampling {n} points took {time.time() - t1:.2f} seconds.")
        return None


    def _get_images_from_latents(self, Z):
        """
        Given a batch of latent vectors Z, return the corresponding images from the loader
        using exact matching in the filtered latents.
        """
        latents_values = self.loader.latents_values
        imgs = self.loader.imgs
        X_list = []
        for z in Z:
            matches = (latents_values == z).all(dim=1)
            idx = torch.nonzero(matches, as_tuple=False)
            if idx.numel() == 0:
                raise ValueError(f"No matching latent found for {z}")
            X_list.append(imgs[idx[0,0]])
        return torch.stack(X_list)
    
    def __len__(self):
        if self.X is None:
            raise ValueError("Dataset not sampled yet. Call sample(n) first.")
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.X is None or self.Z is None or self.A is None or self.Y is None:
            raise ValueError("Dataset not sampled yet. Call sample(n) first.")
        return {
            'X': self.X[idx].unsqueeze(0),  # add channel dimension
            'Z': self.Z[idx],
            'A': self.A[idx],
            'Y': self.Y[idx]
        }
    
    def save_mixing_funcs(self, path):
        torch.save({
            'instrument_mixing': self.instrument_mixing,
            'target_fn': self.target_fn,
            'confounding_fn': self.confounding_fn
        }, path)
        print(f"Saved mixing functions to {path}")
        return None
