import torch

# Global dictionary mapping latent variable names to their indices
LATENT_IDX = {
    'color': 0,
    'shape': 1,
    'scale': 2,
    'orientation': 3,
    'posX': 4,
    'posY': 5
}

def shape_target(latents):
    """
    Example target function: classify shape (assuming shape is latent index 1)
    Returns 1 if shape==2, else 0.
    """
    idx = LATENT_IDX['shape']
    return (latents[:, idx] == 2).long()

def posX_scale_target(latents):
    """
    Example target function: sum of posX and scale
    """
    idx_posX = LATENT_IDX['posX']
    idx_scale = LATENT_IDX['scale']
    return latents[:, idx_posX] + latents[:, idx_scale]

def posY_fn(latents):
    """
    Example target function: posY value
    """
    idx_posY = LATENT_IDX['posY']
    return latents[:, idx_posY]

def identity(latents):
    """
    Identity function: returns the latents as is
    """
    return latents

class Linear_mixing:
    def __init__(self, weights, device='cpu'):
        self.weights = weights.to(device)

    def __call__(self, latents):
        return torch.matmul(latents, self.weights)

class Centered_identity:
    def __init__(self, intercept, device='cpu'):
        self.device = device
        self.intercept = intercept.to(device)

    def __call__(self, latents):
        return latents - self.intercept