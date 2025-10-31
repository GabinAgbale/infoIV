import torch
from torch import nn

from model import utils 


class xTanh(nn.Module):
    """Tanh function plus an additional linear term."""
    def __init__(self, alpha: float = 0.1):
        super(xTanh, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.tanh(x) + self.alpha * x

class LogisticRegression(nn.Module):
    """
    Logistic regression model for binary classification.

    Args:
        input_dim (int): Dimension of the input data.
        output_dim (int): Dimension of the output data.
    """
    def __init__(self, 
                 input_dim: int,
                 num_layers: int = 3,
                 hidden_dim: int = 32):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = utils.logreg_mlp(input_dim, hidden_dim, num_layers)

    def forward(self, x):
        return self.model(x)
    


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, num_layers=3, activation='leaky_relu', slope=0.1, dropout_rate=0.0):
        super(Encoder, self).__init__()        
        layers = []

        assert len(hidden_dims) == num_layers, \
            "hidden_dims must be a list with length equal to num_layers"

        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        match activation:
            case 'relu':
                layers.append(nn.ReLU())
            case 'leaky_relu':
                layers.append(nn.LeakyReLU(slope))
            case 'tanh':
                layers.append(nn.Tanh())
            case 'xtanh':
                layers.append(xTanh(slope))
            case _:
                raise ValueError(f"Unsupported activation: {activation}")
        layers.append(nn.Dropout(dropout_rate))

        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            match activation:
                case 'relu':
                    layers.append(nn.ReLU())
                case 'leaky_relu':
                    layers.append(nn.LeakyReLU(slope))
                case 'tanh':
                    layers.append(nn.Tanh())
                case 'xtanh':
                    layers.append(xTanh(slope))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, num_layers=3, activation='leaky_relu', slope=0.1, dropout_rate=0.0):
        super(Decoder, self).__init__()
        layers = []

        assert len(hidden_dims) == num_layers, \
            "hidden_dims must be a list with length equal to num_layers"

        layers.append(nn.Linear(latent_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        match activation:
            case 'relu':
                layers.append(nn.ReLU())
            case 'leaky_relu':
                layers.append(nn.LeakyReLU(slope))
            case 'tanh':
                layers.append(nn.Tanh())
            case 'xtanh':
                layers.append(xTanh(slope))
            case _:
                raise ValueError(f"Unsupported activation: {activation}")

        layers.append(nn.Dropout(dropout_rate))
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            match activation:
                case 'relu':
                    layers.append(nn.ReLU())
                case 'leaky_relu':
                    layers.append(nn.LeakyReLU(slope))
                case 'tanh':
                    layers.append(nn.Tanh())
                case 'xtanh':
                    layers.append(xTanh(slope))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class ConvBlockDown(nn.Module):
    """Convolutional block with downsampling for encoder."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 activation: str = 'leaky_relu',
                 dropout_rate: float = 0.1,
                 slope: float = 0.2):
        super(ConvBlockDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation

        match activation:
            case 'relu':
                self.activation = nn.ReLU()
            case 'leaky_relu':
                self.activation = nn.LeakyReLU(slope)
            case 'elu':
                self.activation = nn.ELU()
            case 'tanh':
                self.activation = nn.Tanh()
            case _:
                raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout2d(dropout_rate)
    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x


class ConvBlockUp(nn.Module):
    """Convolutional block with upsampling for decoder."""
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 activation: str = 'leaky_relu',
                 dropout_rate: float = 0.1,
                 slope: float = 0.2):
        super(ConvBlockUp, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation
        match activation:
            case 'relu':
                self.activation = nn.ReLU()
            case 'leaky_relu':
                self.activation = nn.LeakyReLU(slope)
            case 'elu':
                self.activation = nn.ELU()
            case 'tanh':
                self.activation = nn.Tanh()
            case _:
                raise ValueError(f"Unsupported activation: {activation}")
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x


class ConvEncoder(nn.Module):
    """Convolutional Encoder."""
    def __init__(self, 
                 in_channels: int = 1, 
                 latent_dim: int = 10, 
                 hidden_dims: list[int] = [32, 64, 128, 256],
                 activation: str = 'leaky_relu',
                 input_size: int = 64,
                 dropout_rate: float = 0.1,
                 slope: float = 0.2,    
                 ):
        super(ConvEncoder, self).__init__()
        layers = []
        prev_channels = in_channels
        for h_dim in hidden_dims:
            layers.append(ConvBlockDown(prev_channels, h_dim, activation=activation, dropout_rate=dropout_rate, slope=slope))
            prev_channels = h_dim
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(hidden_dims[-1]*4*4, latent_dim)

        # compute output feature shape dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.conv(dummy)
            self.feature_shape = out.shape[1:]        # (C, H, W)
            self.flatten_dim = out.numel()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x) 
        return x


class ConvDecoder(nn.Module):
    """Convolutional Decoder."""
    def __init__(self, 
                 latent_dim: int = 10, 
                 out_channels: int = 1, 
                 hidden_dims: list[int] = [256, 128, 64, 32],
                 activation: str = 'leaky_relu',
                 feature_shape: tuple = (256, 4, 4),
                 dropout_rate: float = 0.1,
                 slope: float = 0.2):
        super(ConvDecoder, self).__init__()
        self.dense = nn.Linear(latent_dim, int(torch.prod(torch.tensor(feature_shape))))
        layers = []
        prev_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(ConvBlockUp(prev_channels, h_dim, activation=activation, dropout_rate=dropout_rate, slope=slope))
            prev_channels = h_dim
        self.conv = nn.Sequential(*layers)
        self.final_conv = nn.ConvTranspose2d(prev_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_activation = nn.Sigmoid()

        self.feature_shape = feature_shape

    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0), *self.feature_shape)  # Reshape to (batch_size, channels, height, width)
        x = self.conv(x)
        x = self.final_conv(x)
        x = self.output_activation(x)
        return x
