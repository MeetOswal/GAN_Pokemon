import torch
from torch import nn

""" The Generator Block"""
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels = 3, kernel_size = 4, strides = 2, padding = 1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, padding, bias = False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
    
n_G = 64
net_G = nn.Sequential(
    G_block(in_channels = 100, out_channels = n_G * 8,
           strides = 1, padding = 0),                          # Output: (64 * 8, 4, 4)
    G_block(in_channels = n_G * 8, out_channels = n_G * 4),    # Output: (64 * 4, 8, 8)
    G_block(in_channels = n_G * 4, out_channels = n_G * 2),    # Output: (64 * 2, 16, 16)
    G_block(in_channels = n_G * 2, out_channels = n_G),        # Output: (64, 32, 32)
    nn.ConvTranspose2d(in_channels = n_G, out_channels = 3, kernel_size = 4, stride = 2, padding = 1, bias = False),
    nn.Tanh()) # Output : (3,64,64)

""" Discriminator Block"""
class D_block(nn.Module):
    def __init__(self, out_channels, in_channels = 3, kernel_size = 4, strides = 2, 
                 padding = 1, alpha = 0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias = False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace = True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
    
n_D = 64
net_D = nn.Sequential(
    D_block(n_D), # Output : (64, 32, 32)
    D_block(in_channels = n_D, out_channels = n_D * 2), # Output: (64 * 2, 16, 16)
    D_block(in_channels = n_D * 2, out_channels = n_D * 4),  # Output: (64 * 4, 8, 8)
    D_block(in_channels = n_D * 4, out_channels = n_D * 8), # Output: (64 * 8, 4, 4)
    nn.Conv2d(in_channels = n_D * 8, out_channels = 1, kernel_size = 4, bias = False))  # Output: (1, 1, 1)