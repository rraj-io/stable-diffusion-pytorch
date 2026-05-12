import torch
from torch import nn 
from torch.nn import functional as F 
from decoder import VAE_AttentionBlock, VAE_ResidualBlock 

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Height, Width) -> (Batch_size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
 
            # (Batch_size, 128, Height / 2, Width / 2) -> (Batch_size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256),

            # (Batch_size, 256, Height / 2, Width / 2) -> (Batch_size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, Height / 2, Width / 2) -> (Batch_size, 256, Height / 4, Width / 4)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_size, 256, Height / 4, Width / 4) -> (Batch_size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512),

            # (Batch_size, 512, Height / 4, Width / 4) -> (Batch_size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 4, Width / 4) -> (Batch_size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512),

            # (Batch_size, 512, Height / 8, Width / 8) -> (Batch_size, 512, Height / 8, Width / 8)
            nn.SiLU(),

        )
