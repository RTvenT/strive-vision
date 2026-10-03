from collections.abc import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)
    
    
class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.dconv = ConvBlock(self.in_channels, self.out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self ,x):
        x = self.dconv(x)
        out = self.pool(x)
        
        return out, x
    


class Decoder(nn.Module):
    def __init__(self, in_channels: int,  out_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.skip_channels = out_channels # это допущение работает только при условии, что upsample_channels = in_channels / 2
        self.out_channels = out_channels
        
        self.upsample_channels = int(self.in_channels / 2)
        
    
        self.convT = nn.ConvTranspose2d(self.in_channels, self.upsample_channels, kernel_size=2, stride=2, bias=True)
        self.dconv = ConvBlock(self.upsample_channels + self.skip_channels, self.out_channels)
        
    def forward(self, x, skip):
        x = self.convT(x)
        y = torch.concat([x, skip], dim=1)
        return self.dconv(y)


class Unet(nn.Module):
    def __init__(
        self, 
        input_channels: int = 1, 
        hidden_channels: Sequence[int] = (64, 128, 256, 512), 
        bn_channels: int = 1024,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = tuple(hidden_channels)
        self.bn_channels = bn_channels
        self.num_classes = num_classes
        
        self.config = {
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "bn_channels": self.bn_channels,
            "num_classes": self.num_classes
        }
        
        self.hid_len = len(self.hidden_channels)
    
        # Encoders and Decoders initialization
        self.encoders = nn.ModuleList([Encoder(self.input_channels, self.hidden_channels[0])])
        self.decoders = nn.ModuleList([Decoder(self.bn_channels, self.hidden_channels[-1])])
        
        if self.hid_len >= 2:
            for i in range(self.hid_len - 1):
                self.encoders.append(
                    Encoder(self.hidden_channels[i], self.hidden_channels[i+1])
                )
                
            for i in range(self.hid_len-1, 0, -1):
                self.decoders.append(
                    Decoder(self.hidden_channels[i], self.hidden_channels[i-1])
                )
        
        # Bottle Neck initialization
        self.bn = ConvBlock(self.hidden_channels[-1], self.bn_channels)
     
        # Out Conv initizalization
        self.out_conv = nn.Conv2d(self.hidden_channels[0], self.num_classes, 1, 1)
            
    
    def forward(self, x):        
        self._validation_input(x)
        
        skip_list = []
        
        # Encoding
        encoded = x
        for block in self.encoders:
            encoded, skip = block(encoded)
            skip_list.append(skip)
            
        # Bottle Neck
        bn = self.bn(encoded)
        
        # Decoding
        decoded = bn
        
        for block, skip in zip(self.decoders, reversed(skip_list)):
            decoded = block(decoded, skip)
    
            
        # Out conved
        out = self.out_conv(decoded)
        
        return out
    
    
    def _validation_input(self, x: torch.Tensor): 
        x_shape = x.shape
        
        if len(x_shape) != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input)")
        
        h, w = x_shape[2:]
        divider = 2 ** len(self.hidden_channels)
    
        if (h % divider != 0) or (w % divider != 0): 
            raise ValueError(f"Input tensor's H and W must be divisible by {divider} (got {h, w})")
        