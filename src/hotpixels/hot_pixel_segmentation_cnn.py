import torch
import torch.nn as nn

IMG_SIZE = 128  # Sizes in comments based on IMG_SIZE=32


class HotPixelSegmentationCNN(nn.Module):  # U-Net architecture
    """
    U-Net style architecture for pixel-level segmentation.
    Input: 1xIMG_SIZExIMG_SIZE grayscale image
    Output: 1xIMG_SIZExIMG_SIZE probability mask (values 0-1 for each pixel)
    """
    def __init__(self):
        super(HotPixelSegmentationCNN, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(1, IMG_SIZE)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32 -> 16
        
        self.enc2 = self._conv_block(IMG_SIZE, IMG_SIZE*2)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16 -> 8
        
        self.enc3 = self._conv_block(IMG_SIZE*2, IMG_SIZE*4)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8 -> 4
        
        # Bottleneck
        self.bottleneck = self._conv_block(IMG_SIZE*4, IMG_SIZE*8)
        
        # Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose2d(IMG_SIZE*8, IMG_SIZE*4, 2, stride=2)  # 4 -> 8
        self.dec3 = self._conv_block(IMG_SIZE*8, IMG_SIZE*4)  # 256 = 128 (upconv) + 128 (skip connection)

        self.upconv2 = nn.ConvTranspose2d(IMG_SIZE*4, IMG_SIZE*2, 2, stride=2)  # 8 -> 16
        self.dec2 = self._conv_block(IMG_SIZE*4, IMG_SIZE*2)  # 128 = 64 + 64

        self.upconv1 = nn.ConvTranspose2d(IMG_SIZE*2, IMG_SIZE, 2, stride=2)  # 16 -> 32
        self.dec1 = self._conv_block(IMG_SIZE*2, IMG_SIZE)  # 64 = 32 + 32
        
        # Final output layer
        self.out = nn.Conv2d(IMG_SIZE, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _conv_block(self, in_channels, out_channels):
        """Helper function to create a conv block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # 32x32
        enc2 = self.enc2(self.pool1(enc1))  # 16x16
        enc3 = self.enc3(self.pool2(enc2))  # 8x8
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))  # 4x4
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)  # 8x8
        dec3 = torch.cat([dec3, enc3], dim=1)  # Concatenate skip connection
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)  # 16x16
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)  # 32x32
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.sigmoid(self.out(dec1))  # 1x32x32
        
        return out
