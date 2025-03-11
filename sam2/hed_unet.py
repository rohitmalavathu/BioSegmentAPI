import torch
import torch.nn as nn
import torchvision.models as models

class HED_UNet(nn.Module):
    def __init__(self):
        super(HED_UNet, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features[:23]  # Use first 23 layers for HED

        # Side outputs for deep supervision
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)

        # U-Net Decoder
        self.up4 = self._upsample_block(512, 256)
        self.up3 = self._upsample_block(256, 128)
        self.up2 = self._upsample_block(128, 64)
        self.up1 = self._upsample_block(64, 32)

        # Final edge detection layer
        self.final_edge = nn.Conv2d(32, 1, kernel_size=1)

        # Fusion layer for multi-scale side outputs
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # Encoder (HED feature extraction)
        x1 = self.features[:4](x)   # First conv block
        x2 = self.features[4:9](x1)  # Second conv block
        x3 = self.features[9:16](x2) # Third conv block
        x4 = self.features[16:23](x3) # Fourth conv block

        # Side outputs
        s1 = self.side1(x1)
        s2 = self.side2(x2)
        s3 = self.side3(x3)
        s4 = self.side4(x4)
        s5 = self.side5(x4)

        # Upsample side outputs to match input size
        s1 = nn.functional.interpolate(s1, size=x.shape[2:], mode='bilinear', align_corners=False)
        s2 = nn.functional.interpolate(s2, size=x.shape[2:], mode='bilinear', align_corners=False)
        s3 = nn.functional.interpolate(s3, size=x.shape[2:], mode='bilinear', align_corners=False)
        s4 = nn.functional.interpolate(s4, size=x.shape[2:], mode='bilinear', align_corners=False)
        s5 = nn.functional.interpolate(s5, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Fusion of side outputs
        fused = self.fuse(torch.cat((s1, s2, s3, s4, s5), dim=1))

        # Decoder (U-Net upsampling with skip connections)
        d4 = self.up4(x4) + x3  # Skip connection
        d3 = self.up3(d4) + x2
        d2 = self.up2(d3) + x1
        d1 = self.up1(d2)

        # Final edge map
        edge_map = self.final_edge(d1)

        edge_map = nn.functional.interpolate(edge_map, size=(512, 512), mode='bilinear', align_corners=False)

        return [s1, s2, s3, s4, s5, fused, edge_map]  # Multi-scale + final refined map