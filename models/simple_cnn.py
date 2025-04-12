import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNNEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*2), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels*4), nn.ReLU(inplace=True)
        )

        self.feature_channels = [base_channels, base_channels*2, base_channels*4]

    def forward(self, x, hierarchical=False):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        if hierarchical:
            return [out1, out2, out3]
        return out3

    def get_downsample_ratio(self):
        return 8  # since we applied stride=2 for 3 times

    def get_feature_map_channels(self):
        return self.feature_channels
