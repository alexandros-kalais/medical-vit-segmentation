import torch.nn as nn
from monai.networks.nets import UNet
from . import register

@register("unet")
class UNetSmall(nn.Module):
    """
    Minimal MONAI U-Net wrapper.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units: int = 2,
        spatial_dims: int = 2,
    ):
        super().__init__()
        self.net = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

    def forward(self, x):
        return self.net(x)
