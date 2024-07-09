import torch
import torch.nn as nn

from dl_project.utils.custom_typing import EncoderOutput


class BaseEncoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_filters: int,
        kernel_size: int,
        repr_dim: int,
    ):
        """Encoder to extract the representations

        Args:
            img_size (int): [Image size (must be squared size)]
            in_channels (int): Number of input channels
            num_filters (int): Intermediate number of filters
            kernel_size (int): Convolution kernel size
            repr_dim (int): Dimension of the desired representation
        """
        super().__init__()

        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters * 2 ** 0,
            kernel_size=kernel_size,
            stride=1,device='cuda'
        )
        self.conv1 = nn.Conv2d(
            in_channels=num_filters * 2 ** 0,
            out_channels=num_filters * 2 ** 1,
            kernel_size=kernel_size,
            stride=2,device='cuda',bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=num_filters * 2 ** 1)
        self.conv2 = nn.Conv2d(
            in_channels=num_filters * 2 ** 1,
            out_channels=num_filters * 2 ** 2,
            kernel_size=kernel_size,
            stride=2,device='cuda',bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=num_filters * 2 ** 2)
        self.conv3 = nn.Conv2d(
            in_channels=num_filters * 2 ** 2,
            out_channels=num_filters * 2 ** 3,
            kernel_size=kernel_size,
            stride=2, device='cuda',bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=num_filters * 2 ** 3,device='cuda')

        self.leaky_relu = nn.LeakyReLU()

        self.flatten = nn.Flatten()

        # Get the img_size after the convolutions
        for _ in range(3):
            img_size = (img_size - kernel_size + 1) // 2

        self.dense = nn.Linear(
            in_features=(img_size ** 2) * (num_filters * 2 ** 3),
            out_features=repr_dim,device='cuda'
        )

    def forward(self, Input0: torch.Tensor) -> EncoderOutput:
        """Forward encoder

        Args:
            Input0 (torch.Tensor): Image from a given domain

        Returns:
            EncoderOutput: Representation and feature map
        """
        Conv0 = self.conv0(Input0)
        Conv0 = self.leaky_relu(Conv0)
        Conv1 = self.conv1(Conv0)
        Conv1 = self.leaky_relu(Conv1)
        Conv1 = self.bn1(Conv1)
        Conv2 = self.conv2(Conv1)
        Conv2 = self.leaky_relu(Conv2)
        Conv2 = self.bn2(Conv2)
        Conv3 = self.conv3(Conv2)
        Conv3 = self.leaky_relu(Conv3)
        Conv3 = self.bn3(Conv3)
        ShGInput0 = Conv3
        Flat0 = self.flatten(Conv3)
        ShGInput1 = self.dense(Flat0)

        return EncoderOutput(representation=ShGInput1, feature=ShGInput0)
