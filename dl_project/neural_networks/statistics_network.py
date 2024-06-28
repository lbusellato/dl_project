import torch.nn as nn
import torch
from dl_project.neural_networks.encoder import BaseEncoder


def tile_and_concat(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Merge 1D and 2D tensor (use to aggregate feature maps and representation
    and compute local mutual information estimation)

    Args:
        tensor (torch.Tensor): 2D tensor (feature maps)
        vector (torch.Tensor): 1D tensor representation

    Returns:
        torch.Tensor: Merged tensor (2D)
    """

    B, C, H, W = tensor.size()
    vector = vector.unsqueeze(2).unsqueeze(2)
    expanded_vector = vector.expand((B, vector.size(1), H, W))
    return torch.cat([tensor, expanded_vector], dim=1)


class LocalStatisticsNetwork(nn.Module):
    def __init__(self, img_feature_channels: int):
        """Local statistique nerwork

        Args:
            img_feature_channels (int): [Number of input channels]
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=img_feature_channels, out_channels=512, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1, stride=1
        )
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, concat_feature: torch.Tensor) -> torch.Tensor:
        x = self.conv1(concat_feature)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        local_statistics = self.conv3(x)
        return local_statistics


class GlobalStatisticsNetwork(nn.Module):
    """Global statistics network

    Args:
        feature_map_size (int): Size of input feature maps
        feature_map_channels (int): Number of channels in the input feature maps
        num_filters (int): Intermediate number of filters
        kernel_size (int): Convolution kernel size
        latent_dim (int): Dimension of the representationss
    """

    def __init__(
        self, feature_map_size: int, 
        feature_map_channels: int, 
        num_filters: int,
        kernel_size: int,
        latent_dim: int
    ):

        super().__init__()

        self.ShGConv0 = nn.Conv2d(
            in_channels=feature_map_channels,
            out_channels=num_filters * 2 ** 1,
            kernel_size=kernel_size,
            stride=1,
        )
        self.ShGConv1 = nn.Conv2d(
            in_channels=num_filters * 2 ** 1,
            out_channels=num_filters * 2 ** 0,
            kernel_size=kernel_size,
            stride=1,
        )

        # Compute the size of the input features for the first dense layer
        conv_output_size = feature_map_size - 2 * (kernel_size - 1)
        flattened_size = num_filters * 2 ** 0 * conv_output_size * conv_output_size
        concat_size = flattened_size + latent_dim        
        self.ShGDense0 = nn.Linear(
            in_features=concat_size,
            out_features=feature_map_channels,
        )
        self.ShGDense1 = nn.Linear(in_features=feature_map_channels, out_features=feature_map_channels)
        self.ShGOutput0 = nn.Linear(in_features=feature_map_channels, out_features=1)

        self.flatten = nn.Flatten()

        self.relu = nn.ReLU()

    def forward(
        self, ShGInput0: torch.Tensor, ShGInput1: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the Global Statistics Network

        Parameters
        ----------
        ShGInput0 : torch.Tensor
            The feature map.
        ShGInput1 : torch.Tensor
            The feature representation.

        Returns
        -------
        torch.Tensor
            The global mutual information.
        """
        ShGConv0 = self.ShGConv0(ShGInput0)
        ShGConv0 = self.relu(ShGConv0)
        ShGConv1 = self.ShGConv1(ShGConv0)
        ShGFlat0 = self.flatten(ShGConv1)
        ShGConcat0 = torch.cat([ShGFlat0, ShGInput1], dim=1)
        ShGDense0 = self.ShGDense0(ShGConcat0)
        ShGDense0 = self.relu(ShGDense0)
        ShGDense1 = self.ShGDense1(ShGDense0)
        ShGDense1 = self.relu(ShGDense0)
        ShGOutput0 = self.ShGOutput0(ShGDense1)

        return ShGOutput0
    