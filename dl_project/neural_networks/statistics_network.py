import torch.nn as nn
import torch


class LocalStatisticsNetwork(nn.Module):
    def __init__(self, 
                 feature_map_channels: int, 
                 img_feature_channels: int,
                 kernel_size: int,
                 latent_dim: int):
        """Local statistics network

        Args:
            feature_map_channels (int): Number of channels in the input feature maps
            img_feature_channels (int): [Number of input channels]
            kernel_size (int): Convolution kernel size
            latent_dim (int): Dimension of the representationss
        """

        super().__init__()

        self.ShLConv0 = nn.Conv2d(
            in_channels=img_feature_channels, 
            out_channels=feature_map_channels, 
            kernel_size=kernel_size, 
            stride=1
        )
        self.ShLConv1 = nn.Conv2d(
            in_channels=feature_map_channels, 
            out_channels=feature_map_channels, 
            kernel_size=kernel_size, 
            stride=1
        )
        self.ShLOutput0 = nn.Conv2d(
            in_channels=feature_map_channels, 
            out_channels=1, 
            kernel_size=kernel_size, 
            stride=1)
        
        self.relu = nn.ReLU()

    def tile_and_concat(self, tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Merge 1D and 2D tensor (use to aggregate feature maps and representation
        and compute local mutual information estimation)

        Args:
            tensor (torch.Tensor): 2D tensor (feature maps)
            vector (torch.Tensor): 1D tensor representation

        Returns:
            torch.Tensor: Merged tensor (2D)
        """

        B, _, H, W = tensor.size()
        vector = vector.unsqueeze(2).unsqueeze(2)
        expanded_vector = vector.expand((B, vector.size(1), H, W))
        return torch.cat([tensor, expanded_vector], dim=1)

    def forward(self, ShLInput0: torch.Tensor, ShLInput1: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Local Statistics Network.

        Parameters
        ----------
        ShLConcat0 : torch.Tensor
            The concatenation of the encoded feature map and tiled feature representation (see tile_and_concat).

        Returns
        -------
        torch.Tensor
            The local mutual information.
        """
        ShLConcat0 = self.tile_and_concat(ShLInput0, ShLInput1)
        ShLConv0 = self.ShLConv0(ShLConcat0)
        ShLConv0 = self.relu(ShLConv0)
        ShLConv1 = self.ShLConv1(ShLConv0)
        ShLConv1 = self.relu(ShLConv1)
        ShLOutput0 = self.ShLOutput0(ShLConv1)

        return ShLOutput0


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
        self, 
        feature_map_size: int, 
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
            out_features=512,
        )
        self.ShGDense1 = nn.Linear(in_features=512, out_features=512)
        self.ShGOutput0 = nn.Linear(in_features=512, out_features=1)

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
    