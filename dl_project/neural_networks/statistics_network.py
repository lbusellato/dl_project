import torch.nn as nn
import torch


class LocalStatisticsNetwork(nn.Module):
    def __init__(self, 
                 feature_map_channels: int, 
                 img_feature_channels: int,
                 kernel_size: int):
        """Local statistics network

        Args:
            feature_map_channels (int): Number of channels in the input feature maps
            img_feature_channels (int): [Number of input channels]
            kernel_size (int): Convolution kernel size
        """

        super().__init__()

        self.LConv0 = nn.Conv2d(
            in_channels=img_feature_channels, 
            out_channels=feature_map_channels, 
            kernel_size=kernel_size, 
            stride=1,device='cuda'
        )
        self.LConv1 = nn.Conv2d(
            in_channels=feature_map_channels, 
            out_channels=feature_map_channels, 
            kernel_size=kernel_size, 
            stride=1,device='cuda'
        )
        self.LOutput0 = nn.Conv2d(
            in_channels=feature_map_channels, 
            out_channels=1, 
            kernel_size=kernel_size, 
            stride=1,device='cuda')
        
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

    def forward(self, LInput0: torch.Tensor, LInput1: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Local Statistics Network.

        Parameters
        ----------
        LInput0 : torch.Tensor
            Feature map
        LInput0 : torch.Tensor
            Feature representation

        Returns
        -------
        torch.Tensor
            The local mutual information.
        """
        LConcat0 = self.tile_and_concat(LInput0, LInput1)
        LConv0 = self.LConv0(LConcat0)
        LConv0 = self.relu(LConv0)
        LConv1 = self.LConv1(LConv0)
        LConv1 = self.relu(LConv1)
        LOutput0 = self.LOutput0(LConv1)

        return LOutput0


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

        self.GConv0 = nn.Conv2d(
            in_channels=feature_map_channels,
            out_channels=num_filters * 2 ** 1,
            kernel_size=kernel_size,
            stride=1,device='cuda'
        )
        self.GConv1 = nn.Conv2d(
            in_channels=num_filters * 2 ** 1,
            out_channels=num_filters * 2 ** 0,
            kernel_size=kernel_size,
            stride=1,device='cuda'
        )

        # Compute the size of the input features for the first dense layer
        conv_output_size = feature_map_size - 2 * (kernel_size - 1)
        flattened_size = num_filters * 2 ** 0 * conv_output_size * conv_output_size
        concat_size = flattened_size + latent_dim        
        self.GDense0 = nn.Linear(
            in_features=concat_size,
            out_features=512,device='cuda'
        )
        self.GDense1 = nn.Linear(in_features=512, out_features=512,device='cuda')
        self.GOutput0 = nn.Linear(in_features=512, out_features=1,device='cuda')

        self.flatten = nn.Flatten()

        self.relu = nn.ReLU()

    def forward(
        self, GInput0: torch.Tensor, GInput1: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the Global Statistics Network

        Parameters
        ----------
        GInput0 : torch.Tensor
            The feature map.
        GInput1 : torch.Tensor
            The feature representation.

        Returns
        -------
        torch.Tensor
            The global mutual information.
        """
        GConv0 = self.GConv0(GInput0)
        GConv0 = self.relu(GConv0)
        GConv1 = self.GConv1(GConv0)
        GFlat0 = self.flatten(GConv1)
        GConcat0 = torch.cat([GFlat0, GInput1], dim=1)
        GDense0 = self.GDense0(GConcat0)
        GDense0 = self.relu(GDense0)
        GDense1 = self.GDense1(GDense0)
        GDense1 = self.relu(GDense0)
        GOutput0 = self.GOutput0(GDense1)

        return GOutput0
    