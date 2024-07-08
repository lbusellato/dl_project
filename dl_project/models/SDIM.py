import torch
import torch.nn as nn

from dl_project.neural_networks.encoder import BaseEncoder
from dl_project.neural_networks.statistics_network import (
    LocalStatisticsNetwork,
    GlobalStatisticsNetwork
)
from dl_project.utils.custom_typing import SDIMOutputs
from dl_project.neural_networks.classifier import Classifier


class SDIM(nn.Module):
    def __init__(self, img_size: int, channels: int, shared_dim: int):
        """Shared Deep Info Max model. Extract the shared information from the images

        Args:
            img_size (int): Image size (must be squared size)
            channels (int): Number of inputs channels
            shared_dim (int): Dimension of the desired shared representation
        """
        super().__init__()

        self.img_size = img_size
        self.channels = channels
        self.shared_dim = shared_dim

        self.img_feature_size = 5 # Feature map dimension
        self.img_feature_channels = 512 # Feature map channels

        # Encoders
        self.sh_enc = BaseEncoder(
            img_size=img_size,
            in_channels=channels,
            num_filters=64,
            kernel_size=4,
            repr_dim=shared_dim,
        )

        # Local statistics network
        self.local_stat = LocalStatisticsNetwork(
            feature_map_channels=self.img_feature_channels,
            img_feature_channels=self.img_feature_channels + self.shared_dim,
            kernel_size=1,
            latent_dim=self.shared_dim,
        )

        # Global statistics network
        self.global_stat = GlobalStatisticsNetwork(
            feature_map_size=self.img_feature_size,
            feature_map_channels=self.img_feature_channels,
            num_filters=32,
            kernel_size=3,
            latent_dim=self.shared_dim,
        )

        # Metric nets
        self.scale_classifier = Classifier(feature_dim=shared_dim, output_dim=8)
        self.shape_classifier = Classifier(feature_dim=shared_dim, output_dim=4)
        self.orientation_classifier = Classifier(feature_dim=shared_dim, output_dim=15)
        self.x_floor_hue_classifier = Classifier(feature_dim=shared_dim, output_dim=10)
        self.x_wall_hue_classifier = Classifier(feature_dim=shared_dim, output_dim=10)
        self.x_object_hue_classifier = Classifier(feature_dim=shared_dim, output_dim=10)
        self.y_floor_hue_classifier = Classifier(feature_dim=shared_dim, output_dim=10)
        self.y_wall_hue_classifier = Classifier(feature_dim=shared_dim, output_dim=10)
        self.y_object_hue_classifier = Classifier(feature_dim=shared_dim, output_dim=10)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> SDIMOutputs:
        """Forward pass of the shared model

        Args:
            x (torch.Tensor): Image from domain X
            y (torch.Tensor): Image from domain Y

        Returns:
            SDIMOutputs: Outputs of the SDIM model
        """

        # Get the shared and exclusive features from x and y
        R_y_x, M_x = self.sh_enc(x)
        R_x_y, M_y = self.sh_enc(y)
        # Shuffle M to create M'
        M_x_prime = torch.cat([M_x[1:], M_x[0].unsqueeze(0)], dim=0)
        M_y_prime = torch.cat([M_y[1:], M_y[0].unsqueeze(0)], dim=0)

        # Global mutual information estimation
        global_mutual_M_R_x = self.global_stat(M_x, R_y_x)
        global_mutual_M_R_x_prime = self.global_stat(M_x_prime, R_y_x)

        global_mutual_M_R_y = self.global_stat(M_y, R_x_y)
        global_mutual_M_R_y_prime = self.global_stat(M_y_prime, R_x_y)

        # Local mutual information estimation

        local_mutual_M_R_x = self.local_stat(M_x, R_y_x)
        local_mutual_M_R_x_prime = self.local_stat(M_x_prime, R_y_x)
        local_mutual_M_R_y = self.local_stat(M_y, R_y_x)
        local_mutual_M_R_y_prime = self.local_stat(M_y_prime, R_x_y)

        # Stop the gradient and compute classification task
        x_floor_hue_logits =  self.x_floor_hue_classifier(R_x_y.detach())
        x_wall_hue_logits =   self.x_wall_hue_classifier(R_x_y.detach())
        x_object_hue_logits = self.x_object_hue_classifier(R_x_y.detach())
        y_floor_hue_logits =  self.y_floor_hue_classifier(R_x_y.detach())
        y_wall_hue_logits =   self.y_wall_hue_classifier(R_x_y.detach())
        y_object_hue_logits = self.y_object_hue_classifier(R_x_y.detach())
        scale_logits =        self.scale_classifier(R_x_y.detach())
        shape_logits =        self.shape_classifier(R_x_y.detach())
        orientation_logits =  self.orientation_classifier(R_x_y.detach())

        return SDIMOutputs(
            global_mutual_M_R_x=global_mutual_M_R_x,
            global_mutual_M_R_x_prime=global_mutual_M_R_x_prime,
            global_mutual_M_R_y=global_mutual_M_R_y,
            global_mutual_M_R_y_prime=global_mutual_M_R_y_prime,
            local_mutual_M_R_x=local_mutual_M_R_x,
            local_mutual_M_R_x_prime=local_mutual_M_R_x_prime,
            local_mutual_M_R_y=local_mutual_M_R_y,
            local_mutual_M_R_y_prime=local_mutual_M_R_y_prime,
            x_floor_hue_logits =  x_floor_hue_logits,
            x_wall_hue_logits =   x_wall_hue_logits,
            x_object_hue_logits = x_object_hue_logits,
            y_floor_hue_logits =  y_floor_hue_logits,
            y_wall_hue_logits =   y_wall_hue_logits,
            y_object_hue_logits = y_object_hue_logits,
            scale_logits = scale_logits,
            shape_logits = shape_logits,
            orientation_logits = orientation_logits,
            shared_x=R_x_y,
            shared_y=R_y_x,
        )
