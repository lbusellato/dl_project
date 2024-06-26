import torch
import torch.nn as nn
from dl_project.neural_networks.encoder import BaseEncoder
from dl_project.neural_networks.statistics_network import (
    LocalStatisticsNetwork,
    GlobalStatisticsNetwork,
)
from dl_project.neural_networks.gan import Discriminator
from dl_project.utils.custom_typing import (
    EDIMOutputs,
    DiscriminatorOutputs,
    ClassifierOutputs,
)
from dl_project.neural_networks.classifier import Classifier


class EDIM(nn.Module):
    """Exclusive Deep Info Max model. Extract the exclusive information from the images.

    Args:
        img_size (int): Image size (must be squared size)
        channels (int): Number of inputs channels
        shared_dim (int): Dimension of the pretrained shared representation
        exclusive_dim (int): Dimension of the desired shared representation
        trained_encoder_x (nn.Module): Trained encoder on domain X (pretrained with SDIM)
        trained_encoder_y (nn.Module): Trained encoder on domain Y (pretrained with SDIM)
    """

    def __init__(
        self,
        img_size: int,
        channels: int,
        shared_dim: int,
        exclusive_dim: int,
        trained_encoder_x: nn.Module,
        trained_encoder_y: nn.Module,
    ):

        super().__init__()

        self.img_size = img_size
        self.channels = channels
        self.shared_dim = shared_dim
        self.exclusive_dim = exclusive_dim

        self.img_feature_size = 5 # Feature map size
        self.img_feature_channels = 512 # Feature map channels

        # Encoders
        self.sh_enc_x = trained_encoder_x

        self.sh_enc_y = trained_encoder_y

        self.ex_enc_x = BaseEncoder(
            img_size=img_size,
            in_channels=channels,
            num_filters=64,
            kernel_size=4,
            repr_dim=exclusive_dim,
        )

        self.ex_enc_y = BaseEncoder(
            img_size=img_size,
            in_channels=channels,
            num_filters=64,
            kernel_size=4,
            repr_dim=exclusive_dim,
        )
        # Local statistics network
        self.local_stat_x = LocalStatisticsNetwork(
            feature_map_channels=self.img_feature_channels,
            img_feature_channels=2 * self.img_feature_channels
            + self.shared_dim
            + self.exclusive_dim,
            kernel_size=1,
            latent_dim=self.shared_dim,
        )

        self.local_stat_y = LocalStatisticsNetwork(
            feature_map_channels=self.img_feature_channels,
            img_feature_channels=2 * self.img_feature_channels
            + self.shared_dim
            + self.exclusive_dim,
            kernel_size=1,
            latent_dim=self.shared_dim,
        )

        # Global statistics network
        self.global_stat_x = GlobalStatisticsNetwork(
            feature_map_size=self.img_feature_size,
            feature_map_channels=2 * self.img_feature_channels,
            num_filters=32,
            kernel_size=3,
            latent_dim=self.shared_dim + self.exclusive_dim,
        )

        self.global_stat_y = GlobalStatisticsNetwork(
            feature_map_size=self.img_feature_size,
            feature_map_channels=2 * self.img_feature_channels,
            num_filters=32,
            kernel_size=3,
            latent_dim=self.shared_dim + self.exclusive_dim,
        )

        # Gan discriminator (disentangling network)

        self.discriminator_x = Discriminator(
            shared_dim=shared_dim, exclusive_dim=exclusive_dim
        )

        self.discriminator_y = Discriminator(
            shared_dim=shared_dim, exclusive_dim=exclusive_dim
        )
        # Metric nets
        self.x_scale_classifier = Classifier(feature_dim=exclusive_dim, output_dim=8)
        self.x_shape_classifier = Classifier(feature_dim=exclusive_dim, output_dim=4)
        self.x_orientation_classifier = Classifier(feature_dim=exclusive_dim, output_dim=15)
        self.y_scale_classifier = Classifier(feature_dim=exclusive_dim, output_dim=8)
        self.y_shape_classifier = Classifier(feature_dim=exclusive_dim, output_dim=4)
        self.y_orientation_classifier = Classifier(feature_dim=exclusive_dim, output_dim=15)
        self.x_floor_hue_classifier = Classifier(feature_dim=exclusive_dim, output_dim=10)
        self.x_wall_hue_classifier = Classifier(feature_dim=exclusive_dim, output_dim=10)
        self.x_object_hue_classifier = Classifier(feature_dim=exclusive_dim, output_dim=10)
        self.y_floor_hue_classifier = Classifier(feature_dim=exclusive_dim, output_dim=10)
        self.y_wall_hue_classifier = Classifier(feature_dim=exclusive_dim, output_dim=10)
        self.y_object_hue_classifier = Classifier(feature_dim=exclusive_dim, output_dim=10)

    def forward_generator(self, x: torch.Tensor, y: torch.Tensor) -> EDIMOutputs:
        """Forward pass of the generator

        Args:
            x (torch.Tensor): Image from domain X
            y (torch.Tensor): Image from domain Y

        Returns:
            EDIMOutputs: Generator outputs
        """
        # Get the shared and exclusive features from x and y
        shared_x, shared_M_x = self.sh_enc_x(x)
        shared_y, shared_M_y = self.sh_enc_y(y)

        # shared_x, shared_M_x = shared_x.detach(), shared_M_x.detach()
        # shared_y, shared_M_y = shared_y.detach(), shared_M_y.detach()

        exclusive_x, exclusive_M_x = self.ex_enc_x(x)
        exclusive_y, exclusive_M_y = self.ex_enc_y(y)

        # Concat exclusive and shared feature map
        M_x = torch.cat([shared_M_x, exclusive_M_x], dim=1)
        M_y = torch.cat([shared_M_y, exclusive_M_y], dim=1)

        # Shuffle M to create M'
        M_x_prime = torch.cat([M_x[1:], M_x[0].unsqueeze(0)], dim=0)
        M_y_prime = torch.cat([M_y[1:], M_y[0].unsqueeze(0)], dim=0)

        R_x_y = torch.cat([shared_x, exclusive_y], dim=1)
        R_y_x = torch.cat([shared_y, exclusive_x], dim=1)

        # Global mutual information estimation
        global_mutual_M_R_x = self.global_stat_x(M_x, R_y_x)
        global_mutual_M_R_x_prime = self.global_stat_x(M_x_prime, R_y_x)

        global_mutual_M_R_y = self.global_stat_y(M_y, R_x_y)
        global_mutual_M_R_y_prime = self.global_stat_y(M_y_prime, R_x_y)

        # Local mutual information estimation

        local_mutual_M_R_x = self.local_stat_x(M_x, R_y_x)
        local_mutual_M_R_x_prime = self.local_stat_x(M_x_prime, R_y_x)
        local_mutual_M_R_y = self.local_stat_y(M_y, R_x_y)
        local_mutual_M_R_y_prime = self.local_stat_y(M_y_prime, R_x_y)

        # Disentangling discriminator

        shared_x_prime = torch.cat([shared_x[1:], shared_x[0].unsqueeze(0)], axis=0)
        shared_y_prime = torch.cat([shared_y[1:], shared_y[0].unsqueeze(0)], axis=0)

        shuffle_x = torch.cat([shared_y_prime, exclusive_x], axis=1)
        shuffle_y = torch.cat([shared_x_prime, exclusive_y], axis=1)

        fake_x = self.discriminator_x(R_y_x)
        fake_y = self.discriminator_y(R_x_y)

        return EDIMOutputs(
            global_mutual_M_R_x=global_mutual_M_R_x,
            global_mutual_M_R_x_prime=global_mutual_M_R_x_prime,
            global_mutual_M_R_y=global_mutual_M_R_y,
            global_mutual_M_R_y_prime=global_mutual_M_R_y_prime,
            local_mutual_M_R_x=local_mutual_M_R_x,
            local_mutual_M_R_x_prime=local_mutual_M_R_x_prime,
            local_mutual_M_R_y=local_mutual_M_R_y,
            local_mutual_M_R_y_prime=local_mutual_M_R_y_prime,
            shared_x=shared_x,
            shared_y=shared_y,
            fake_x=fake_x,
            fake_y=fake_y,
            R_y_x=R_y_x,
            R_x_y=R_x_y,
            shuffle_x=shuffle_x,
            shuffle_y=shuffle_y,
            exclusive_x=exclusive_x,
            exclusive_y=exclusive_y,
        )

    def forward_discriminator(self, edim_outputs: EDIMOutputs) -> DiscriminatorOutputs:
        """Forward pass of the discriminator

        Args:
            edim_outputs (EDIMOutputs): Outputs from the generator

        Returns:
            DiscriminatorOutputs: Discriminator outputs
        """
        out = edim_outputs
        disentangling_information_x = self.discriminator_x(out.R_y_x.detach())
        disentangling_information_x_prime = self.discriminator_x(out.shuffle_x.detach())
        disentangling_information_y = self.discriminator_y(out.R_x_y.detach())
        disentangling_information_y_prime = self.discriminator_y(out.shuffle_y.detach())
        return DiscriminatorOutputs(
            disentangling_information_x=disentangling_information_x,
            disentangling_information_x_prime=disentangling_information_x_prime,
            disentangling_information_y=disentangling_information_y,
            disentangling_information_y_prime=disentangling_information_y_prime,
        )

    def forward_classifier(self, edim_outputs: EDIMOutputs) -> ClassifierOutputs:
        """Forward pass of the classifiers

        Args:
            edim_outputs (EDIMOutputs): Outputs from the generator

        Returns:
            ClassifierOutputs: Classifiers Outputs
        """
        out = edim_outputs
        # detach because we do not want compute the gradient here
        x_floor_hue_logits = self.x_floor_hue_classifier(out.exclusive_x.detach())
        x_wall_hue_logits = self.x_wall_hue_classifier(out.exclusive_x.detach())
        x_object_hue_logits = self.x_object_hue_classifier(out.exclusive_x.detach())
        y_floor_hue_logits = self.y_floor_hue_classifier(out.exclusive_y.detach())
        y_wall_hue_logits = self.y_wall_hue_classifier(out.exclusive_y.detach())
        y_object_hue_logits = self.y_object_hue_classifier(out.exclusive_y.detach())
        x_scale_logits = self.x_scale_classifier(out.exclusive_x.detach())
        x_shape_logits = self.x_shape_classifier(out.exclusive_x.detach())
        x_orientation_logits = self.x_orientation_classifier(out.exclusive_x.detach())
        y_scale_logits = self.y_scale_classifier(out.exclusive_y.detach())
        y_shape_logits = self.y_shape_classifier(out.exclusive_y.detach())
        y_orientation_logits = self.y_orientation_classifier(out.exclusive_y.detach())

        return ClassifierOutputs(
            x_floor_hue_logits=x_floor_hue_logits,
            x_wall_hue_logits=x_wall_hue_logits,
            x_object_hue_logits=x_object_hue_logits,
            y_floor_hue_logits=y_floor_hue_logits,
            y_wall_hue_logits=y_wall_hue_logits,
            y_object_hue_logits=y_object_hue_logits,
            x_scale_logits=x_scale_logits,
            x_shape_logits=x_shape_logits,
            x_orientation_logits=x_orientation_logits,
            y_scale_logits=y_scale_logits,
            y_shape_logits=y_shape_logits,
            y_orientation_logits=y_orientation_logits,
        )
