import torch.nn as nn
import torch
from dl_project.losses.loss_functions import (
    DJSLoss,
    ClassifLoss,
    DiscriminatorLoss,
    GeneratorLoss,
)
from dl_project.utils.custom_typing import (
    DiscriminatorOutputs,
    ClassifierOutputs,
    EDIMOutputs,
    GenLosses,
    DiscrLosses,
    ClassifLosses,
)


class EDIMLoss(nn.Module):
    """Loss function to extract exclusive information from the image, see paper equation (8)

    Args:
        local_mutual_loss_coeff (float): Coefficient of the local Jensen Shannon loss
        global_mutual_loss_coeff (float): Coefficient of the global Jensen Shannon loss
        disentangling_loss_coeff (float): Coefficient of the Gan loss
    """

    def __init__(
        self,
        local_mutual_loss_coeff: float,
        global_mutual_loss_coeff: float,
        disentangling_loss_coeff: float,
    ):

        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.disentangling_loss_coeff = disentangling_loss_coeff

        self.djs_loss = DJSLoss()
        self.classif_loss = ClassifLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

    def compute_generator_loss(self, edim_outputs: EDIMOutputs) -> GenLosses:
        """Generator loss function

        Args:
            edim_outputs (EDIMOutputs): Output of the forward pass of the exclusive information model

        Returns:
            GenLosses: Generator losses
        """

        # Compute Global mutual loss
        global_mutual_loss_x = self.djs_loss(
            T=edim_outputs.global_mutual_M_R_x,
            T_prime=edim_outputs.global_mutual_M_R_x_prime,
        )
        global_mutual_loss_y = self.djs_loss(
            T=edim_outputs.global_mutual_M_R_y,
            T_prime=edim_outputs.global_mutual_M_R_y_prime,
        )

        global_mutual_loss = (
            global_mutual_loss_x + global_mutual_loss_y
        ) * self.global_mutual_loss_coeff

        # Compute Local mutual loss

        local_mutual_loss_x = self.djs_loss(
            T=edim_outputs.local_mutual_M_R_x,
            T_prime=edim_outputs.local_mutual_M_R_x_prime,
        )
        local_mutual_loss_y = self.djs_loss(
            T=edim_outputs.local_mutual_M_R_y,
            T_prime=edim_outputs.local_mutual_M_R_y_prime,
        )
        local_mutual_loss = (
            local_mutual_loss_x + local_mutual_loss_y
        ) * self.local_mutual_loss_coeff

        gan_loss_x_g = self.generator_loss(fake_logits=edim_outputs.fake_x)
        gan_loss_y_g = self.generator_loss(fake_logits=edim_outputs.fake_y)

        gan_loss_g = (gan_loss_x_g + gan_loss_y_g) * self.disentangling_loss_coeff

        # Get classification error

        # For each network, we assign a loss objective
        encoder_loss = global_mutual_loss + local_mutual_loss + gan_loss_g

        return GenLosses(
            encoder_loss=encoder_loss,
            local_mutual_loss=local_mutual_loss,
            global_mutual_loss=global_mutual_loss,
            gan_loss_g=gan_loss_g,
        )

    def compute_discriminator_loss(
        self, discr_outputs: DiscriminatorOutputs
    ) -> DiscrLosses:
        """Discriminator loss see paper equation (9)

        Args:
            discr_outputs (DiscriminatorOutputs): Output of the forward pass of the discriminators model

        Returns:
            DiscrLosses: Discriminator losses
        """
        gan_loss_x_d = self.discriminator_loss(
            real_logits=discr_outputs.disentangling_information_x_prime,
            fake_logits=discr_outputs.disentangling_information_x,
        )
        gan_loss_y_d = self.discriminator_loss(
            real_logits=discr_outputs.disentangling_information_y_prime,
            fake_logits=discr_outputs.disentangling_information_y,
        )

        gan_loss_d = (gan_loss_x_d + gan_loss_y_d) * self.disentangling_loss_coeff

        return DiscrLosses(gan_loss_d=gan_loss_d)

    def compute_classif_loss(
        self,
        classif_outputs: ClassifierOutputs,
        x_floor_hue_labels: torch.Tensor,
        x_wall_hue_labels: torch.Tensor,
        x_object_hue_labels: torch.Tensor,
        y_floor_hue_labels: torch.Tensor,
        y_wall_hue_labels: torch.Tensor,
        y_object_hue_labels: torch.Tensor,
        scale_labels: torch.Tensor,
        shape_labels: torch.Tensor,
        orientation_labels: torch.Tensor,
    ) -> ClassifLosses:
        """Compute classifiers losses. The accuracy of the classifiers allow to quantify the representations level of disentanglement.

        Args:
            classif_outputs (ClassifierOutputs): Classifiers Outputs
            digit_labels (torch.Tensor): Label of the digit
            color_bg_labels (torch.Tensor): Background color of the images
            color_fg_labels (torch.Tensor): Foreground color of the images

        Returns:
            ClassifLosses: Classifiers losses
        """

        # Get classification error

        x_floor_hue_classif_loss, x_floor_hue_accuracy = self.classif_loss(
            y_pred=classif_outputs.x_floor_hue_logits , target=x_floor_hue_labels
        )
        x_wall_hue_classif_loss, x_wall_hue_accuracy = self.classif_loss(
            y_pred=classif_outputs.x_wall_hue_logits , target=x_wall_hue_labels
        )
        x_object_hue_classif_loss, x_object_hue_accuracy = self.classif_loss(
            y_pred=classif_outputs.x_object_hue_logits , target=x_object_hue_labels
        )
        x_scale_classif_loss, x_scale_accuracy = self.classif_loss(
            y_pred=classif_outputs.x_scale_logits , target=scale_labels
        )
        x_shape_classif_loss, x_shape_accuracy = self.classif_loss(
            y_pred=classif_outputs.x_shape_logits , target=shape_labels
        )
        x_orientation_classif_loss, x_orientation_accuracy = self.classif_loss(
            y_pred=classif_outputs.x_orientation_logits , target=orientation_labels
        )
        
        y_floor_hue_classif_loss, y_floor_hue_accuracy = self.classif_loss(
            y_pred=classif_outputs.y_floor_hue_logits , target=y_floor_hue_labels
        )
        y_wall_hue_classif_loss, y_wall_hue_accuracy = self.classif_loss(
            y_pred=classif_outputs.y_wall_hue_logits , target=y_wall_hue_labels
        )
        y_object_hue_classif_loss, y_object_hue_accuracy = self.classif_loss(
            y_pred=classif_outputs.y_object_hue_logits , target=y_object_hue_labels
        )
        y_scale_classif_loss, y_scale_accuracy = self.classif_loss(
            y_pred=classif_outputs.y_scale_logits , target=scale_labels
        )
        y_shape_classif_loss, y_shape_accuracy = self.classif_loss(
            y_pred=classif_outputs.y_shape_logits , target=shape_labels
        )
        y_orientation_classif_loss, y_orientation_accuracy = self.classif_loss(
            y_pred=classif_outputs.y_orientation_logits , target=orientation_labels
        )

        classif_loss = (
            x_floor_hue_classif_loss
            + x_wall_hue_classif_loss
            + x_object_hue_classif_loss
            + x_scale_classif_loss
            + x_shape_classif_loss
            + x_orientation_classif_loss
            + y_floor_hue_classif_loss
            + y_wall_hue_classif_loss
            + y_object_hue_classif_loss
            + y_scale_classif_loss
            + y_shape_classif_loss
            + y_orientation_classif_loss
        )

        return ClassifLosses(
            classif_loss=classif_loss,
            x_floor_hue_classif_loss=x_floor_hue_classif_loss,
            x_wall_hue_classif_loss=x_wall_hue_classif_loss,
            x_object_hue_classif_loss=x_object_hue_classif_loss,
            x_scale_classif_loss=x_scale_classif_loss,
            x_shape_classif_loss=x_shape_classif_loss,
            x_orientation_classif_loss=x_orientation_classif_loss,
            x_floor_hue_accuracy=x_floor_hue_accuracy,
            x_wall_hue_accuracy=x_wall_hue_accuracy,
            x_object_hue_accuracy=x_object_hue_accuracy,
            x_scale_accuracy=x_scale_accuracy,
            x_shape_accuracy=x_shape_accuracy,
            x_orientation_accuracy=x_orientation_accuracy,
            y_floor_hue_classif_loss=y_floor_hue_classif_loss,
            y_wall_hue_classif_loss=y_wall_hue_classif_loss,
            y_object_hue_classif_loss=y_object_hue_classif_loss,
            y_scale_classif_loss=y_scale_classif_loss,
            y_shape_classif_loss=y_shape_classif_loss,
            y_orientation_classif_loss=y_orientation_classif_loss,
            y_floor_hue_accuracy=y_floor_hue_accuracy,
            y_wall_hue_accuracy=y_wall_hue_accuracy,
            y_object_hue_accuracy=y_object_hue_accuracy,
            y_scale_accuracy=y_scale_accuracy,
            y_shape_accuracy=y_shape_accuracy,
            y_orientation_accuracy=y_orientation_accuracy,
        )
