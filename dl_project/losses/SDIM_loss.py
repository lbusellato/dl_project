import torch
import torch.nn as nn
from dl_project.losses.loss_functions import DJSLoss, ClassifLoss
from dl_project.utils.custom_typing import SDIMLosses, SDIMOutputs


class SDIMLoss(nn.Module):
    """Loss function to extract shared information from the image, see paper equation (5)

    Args:
        local_mutual_loss_coeff (float): Coefficient of the local Jensen Shannon loss
        global_mutual_loss_coeff (float): Coefficient of the global Jensen Shannon loss
        shared_loss_coeff (float): Coefficient of L1 loss, see paper equation (6)
    """

    def __init__(
        self,
        local_mutual_loss_coeff: float,
        global_mutual_loss_coeff: float,
        shared_loss_coeff: float,
    ):

        super().__init__()
        self.local_mutual_loss_coeff = local_mutual_loss_coeff
        self.global_mutual_loss_coeff = global_mutual_loss_coeff
        self.shared_loss_coeff = shared_loss_coeff

        self.djs_loss = DJSLoss()
        self.classif_loss = ClassifLoss()
        self.l1_loss = nn.L1Loss()  # see equation (6)

    def __call__(
        self,
        sdim_outputs: SDIMOutputs,
        x_floor_hue_labels: torch.Tensor,
        x_wall_hue_labels: torch.Tensor,
        x_object_hue_labels: torch.Tensor,
        y_floor_hue_labels: torch.Tensor,
        y_wall_hue_labels: torch.Tensor,
        y_object_hue_labels: torch.Tensor,
        scale_labels: torch.Tensor,
        shape_labels: torch.Tensor,
        orientation_labels: torch.Tensor,
    ) -> SDIMLosses:
        """Compute all the loss functions needed to extract the shared part

        Args:
            sdim_outputs (SDIMOutputs): Output of the forward pass of the shared information model
            floor_hue_labels (torch.Tensor): Label of the floor's color
            wall_hue_labels (torch.Tensor): Label of the wall's color
            object_hue_labels (torch.Tensor): Label of the object's color
            scale_labels (torch.Tensor): Label of the object's scale
            shape_labels (torch.Tensor): Label of the object's shape
            orientation_labels (torch.Tensor): Label of the object's orientation

        Returns:
            SDIMLosses: Shared information losses
        """

        # Compute Global mutual loss
        global_mutual_loss_x = self.djs_loss(
            T=sdim_outputs.global_mutual_M_R_x,
            T_prime=sdim_outputs.global_mutual_M_R_x_prime,
        )
        global_mutual_loss_y = self.djs_loss(
            T=sdim_outputs.global_mutual_M_R_y,
            T_prime=sdim_outputs.global_mutual_M_R_y_prime,
        )
        global_mutual_loss = (
            global_mutual_loss_x + global_mutual_loss_y
        ) * self.global_mutual_loss_coeff

        # Compute Local mutual loss

        local_mutual_loss_x = self.djs_loss(
            T=sdim_outputs.local_mutual_M_R_x,
            T_prime=sdim_outputs.local_mutual_M_R_x_prime,
        )
        local_mutual_loss_y = self.djs_loss(
            T=sdim_outputs.local_mutual_M_R_y,
            T_prime=sdim_outputs.local_mutual_M_R_y_prime,
        )
        local_mutual_loss = (
            local_mutual_loss_x + local_mutual_loss_y
        ) * self.local_mutual_loss_coeff

        # Compute L1 on shared features
        shared_loss = self.l1_loss(sdim_outputs.shared_x, sdim_outputs.shared_y)
        shared_loss = shared_loss * self.shared_loss_coeff

        # Get classification error

        x_floor_hue_classif_loss, x_floor_hue_accuracy = self.classif_loss(
            y_pred=sdim_outputs.x_floor_hue_logits , target=x_floor_hue_labels
        )
        x_wall_hue_classif_loss, x_wall_hue_accuracy = self.classif_loss(
            y_pred=sdim_outputs.x_wall_hue_logits , target=x_wall_hue_labels
        )
        x_object_hue_classif_loss, x_object_hue_accuracy = self.classif_loss(
            y_pred=sdim_outputs.x_object_hue_logits , target=x_object_hue_labels
        )
        y_floor_hue_classif_loss, y_floor_hue_accuracy = self.classif_loss(
            y_pred=sdim_outputs.y_floor_hue_logits , target=y_floor_hue_labels
        )
        y_wall_hue_classif_loss, y_wall_hue_accuracy = self.classif_loss(
            y_pred=sdim_outputs.y_wall_hue_logits , target=y_wall_hue_labels
        )
        y_object_hue_classif_loss, y_object_hue_accuracy = self.classif_loss(
            y_pred=sdim_outputs.y_object_hue_logits , target=y_object_hue_labels
        )
        scale_classif_loss, scale_accuracy = self.classif_loss(
            y_pred=sdim_outputs.scale_logits , target=scale_labels
        )
        shape_classif_loss, shape_accuracy = self.classif_loss(
            y_pred=sdim_outputs.shape_logits , target=shape_labels
        )
        orientation_classif_loss, orientation_accuracy = self.classif_loss(
            y_pred=sdim_outputs.orientation_logits , target=orientation_labels
        )
        encoder_loss = global_mutual_loss + local_mutual_loss + shared_loss

        total_loss = (
            global_mutual_loss
            + local_mutual_loss
            + shared_loss
            + x_floor_hue_classif_loss
            + x_wall_hue_classif_loss
            + x_object_hue_classif_loss
            + y_floor_hue_classif_loss
            + y_wall_hue_classif_loss
            + y_object_hue_classif_loss
            + scale_classif_loss
            + shape_classif_loss
            + orientation_classif_loss
        )

        return SDIMLosses(
            total_loss=total_loss,
            encoder_loss=encoder_loss,
            local_mutual_loss=local_mutual_loss,
            global_mutual_loss=global_mutual_loss,
            shared_loss=shared_loss,
            x_floor_hue_classif_loss= x_floor_hue_classif_loss,
            x_wall_hue_classif_loss=  x_wall_hue_classif_loss,
            x_object_hue_classif_loss=x_object_hue_classif_loss,
            y_floor_hue_classif_loss= y_floor_hue_classif_loss,
            y_wall_hue_classif_loss=  y_wall_hue_classif_loss,
            y_object_hue_classif_loss=y_object_hue_classif_loss,
            scale_classif_loss=scale_classif_loss,
            shape_classif_loss=shape_classif_loss,
            orientation_classif_loss=orientation_classif_loss,
            x_floor_hue_accuracy= x_floor_hue_accuracy,
            x_wall_hue_accuracy=  x_wall_hue_accuracy,
            x_object_hue_accuracy=x_object_hue_accuracy,
            y_floor_hue_accuracy= y_floor_hue_accuracy,
            y_wall_hue_accuracy=  y_wall_hue_accuracy,
            y_object_hue_accuracy=y_object_hue_accuracy,
            scale_accuracy=scale_accuracy,
            shape_accuracy=shape_accuracy,
            orientation_accuracy=orientation_accuracy
        )
