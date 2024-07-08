from typing import NamedTuple, Tuple
import torch


class GanLossOutput(NamedTuple):
    discriminator: torch.Tensor
    generator: torch.Tensor


class EncoderOutput(NamedTuple):
    representation: torch.Tensor
    feature: torch.Tensor


class Shapes3DData(NamedTuple):
    
    x: torch.Tensor
    x_floor_hue_label: torch.Tensor
    x_wall_hue_label: torch.Tensor
    x_object_hue_label: torch.Tensor
    scale_label: torch.Tensor
    shape_label: torch.Tensor
    orientation_label: torch.Tensor
    y: torch.Tensor=torch.empty((1,1,1,1))
    y_floor_hue_label: torch.Tensor=torch.empty((1,1,1,1))
    y_wall_hue_label: torch.Tensor=torch.empty((1,1,1,1))
    y_object_hue_label: torch.Tensor=torch.empty((1,1,1,1))


class SDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    global_mutual_M_R_y: torch.Tensor
    global_mutual_M_R_y_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_y: torch.Tensor
    local_mutual_M_R_y_prime: torch.Tensor
    x_floor_hue_logits: torch.Tensor
    x_wall_hue_logits: torch.Tensor
    x_object_hue_logits: torch.Tensor
    y_floor_hue_logits: torch.Tensor
    y_wall_hue_logits: torch.Tensor
    y_object_hue_logits: torch.Tensor
    scale_logits: torch.Tensor
    shape_logits: torch.Tensor
    orientation_logits: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor


class EDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    shared_x: torch.Tensor
    fake_x: torch.Tensor
    shuffle_x: torch.Tensor
    R_x_y: torch.Tensor
    exclusive_x: torch.Tensor


class SDIMLosses(NamedTuple):
    total_loss: torch.Tensor
    encoder_loss: torch.Tensor
    local_mutual_loss: torch.Tensor
    global_mutual_loss: torch.Tensor
    shared_loss: torch.Tensor
    x_floor_hue_classif_loss: torch.Tensor
    x_wall_hue_classif_loss: torch.Tensor
    x_object_hue_classif_loss: torch.Tensor
    y_floor_hue_classif_loss: torch.Tensor
    y_wall_hue_classif_loss: torch.Tensor
    y_object_hue_classif_loss: torch.Tensor
    scale_classif_loss: torch.Tensor
    shape_classif_loss: torch.Tensor
    orientation_classif_loss: torch.Tensor
    x_floor_hue_accuracy: torch.Tensor
    x_wall_hue_accuracy: torch.Tensor
    x_object_hue_accuracy: torch.Tensor
    y_floor_hue_accuracy: torch.Tensor
    y_wall_hue_accuracy: torch.Tensor
    y_object_hue_accuracy: torch.Tensor
    scale_accuracy: torch.Tensor
    shape_accuracy: torch.Tensor
    orientation_accuracy: torch.Tensor


class GenLosses(NamedTuple):
    encoder_loss: torch.Tensor
    local_mutual_loss: torch.Tensor
    global_mutual_loss: torch.Tensor
    gan_loss_g: torch.Tensor


class ClassifLosses(NamedTuple):
    classif_loss: torch.Tensor
    x_floor_hue_classif_loss: torch.Tensor
    x_wall_hue_classif_loss: torch.Tensor
    x_object_hue_classif_loss: torch.Tensor
    x_scale_classif_loss: torch.Tensor
    x_shape_classif_loss: torch.Tensor
    x_orientation_classif_loss: torch.Tensor
    x_floor_hue_accuracy: torch.Tensor
    x_wall_hue_accuracy: torch.Tensor
    x_object_hue_accuracy: torch.Tensor
    x_scale_accuracy: torch.Tensor
    x_shape_accuracy: torch.Tensor
    x_orientation_accuracy: torch.Tensor


class DiscrLosses(NamedTuple):
    gan_loss_d: torch.Tensor


class GeneratorOutputs(NamedTuple):
    real_x: torch.Tensor
    fake_x: torch.Tensor
    real_y: torch.Tensor
    fake_y: torch.Tensor
    exclusive_x: torch.Tensor
    exclusive_y: torch.Tensor


class DiscriminatorOutputs(NamedTuple):
    disentangling_information_x: torch.Tensor
    disentangling_information_x_prime: torch.Tensor


class ClassifierOutputs(NamedTuple):
    x_floor_hue_logits: torch.Tensor
    x_wall_hue_logits: torch.Tensor
    x_object_hue_logits: torch.Tensor
    x_scale_logits: torch.Tensor
    x_shape_logits: torch.Tensor
    x_orientation_logits: torch.Tensor

