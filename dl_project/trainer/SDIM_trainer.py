import torch.optim as optim
import torch
from dl_project.models.SDIM import SDIM
from dl_project.losses.SDIM_loss import SDIMLoss
from torch.cuda.amp import GradScaler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import mlflow
import mlflow.pytorch as mpy

from dl_project.utils.custom_typing import SDIMOutputs, SDIMLosses


class SDIMTrainer:
    def __init__(
        self,
        model: SDIM,
        loss: SDIMLoss,
        dataset_train: Dataset,
        learning_rate: float,
        batch_size: int,
        device: str,
    ):
        """Shared Deep Info Max trainer

        Args:
            model (SDIM): Shared model backbone
            loss (SDIMLoss): Shared loss
            dataset_train (Dataset): Train dataset
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            device (str): Device among cuda/cpu
        """
        self.train_dataloader = DataLoader(dataset_train, batch_size=batch_size)
        self.model = model.to(device)
        self.loss = loss
        self.device = device

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network optimizers
        self.optimizer_encoder = optim.Adam(
            model.sh_enc.parameters(), lr=learning_rate
        )
        self.optimizer_local_stat = optim.Adam(
            model.local_stat.parameters(), lr=learning_rate
        )
        self.optimizer_global_stat = optim.Adam(
            model.global_stat.parameters(), lr=learning_rate
        )

        self.x_optimizer_floor_hue_classifier = optim.Adam(
            model.x_floor_hue_classifier.parameters(), lr=learning_rate
        )
        self.x_optimizer_wall_hue_classifier = optim.Adam(
            model.x_wall_hue_classifier.parameters(), lr=learning_rate
        )
        self.x_optimizer_object_hue_classifier = optim.Adam(
            model.x_object_hue_classifier.parameters(), lr=learning_rate
        )
        self.y_optimizer_floor_hue_classifier = optim.Adam(
            model.y_floor_hue_classifier.parameters(), lr=learning_rate
        )
        self.y_optimizer_wall_hue_classifier = optim.Adam(
            model.y_wall_hue_classifier.parameters(), lr=learning_rate
        )
        self.y_optimizer_object_hue_classifier = optim.Adam(
            model.y_object_hue_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_scale_classifier = optim.Adam(
            model.scale_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_shape_classifier = optim.Adam(
            model.shape_classifier.parameters(), lr=learning_rate
        )
        self.optimizer_orientation_classifier = optim.Adam(
            model.orientation_classifier.parameters(), lr=learning_rate
        )

        self.scaler = GradScaler()

    def gradient_zero(self):
        """Set all the networks gradient to zero"""
        self.optimizer_encoder.zero_grad(set_to_none=True)

        self.optimizer_local_stat.zero_grad(set_to_none=True)

        self.optimizer_global_stat.zero_grad(set_to_none=True)

        self.x_optimizer_floor_hue_classifier.zero_grad(set_to_none=True)
        self.x_optimizer_wall_hue_classifier.zero_grad(set_to_none=True)
        self.x_optimizer_object_hue_classifier.zero_grad(set_to_none=True)
        self.y_optimizer_floor_hue_classifier.zero_grad(set_to_none=True)
        self.y_optimizer_wall_hue_classifier.zero_grad(set_to_none=True)
        self.y_optimizer_object_hue_classifier.zero_grad(set_to_none=True)
        self.optimizer_scale_classifier.zero_grad(set_to_none=True)
        self.optimizer_shape_classifier.zero_grad(set_to_none=True)
        self.optimizer_orientation_classifier.zero_grad(set_to_none=True)

    def compute_gradient(
        self,
        sdim_output: SDIMOutputs,
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
        """Compute the SDIM gradient

        Args:
            sdim_outputs (SDIMOutputs): Output of the forward pass of the shared information model
            x_floor_hue_labels (torch.Tensor): Label of the floor's color for image X
            x_wall_hue_labels (torch.Tensor): Label of the wall's color for image X
            x_object_hue_labels (torch.Tensor): Label of the object's color for image X
            y_floor_hue_labels (torch.Tensor): Label of the floor's color for image Y
            y_wall_hue_labels (torch.Tensor): Label of the wall's color for image Y
            y_object_hue_labels (torch.Tensor): Label of the object's color for image Y
            scale_labels (torch.Tensor): Label of the object's scale
            shape_labels (torch.Tensor): Label of the object's shape
            orientation_labels (torch.Tensor): Label of the object's orientation

        Returns:
            SDIMLosses: [Shared model losses value]
        """
        losses = self.loss(
            sdim_outputs=sdim_output,
            x_floor_hue_labels=x_floor_hue_labels,
            x_wall_hue_labels=x_wall_hue_labels,
            x_object_hue_labels=x_object_hue_labels,
            y_floor_hue_labels=y_floor_hue_labels,
            y_wall_hue_labels=y_wall_hue_labels,
            y_object_hue_labels=y_object_hue_labels,
            scale_labels=scale_labels,
            shape_labels=shape_labels,
            orientation_labels=orientation_labels,
        )
        #losses.total_loss.backward()
        self.scaler.scale(losses.total_loss).backward()
        return losses

    def gradient_step(self):
        """Make an optimisation step for all the networks"""
        self.scaler.step(self.optimizer_encoder)
        self.scaler.step(self.optimizer_local_stat)
        self.scaler.step(self.optimizer_global_stat)
        self.scaler.step(self.x_optimizer_floor_hue_classifier)
        self.scaler.step(self.x_optimizer_wall_hue_classifier)
        self.scaler.step(self.x_optimizer_object_hue_classifier)
        self.scaler.step(self.y_optimizer_floor_hue_classifier)
        self.scaler.step(self.y_optimizer_wall_hue_classifier)
        self.scaler.step(self.y_optimizer_object_hue_classifier)
        self.scaler.step(self.optimizer_scale_classifier)
        self.scaler.step(self.optimizer_shape_classifier)
        self.scaler.step(self.optimizer_orientation_classifier)
        
    def train(self, epochs, xp_name="test"):
        """Trained shared model and log losses and accuracy on Mlflow.

        Args:
            epochs (int): Number of epochs
            xp_name (str, optional): Name of the Mlfow experiment. Defaults to "test".
        """
        mlflow.set_experiment(experiment_name=xp_name)
        with mlflow.start_run() as run:
            mlflow.log_param("Batch size", self.batch_size)
            mlflow.log_param("Learning rate", self.learning_rate)
            mlflow.log_param("Local mutual weight", self.loss.local_mutual_loss_coeff)
            mlflow.log_param("Global mutual weight", self.loss.global_mutual_loss_coeff)
            mlflow.log_param("L1 weight", self.loss.shared_loss_coeff)
            log_step = 0
            for _ in tqdm(range(epochs)):
                sample = next(iter(self.train_dataloader))
                self.gradient_zero()
                sdim_outputs = self.model(
                    x=sample.x.to(self.device), y=sample.y.to(self.device)
                )
                losses = self.compute_gradient(
                    sdim_output=sdim_outputs,
                    x_floor_hue_labels=sample.x_floor_hue_label.to(self.device),
                    x_wall_hue_labels=sample.x_wall_hue_label.to(self.device),
                    x_object_hue_labels=sample.x_object_hue_label.to(self.device),
                    y_floor_hue_labels=sample.y_floor_hue_label.to(self.device),
                    y_wall_hue_labels=sample.y_wall_hue_label.to(self.device),
                    y_object_hue_labels=sample.y_object_hue_label.to(self.device),
                    scale_labels=sample.scale_label.to(self.device),
                    shape_labels=sample.shape_label.to(self.device),
                    orientation_labels=sample.orientation_label.to(self.device),
                )
                dict_losses = losses._asdict()
                mlflow.log_metrics(
                    {k: v.item() for k, v in dict_losses.items()}, step=log_step
                )
                log_step += 1
                self.gradient_step()
                self.scaler.update()

            encoder_x_path, encoder_y_path = "sh_encoder_x", "sh_encoder_y"
            mpy.log_state_dict(self.model.sh_enc.state_dict(), encoder_x_path)
            mpy.log_state_dict(self.model.sh_enc.state_dict(), encoder_y_path)
