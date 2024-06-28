import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, shared_dim: int, exclusive_dim: int):
        """Dense discriminator

        Args:
            shared_dim (int): [Dimension of the shared representation]
            exclusive_dim (int): [Dimension of the exclusive representation]
        """
        super().__init__()
        self.DDense0 = nn.Linear(
            in_features=shared_dim + exclusive_dim, out_features=1000
        )
        self.DDense1 = nn.Linear(in_features=1000, out_features=200)
        self.DOutput0 = nn.Linear(in_features=200, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, DInput: torch.Tensor) -> torch.Tensor:
        """Forward discriminator using the shared and the exclusive representation

        Args:
            DInput (torch.Tensor): Shared & exclusive representation

        Returns:
            torch.Tensor: Probability that the data are fake or real
        """
        DDense0 = self.DDense0(DInput)
        DDense0 = self.relu(DDense0)
        DDense1 = self.DDense1(DDense0)
        DDense1 = self.relu(DDense1)
        DOutput0 = self.DOutput0(DDense1)

        return DOutput0
