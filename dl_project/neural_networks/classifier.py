import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, feature_dim: int, output_dim, units: int = 32) -> None:
        """Simple dense classifier

        Args:
            feature_dim (int): [Number of input feature]
            output_dim ([type]): [Number of classes]
            units (int, optional): [Intermediate layers dimension]. Defaults to 15.
        """
        super().__init__()

        self.CDense0 = nn.Linear(in_features=feature_dim, out_features=units)
        self.bn1 = nn.BatchNorm1d(num_features=units)
        self.CDense1 = nn.Linear(in_features=units, out_features=output_dim)
        self.bn2 = nn.BatchNorm1d(num_features=output_dim)
        self.CDense2 = nn.Linear(in_features=output_dim, out_features=output_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, CInput0: torch.Tensor):
        """Forward pass for the classifier.

        Parameters
        ----------
        CInput0 : (torch.Tensor)
            Feature representations.

        Returns
        -------
        torch.Tensor
            Classifier logits.
        """
        CDense0 = self.CDense0(CInput0)
        CDense0 = self.bn1(CDense0)
        CDense0 = self.relu(CDense0)
        CDense1 = self.CDense1(CDense0)
        CDense1 = self.bn2(CDense1)
        CDense1 = self.relu(CDense1)
        CDense2 = self.CDense2(CDense1)
        COutput0 = self.softmax(CDense2)

        return COutput0