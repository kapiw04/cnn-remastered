from torch import nn
import torch
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.final_channels: int = 32

        self.convolutional_path = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
            ),
            self.max_pool,
            nn.Conv2d(
                in_channels=6,
                out_channels=self.final_channels,
                kernel_size=3,
            ),
            self.max_pool,
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.final_channels * 28 * 28,
                out_features=1024,
            ),
            nn.Linear(
                in_features=1024,
                out_features=10,  # number of classes
            ),
        )

    def forward(self, x):
        x = self.convolutional_path(x)
        x = F.relu(x)
        x = torch.flatten(x)
        x = self.classifier(x)

        return x
