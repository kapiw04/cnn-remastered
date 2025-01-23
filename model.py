from torch import nn
import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy


class Classifier(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), *args, **kwargs):
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

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            flattened_size = self.convolutional_path(dummy_input).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=flattened_size,
                out_features=1024,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=1024,
                out_features=10,  # Number of classes
            ),
        )

    def forward(self, x):
        x = self.convolutional_path(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        return x


class LightningClassifier(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.classifier = Classifier()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.classifier(X)
        loss = F.cross_entropy(y_pred, y)
        self.log("training loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.classifier(X)
        loss = F.cross_entropy(y_pred, y)

        accuracy = Accuracy(task="multiclass", num_classes=10)
        self.log("accuracy", accuracy(y_pred, y))
        self.log("test loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return [optimizer], [lr_scheduler]

    def predict_step(self, batch, batch_idx):
        X, y = batch
        pred = self.classifier(X)
        return pred

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.classifier(X)
        loss = F.cross_entropy(y_pred, y)
        self.log("val loss", loss)
