import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader


def get_trainloader(batch_size):
    transform_sequence = [
        v2.ToImage(),
        v2.RandomRotation(10),
        v2.ToDtype(torch.float32, scale=True),  # Convert to tensor
        v2.RandomResizedCrop(28, scale=(0.8, 1.0)),
        v2.Normalize((0.1307,), (0.3081,)),  # mean and std computed for CNN
    ]

    transform_pipeline = v2.Compose(transform_sequence)

    train = MNIST(
        root="data", train=True, download=True, transform=transform_pipeline
    )  # 60000
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=11)


def get_testloader(batch_size):
    transform_sequence = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Convert to tensor
        v2.Normalize((0.1307,), (0.3081,)),  # mean and std computed for CNN
    ]

    transform_pipeline = v2.Compose(transform_sequence)

    test = MNIST(
        root="data", train=False, download=True, transform=transform_pipeline
    )  # 10000
    return DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=11)
