import torch
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import v2
import matplotlib.pyplot as plt

train = MNIST(root="data", train=True, download=True)  # 60000
test = MNIST(root="data", train=False, download=True)  # 10000

transform_sequence = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # Convert to tensor
    v2.Normalize((0.1307,), (0.3081,)),  # mean and std computed for CNN
    v2.CenterCrop(16),
    v2.RandomResize(min_size=16, max_size=32),
]

transform = v2.Compose(transform_sequence)

if __name__ == "__main__":
    example_image = train[0][0]

    transform_sequence.append(v2.ToPILImage())
    transform = v2.Compose(transform_sequence)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(example_image, cmap="gray")
    ax[1].imshow(transform(example_image), cmap="gray")

    plt.show()
