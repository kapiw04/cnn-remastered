from torchvision.datasets.mnist import MNIST

train = MNIST(root="data", train=True, download=True)  # 60000
test = MNIST(root="data", train=False, download=True)  # 10000

print(len(train), len(test))
