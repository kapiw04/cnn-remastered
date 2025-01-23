from model import LightningClassifier
from dataset import get_testloader, get_trainloader
from lightning import Trainer

if __name__ == "__main__":
    batch_size = 32
    lr = 0.01

    cnn_classifier = LightningClassifier(lr)
    test_loader, train_loader = get_testloader(batch_size), get_trainloader(batch_size)

    trainer = Trainer(max_epochs=10)
    trainer.fit(cnn_classifier, train_loader, test_loader)
