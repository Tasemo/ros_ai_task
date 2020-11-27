
#!/usr/bin/env python

import torch
import torchvision
from multiprocessing import cpu_count
from model import MNISTModel
import matplotlib.pyplot as plt

def trainModel():
    model.train()
    total_loss = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossFunction(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    total_loss /= len(train_loader.dataset)
    train_losses.append(total_loss)

def testModel():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += lossFunction(output, target).item()
            prediced = output.argmax(dim=1, keepdim=True)
            correct += prediced.eq(target.view_as(prediced)).sum().item()
    data_size = len(test_loader.dataset)
    total_loss /= data_size
    test_losses.append(total_loss)
    percentage = 100. * correct / data_size
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{}, ({:.1f}%)".format(total_loss, correct, data_size, percentage))

def getLoader(training):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root="data", train=training, download=training, transform=transform)
    return torch.utils.data.DataLoader(dataset, num_workers=cpu_count(), shuffle=True, batch_size=200)

def savePlot():
    plt.xlabel("Epoch")
    plt.ylabel("Aerage Loss")
    plt.plot(train_losses, label="Training Set")
    plt.plot(test_losses, label="Test Set")
    plt.legend()
    plt.savefig("model_losses.svg")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = getLoader(True)
    test_loader = getLoader(False)
    model = MNISTModel(28, 28).to(device)
    optimizer = torch.optim.Adadelta(model.parameters())
    lossFunction = torch.nn.NLLLoss()
    train_losses = []
    test_losses = []
    for epoch in range(1, 11):
        trainModel()
        testModel()
    torch.save(model.state_dict(), "trainedModel.pt")
    savePlot()
