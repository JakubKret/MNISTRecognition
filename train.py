import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mnistModel import DigitRecognizer

transform = transforms.ToTensor()

trainDataset = datasets.MNIST('./data', train=True, download=True,transform=transform)

trainLoader = DataLoader(dataset=trainDataset,batch_size=64,shuffle=True)

aiModel = DigitRecognizer()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(aiModel.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(trainLoader):
        optimizer.zero_grad()
        outputs = aiModel(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx} | Loss: {loss.item():.4f}")
print("Finished Training")

checkpoint = {
    "epochs": epoch+1,
    "modelState": aiModel.state_dict(),
    "optimizerState": optimizer.state_dict(),
    "loss": loss.item()
}

torch.save(checkpoint, "checkpoint.pt")
print("Model saved")