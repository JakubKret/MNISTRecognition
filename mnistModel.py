import torch.nn as nn

class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hiddenLayer = nn.Linear(in_features=784,out_features=128)
        self.activation = nn.ReLU()
        self.outputLayer = nn.Linear(in_features=128,out_features=10)

    def forward(self, x):
        x =x.view(-1,784)
        x = self.hiddenLayer(x)
        x = self.activation(x)
        x = self.outputLayer(x)
        return x