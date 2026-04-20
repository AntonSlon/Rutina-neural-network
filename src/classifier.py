import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.net(x)

