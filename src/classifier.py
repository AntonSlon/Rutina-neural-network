import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        return self.net(x)

