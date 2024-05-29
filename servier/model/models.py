import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        first_hidden_size: int,
        second_hidden_size: int,
        dropout: float,
    ):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, first_hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(first_hidden_size, second_hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(second_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.to(self.fc1.weight.device)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)

        return out
