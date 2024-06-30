import torch.nn as nn


class ForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(100, 1)
        self.relu = nn.PReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.linear(x)
        return x
