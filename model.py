import torch.nn as nn


class ForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=50, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(100, 1, bias=True)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x
