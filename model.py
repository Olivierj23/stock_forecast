import torch.nn as nn


class ForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=50, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.linear2 = nn.Linear(100, 1)
        self.linear1 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.relu = nn.PReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = self.linear3(x)
        x = self.linear2(x)

        return x
