import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MergingGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MergingGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, device=device)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 128, self.hidden_size)


class MergingMLP(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MergingMLP, self).__init__()
        self.hidden_size = hidden_size
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output = F.relu(input)
        output = self.out(output)
        return output


class MergingDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MergingDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output

    def initHidden(self):
        return torch.zeros(1, 128, self.hidden_size, device=device)
