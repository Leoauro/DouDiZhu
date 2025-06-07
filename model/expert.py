import torch.nn
from torch import nn

class Expert(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int, dropout=0.1):
        super(Expert, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.layer1 = nn.Linear(self.in_features, self.hidden_dim, bias=False)
        self.act1 = nn.SiLU()
        self.layer2 = nn.Linear(self.hidden_dim, self.out_features, bias=False)
        self.act2 = nn.SiLU()
        self.layer3 = nn.Linear(self.in_features, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.layer1(x)
        out = self.act1(out)
        out = self.layer2(out * self.layer3(x))
        out = self.act2(out)
        out = self.dropout(out)
        return out
