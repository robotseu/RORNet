import torch
import torch.nn as nn
import torch.nn.functional as F


class Gripper_MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=512):
        super(Gripper_MLP, self).__init__()
        self.linear0 = nn.Linear(in_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.linear2 = nn.Linear(hidden_dim*2, out_dim)
        self.ReLU = nn.ReLU()

    def forward(self, grip):
        x = self.ReLU(self.linear0(grip))
        x = self.ReLU(self.linear1(x))
        out = self.linear2(x)

        return out
