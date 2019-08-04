import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        actv = nn.LeakyReLU(self.fc1(x))
        return nn.ReLU(self.fc2(actv))

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x, a):
        actv = nn.LeakyReLU(self.fc1(torch.cat([x,a],dim=1)))
        return nn.Tanh(self.fc2(actv))
