import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x):
    return bn(x)

class MLP(nn.Module):
    def __init__(self,n_outputs=10):
        super(MLP, self).__init__()
        self.l1=nn.Linear(784,256)
        self.l2=nn.Linear(256,n_outputs)

    def forward(self, x):
        x=x.view(-1,28*28)
        x=F.relu(self.l1(x))
        x=self.l2(x)
        return x

class CNN(nn.Module):
    def __init__(self, n_outputs=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN_large(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN_large, self).__init__()
        self.c1=nn.Conv2d(input_channel,64,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)
        self.l_c1=nn.Linear(256,n_outputs)
        self.bn1=nn.BatchNorm2d(64,momentum=self.momentum)
        self.bn2=nn.BatchNorm2d(64,momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128,momentum=self.momentum)
        self.bn4=nn.BatchNorm2d(128,momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196,momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16,momentum=self.momentum)

    def forward(self, x):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.l_c1(h)
        return logit

class CNN_Fine(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN_Fine, self).__init__()
        self.c1=nn.Conv2d(input_channel,64,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)
        self.l_c1=nn.Linear(1024,n_outputs)
        self.bn1=nn.BatchNorm2d(64,momentum=self.momentum)
        self.bn2=nn.BatchNorm2d(64,momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128,momentum=self.momentum)
        self.bn4=nn.BatchNorm2d(128,momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196,momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16,momentum=self.momentum)

    def forward(self, x):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.l_c1(h)
        return logit
