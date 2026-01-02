import torch
import torch.nn as nn

# Version: 0.3

class Classifier(nn.Module):
    def __init__(self, in_num, drop_value):
        super().__init__()
        self.norm0 =  nn.BatchNorm1d(in_num)
        self.drop0 = nn.Dropout(p=drop_value)
        self.linear1 = nn.Linear(in_num, 256)
        self.norm1 =  nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(p=drop_value)
        self.linear2 = nn.Linear(256, 128)
        self.norm2 =  nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(p=drop_value)
        self.linear3 = nn.Linear(128, 64)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.norm3 =  nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(p=drop_value)
        self.linear4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.norm0(x) 
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.linear3(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.linear4(x)

        return x

    

    