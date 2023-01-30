import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
import numpy as np

class MDNclustering(torch.nn.Module):
    def __init__(self, nhidden):
        super().__init__()
        self.inputlay = torch.nn.Sequential(nn.Linear(1 ,20),nn.LeakyReLU(0.1))
        
        params = np.linspace(20,200,nhidden)
        modules = []
        for k in range(nhidden-1):
            modules.append(nn.Linear(int(params[k]) ,int(params[k+1])))
            modules.append(nn.LeakyReLU(0.1))  
        self.hiddenlay = nn.Sequential(*modules)
        
        
        self.logalphas = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100, 10))                
        self.means = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100, 10))
        self.logstds = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100 ,10))
        
    def forward(self, inp):
        
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        mu = self.means(x)
        logsig = self.logstds(x)
        logalpha=self.logalphas(x)
        
        logsig = torch.clamp(logsig,-5,5)
        mu = torch.clamp(mu,-100,100)
        
        logalpha = logalpha - torch.logsumexp(logalpha,1)[:,None] 


        
        return logalpha, mu, logsig   