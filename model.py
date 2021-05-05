import torch
import torch.nn as nn
import torchvision
from torch.nn import functional
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self,itemCount):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(itemCount, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, data):
        result = self.dis(data)
        return result  

class Generator(nn.Module):
    def __init__(self, dim, itemCount):
        super(Generator, self).__init__()
        self.inet = nn.Linear(dim, 1024)
        self.mu = nn.Linear(1024, 256)
        self.sigma = nn.Linear(1024, 256)
        self.gnet = nn.ModuleList()
        self.gnet.append(nn.Linear(256, 1024))
        self.gnet.append(nn.Linear(1024, itemCount))

    def encode(self, x):
        h = x
        h = functional.relu(self.inet(h))
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        h = z
        h = functional.relu(self.gnet[0](h))
        return self.gnet[1](h)
         
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else: 
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class RAE(nn.Module):
    def __init__(self, itemCount):
        super(RAE, self).__init__()
        self.encode = nn.Sequential(
        nn.Linear(itemCount, 1024), 
        nn.Sigmoid(), 
        nn.Linear(1024, 64), 
        nn.Sigmoid() 
        )
        self.decode = nn.Sequential(
        nn.Linear(64, 1024), 
        nn.Sigmoid(), 
        nn.Linear(1024, itemCount), 
        nn.Sigmoid()
        )
    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return latent, output

class UAE(nn.Module):
    def __init__(self, dim):
        super(UAE, self).__init__()
        self.encode = nn.Sequential(
        nn.Linear(dim, 128), 
        nn.Sigmoid(), 
        nn.Linear(128, 64), 
        nn.Sigmoid() 
        )
        self.decode = nn.Sequential(
        nn.Linear(64, 128), 
        nn.Sigmoid(), 
        nn.Linear(128, dim), 
        nn.Sigmoid()
        )
    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return latent, output