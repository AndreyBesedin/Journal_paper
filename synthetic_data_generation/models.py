import torch
import torch.nn as nn
import torch.nn.functional as F
class autoencoder(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(2048, 1024),
#      linear_block(3072, 1024),
      linear_block(1024, 512),
      linear_block(512, 128),
      linear_block(128, 64),
      nn.Linear(64, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 64),
      linear_block(64, 128),
      linear_block(128, 512),
      linear_block(512, 1024),
      nn.Linear(1024, 2048),
#      nn.Tanh()
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class Net(nn.Module):
  def __init__(self, nb_classes):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(2048, 1024)
    self.fc2 = nn.Linear(1024, 256)
    self.fc3 = nn.Linear(256, 128)
    self.fc4 = nn.Linear(128, nb_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x

class autoencoder2(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_))
    
    super(autoencoder2, self).__init__()
    self.enc1 = linear_block(2048, 1024)
    self.enc2 = linear_block(1024, code_size)
    self.dec1 = linear_block(code_size, 1024)
    self.dec2 = linear_block(1024, 2048)

  def forward(self, x):
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    x = F.relu(self.dec1(x))
    x = self.dec2(x)
    return x

    
