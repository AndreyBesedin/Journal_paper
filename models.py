import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier_MNIST_28x28(nn.Module):
  def __init__(self, nb_of_classes):
    super(Classifier_MNIST_28x28, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
    self.bn2 = nn.BatchNorm2d(64)
    self.fc1  = nn.Linear(64*28*28, 1024)
    self.fc2 = nn.Linear(1024, nb_of_classes)

  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, 1, 28,28)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = x.view(batch_size, 64*28*28)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class Classifier_2048_features(nn.Module):
  def __init__(self, nb_classes):
    super(Classifier_2048_features, self).__init__()
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
  
class Autoencoder_2048(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(Autoencoder_2048, self).__init__()
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

    
