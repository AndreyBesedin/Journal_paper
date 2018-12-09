import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier_MNIST_28x28(nn.Module):
  def __init__(self, nb_classes):
    super(Classifier_MNIST_28x28, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 64, 5)
    self.fc1 = nn.Linear(64 * 5 * 5, 2048)
    self.fc2 = nn.Linear(2048, 120)
    self.fc3 = nn.Linear(120, 84)
    self.fc4 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 64 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
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
  
class autoencoder_2048(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(autoencoder_2048, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(2048, 1024),
#     linear_block(3072, 1024),
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
      nn.Tanh()
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


class autoencoder_MNIST(nn.Module):
  def __init__(self, code_size):
    super(autoencoder_MNIST, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(1, 32, 5, 1, 2),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=2), 
      nn.Conv2d(32, 64, 5, 1, 2),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
      nn.Conv2d(64, code_size, 5, 1, 2),
      nn.BatchNorm2d(code_size),
      nn.ReLU(True),
      nn.MaxPool2d(7, stride=None)  # b, 8, 2, 2
    )
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(code_size, 32, 7, stride=1, padding=0),  # b, 16, 5, 5
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 64, 6, stride=2, padding=1),  # b, 8, 15, 15
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 16, 7, stride=1, padding=0),  # b, 1, 28, 28
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.ConvTranspose2d(16, 1, 7, stride=1, padding=0),  # b, 1, 28, 28
      nn.Tanh()
    )
  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, 1, 28,28)
    x = self.encoder(x)
    x = self.decoder(x)
    return x

# CGAN generator-discriminator pair
class CGAN_Generator(nn.Module):
  def __init__(self):
    super(CGAN_Generator, self).__init__()
    self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
    def block(in_feat, out_feat, normalize=True):
      layers = [  nn.Linear(in_feat, out_feat)]
      if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers
    self.model = nn.Sequential(
      *block(opt.latent_dim+opt.n_classes, 128, normalize=False),
      *block(128, 256),
      *block(256, 512),
      *block(512, 1024),
      nn.Linear(1024, int(np.prod(img_shape))),
      nn.Tanh()
    )

  def forward(self, noise, labels):
    # Concatenate label embedding and image to produce input
    gen_input = torch.cat((self.label_emb(labels), noise), -1)
    img = self.model(gen_input)
    img = img.view(img.size(0), *img_shape)
    return img

class CGAN_Discriminator(nn.Module):
  def __init__(self):
    super(CGAN_Discriminator, self).__init__()
    self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
    self.model = nn.Sequential(
      nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 512),
      nn.Dropout(0.4),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 512),
      nn.Dropout(0.4),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Linear(512, 1)
    )

  def forward(self, img, labels):
    # Concatenate label embedding and image to produce input
    d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
    validity = self.model(d_in)
    return validity

