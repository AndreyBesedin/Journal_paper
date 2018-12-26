import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch
import sup_functions
import models

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--nb_of_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--feature_size', type=int, default=512, help='dimension of the input data')
opts = parser.parse_args()
print(opts)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm2d') != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.label_emb = nn.Embedding(opts.nb_of_classes, opts.nb_of_classes)
    def block(in_feat, out_feat, normalize=True):
      layers = [nn.Linear(in_feat, out_feat)]
      if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers

    self.model = nn.Sequential(
      *block(opts.latent_dim+opts.nb_of_classes, 128, normalize=False),
      *block(128, 256),
      *block(256, 512),
      nn.Linear(512, opts.feature_size),
      nn.Tanh()
    )

  def forward(self, noise, labels):
    # Concatenate label embedding and image to produce input
    gen_input = torch.cat((self.label_emb(labels), noise), -1)
    img = self.model(gen_input)
    return img

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.label_embedding = nn.Embedding(opts.nb_of_classes, opts.nb_of_classes)

    self.model = nn.Sequential(
      nn.Linear(opts.nb_of_classes + opts.feature_size, 512),
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
    d_in = torch.cat(img, self.label_embedding(labels), -1)
    validity = self.model(d_in)
    return validity

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
  generator.cuda()
  discriminator.cuda()
  adversarial_loss.cuda()
  auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
tensor_train = torch.load('./datasets/MNIST/trainset.pth')
tensor_test  = torch.load('./datasets/MNIST/testset.pth')
train_dataset = TensorDataset(tensor_train[0], tensor_train[1])
test_dataset = TensorDataset(tensor_test[0], tensor_test[1])
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
classifier_dict = torch.load('pretrained_models/batch_classifier_MNIST_features_10_classes.pth')
classifier = models.Classifier_MNIST_512_features(10)
classifier.load_state_dict(classifier_dict)
classifier = classifier.cuda()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row):
    """Saves a grid of generated digits ranging from 0 to nb_of_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (100, opts.latent_dim))))
    # Get labels ranging from 0 to nb_of_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    generator.eval()
    gen_imgs = generator(z, labels)
    generator.train()
    return gen_imgs.data, labels

# ----------
#  Training
# ----------

for epoch in range(opts.n_epochs):
  for i, (imgs, labels) in enumerate(dataloader):
    batch_size = imgs.shape[0]

    # Adversarial ground truths
    valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_imgs = Variable(imgs.type(FloatTensor))
    labels = Variable(labels.type(LongTensor))

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise and labels as generator input
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opts.latent_dim))))
    gen_labels = Variable(LongTensor(np.random.randint(0, opts.nb_of_classes, batch_size)))

    # Generate a batch of images
    gen_imgs = generator(z, gen_labels)
    # Loss measures generator's ability to fool the discriminator
    validity, pred_label = discriminator(gen_imgs)
    g_loss = 0.5 * (adversarial_loss(validity, valid) + \
                    auxiliary_loss(pred_label, gen_labels))

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Loss for real images
    real_pred, real_aux = discriminator(real_imgs)
    d_real_loss =  (adversarial_loss(real_pred, valid) + \
                    auxiliary_loss(real_aux, labels)) / 2

    # Loss for fake images
    fake_pred, fake_aux = discriminator(gen_imgs.detach())
    d_fake_loss =  (adversarial_loss(fake_pred, fake) + \
                   auxiliary_loss(fake_aux, gen_labels)) / 2

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    # Calculate discriminator accuracy
    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
    gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

    d_loss.backward()
    optimizer_D.step()

    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, opts.n_epochs, i, len(dataloader),
                                                        d_loss.item(), 100 * d_acc,
                                                        g_loss.item()))
    batches_done = epoch * len(dataloader) + i
 
  gen_samples = sample_image(n_row=10)
  res = classifier(gen_samples[0]).max(1)[1]
  cm = confusion_matrix(gen_samples[1], res)
  print("Average accuracy: " + str(cm.diagonal().mean()/10))
  print('Confusion matrix')
  print(cm/10)