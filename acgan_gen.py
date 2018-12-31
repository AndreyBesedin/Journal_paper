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
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1000, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--nb_of_classes', type=int, default=500, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--feature_size', type=int, default=2048, help='dimension of the input data')
parser.add_argument('--batches_per_epoch', type=int, default=100, help='nb of batches we generate at each epoch of classifier training')
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
      *block(opts.latent_dim+opts.nb_of_classes, 512, normalize=False),
      *block(512, 512),
      *block(512, 1024),
      nn.Linear(1024, opts.feature_size),
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

    def discriminator_block(in_features, out_features, bn=True):
      """Returns layers of each discriminator block"""
      block = [   nn.Linear(in_features, out_features),
                  nn.LeakyReLU(0.2, inplace=True),
                  nn.Dropout(0.25)]
      if bn:
        block.append(nn.BatchNorm1d(out_features))
      return block

    self.linear_blocks = nn.Sequential(
      *discriminator_block(opts.feature_size, 1024, bn=False),
      *discriminator_block(1024, 512),
      *discriminator_block(512, 512),
    )

    # Output layers
    self.adv_layer = nn.Sequential( nn.Linear(512, 1),
                                   nn.Sigmoid())
    self.aux_layer = nn.Sequential( nn.Linear(512, opts.nb_of_classes),
                                    nn.Softmax())

  def forward(self, img):
    out = self.linear_blocks(img)
    validity = self.adv_layer(out)
    label = self.aux_layer(out)
    return validity, label

# Initialize generator and discriminator
pretrained_ACGAN = torch.load('pretrained_models/ACGAN.pth')
generator = pretrained_ACGAN['generator']
discriminator = pretrained_ACGAN['discriminator']

if cuda:
  generator.cuda()
  discriminator.cuda()
  adversarial_loss.cuda()
  auxiliary_loss.cuda()

# Configure data loader
full_data = torch.load('./datasets/Synthetic/data_train_test_500_classes_2000_samples.pth')
test_dataset = TensorDataset(full_data['data_test'], full_data['labels_test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)
classifier = models.Classifier_2048_features(500)
classifier = classifier.cuda()
# Optimizers

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_gen_batch(class_size):
  """Saves a grid of generated digits ranging from 0 to nb_of_classes"""
  # Sample noise
  z = Variable(FloatTensor(np.random.normal(0, 1, (class_size*opts.nb_of_classes, opts.latent_dim))))
  # Get labels ranging from 0 to nb_of_classes for n rows
  labels = np.array([class_idx for _ in range(class_size) for class_idx in range(opts.nb_of_classes)])
  labels = Variable(LongTensor(labels))
  gen_imgs = generator(z, labels)
  return gen_imgs.data, labels

# ----------
#  Training
# ----------
classification_criterion = nn.CrossEntropyLoss();
classification_criterion = classification_criterion.cuda()
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)
max_acc = 0
for epoch in range(opts.n_epochs):
  for idx in range(opts.batches_per_epoch):
    inputs, labels = sample_gen_batch(int(opts.batch_size/opts.nb_of_classes))
    orig_classes = classifier(inputs.cuda())
    classification_loss = classification_criterion(orig_classes, labels.long().cuda())
    classification_loss.backward()
    classification_optimizer.step()
    classification_optimizer.zero_grad()
    
    if idx%10==0:
      print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, opts.n_epochs,  classification_loss.item()))
    
    
  test_acc = sup_functions.test_classifier(classifier, test_loader)
  
  if test_acc > max_acc
    max_acc = test_acc
  print("Accuracy on the testset: " + str(test_acc) + "; Best accuracy: " + str(max_acc))

#  print("Testing the quality of generated samples:")
#  test_epoch = 100
#  cm = [[0 for i in range(opts.nb_of_classes)] for j in range(opts.nb_of_classes)] 
#  for idx in range(test_epoch):
#    gen_samples = sample_image(n_row=10)
#    res = classifier(gen_samples[0]).max(1)[1]
#    cm += confusion_matrix(gen_samples[1], res)
#  print("Average accuracy: " + str(cm.diagonal().mean()*10/test_epoch))
#  print('Confusion matrix')
#  print(cm*10/test_epoch)
