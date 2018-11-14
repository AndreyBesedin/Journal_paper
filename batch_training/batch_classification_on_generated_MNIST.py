#import autoencoders_MNIST
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os

root = '/home/besedin/workspace/Projects/Journal_paper/'
dataset = 'MNIST'
print('Loading data')
bs = 600
model_type = 'cgan'
loss = 'mixed'
opt = {'dataset': 'MNIST', # 'MNIST', 'LSUN_features', 'Synthetic' 
       'im_size': 32,
       'nb_epochs':25,
       'model_type': 'cgan', # possible values 'autoencoder', 'acgan', 'cgan'
       'loss': 'mixed', # possible values 'MSE', 'classif', 'mixed'
       'latent_dim': 62,
       'n_classes': 10,
       'code_dim': 2,
       'channels': 1,
       }

img_transform = transforms.Compose([
    transforms.Resize(opt['im_size']),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#img_transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=root+'datasets/',
  train=True,
  download=True,
  transform=img_transform) 
test_dataset = datasets.MNIST(root=root+'datasets/',
  train=False,
  download=True,
  transform=img_transform)
  
train_loader = DataLoader(train_dataset, shuffle=False,
  batch_size=bs)
test_loader = DataLoader(test_dataset, shuffle=False,
  batch_size=bs)

#class Generator(nn.Module):
#  def __init__(self):
#    super(Generator, self).__init__()
#    self.label_emb = nn.Embedding(10, 100)
#    self.init_size = opt['im_size'] // 4 # Initial size before upsampling
#    self.l1 = nn.Sequential(nn.Linear(100, 128*self.init_size**2))
#    self.conv_blocks = nn.Sequential(
#      nn.BatchNorm2d(128),
#      nn.Upsample(scale_factor=2),
#      nn.Conv2d(128, 128, 3, stride=1, padding=1),
#      nn.BatchNorm2d(128, 0.8),
#      nn.LeakyReLU(0.2, inplace=True),
#      nn.Upsample(scale_factor=2),
#      nn.Conv2d(128, 64, 3, stride=1, padding=1),
#      nn.BatchNorm2d(64, 0.8),
#      nn.LeakyReLU(0.2, inplace=True),
#      nn.Conv2d(64, 1, 3, stride=1, padding=1),
#      nn.Tanh()
#    )
#  def forward(self, noise, labels):
#    gen_input = torch.mul(self.label_emb(labels), noise)
#    out = self.l1(gen_input)
#    out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#    img = self.conv_blocks(out)
#    return img
img_shape = (1, opt['im_size'], opt['im_size'])
#class Generator(nn.Module):
  ## Cgan generator 
  #def __init__(self):
    #super(Generator, self).__init__()
    #self.label_emb = nn.Embedding(10, 10)
    
    #def block(in_feat, out_feat, normalize=True):
      #layers = [  nn.Linear(in_feat, out_feat)]
      #if normalize:
        #layers.append(nn.BatchNorm1d(out_feat, 0.8))
      #layers.append(nn.LeakyReLU(0.2, inplace=True))
      #return layers
    
    #self.model = nn.Sequential(
      #*block(100+10, 128, normalize=False),
      #*block(128, 256),
      #*block(256, 512),
      #*block(512, 1024),
      #nn.Linear(1024, 28*28),
      #nn.Tanh()
    #)

  #def forward(self, noise, labels):
    ## Concatenate label embedding and image to produce input
    #gen_input = torch.cat((self.label_emb(labels), noise), -1)
    #img = self.model(gen_input)
    #img = img.view(img.size(0), *img_shape)
    #return img
    
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
    self.bn2 = nn.BatchNorm2d(64)
    self.fc1  = nn.Linear(64*opt['im_size']*opt['im_size'], 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, 1, opt['im_size'], opt['im_size'])
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = x.view(batch_size, 64*opt['im_size']*opt['im_size'])
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

def test_model(model, test_loader):
  total = 0
  correct = 0
  for idx, (test_x, test_y) in enumerate(test_loader):
    input_test = test_x.cuda()
    outputs = model(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

def sample_image(generator, n_row):
  """Saves a grid of generated digits ranging from 0 to n_classes"""
  # Sample noise
  z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n_row**2, 100))))
  # Get labels ranging from 0 to n_classes for n rows
  labels = np.array([num for _ in range(n_row) for num in range(n_row)])
  labels = Variable(torch.LongTensor(labels))
  gen_imgs = generator(z.cuda(), labels.cuda())
  return gen_imgs, labels

ls_ = [0]
betta_ = [0]
if opt['model_type'] == 'autoencoder':
  ls_ = [2, 4, 8, 16, 32]
  if opt['loss'] == 'mixed':
    betta_ = [0.5, 0.01, 0.05, 0.1, 0.2, 1, 10]

for betta in betta_:
  for ls in ls_:
    model_name = ''
    if opt['model_type']=='autoencoder':
      model_name = 'autoencoder_multi-class_'+str(ls)+'_'+opt['loss']
      if opt['loss']=='mixed':
        model_name = model_name + '_betta_'+ str(betta)
    elif opt['model_type']=='cgan':
      model_name = 'cgan_generator'
    print('Training for model ' + model_name)
    class autoencoder2(nn.Module):
      def __init__(self):
        global ls
        super(autoencoder2, self).__init__()
        self.encoder = nn.Sequential(
          nn.Conv2d(1, 32, 5, 1, 2),
          nn.BatchNorm2d(32),
          nn.ReLU(True),
          nn.MaxPool2d(2, stride=2), 
          nn.Conv2d(32, 64, 5, 1, 2),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
          nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
          nn.Conv2d(64, ls, 5, 1, 2),
          nn.BatchNorm2d(ls),
          nn.ReLU(True),
          nn.MaxPool2d(7, stride=None)  # b, 8, 2, 2
          )
        self.decoder = nn.Sequential(
          nn.ConvTranspose2d(ls, 32, 7, stride=1, padding=0),  # b, 16, 5, 5
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
        x = x.view(batch_size, 1, 28, 28)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
    #generator_state = torch.load(root+'gen_model_training/MNIST/results/models/' + model_name + '.pth')
    #generator = autoencoder2()
    #generator.load_state_dict(generator_state)
    #generator=generator.cuda()
    #print(generator)
    #elif model_type == 'acgan':
    generator_state = torch.load('/home/besedin/workspace/Projects/Journal_paper/external_codes/cgan_generator.pth')
    ##generator = autoencoder2()
    generator = Generator()
    generator.load_state_dict(generator_state)
    generator = generator.cuda()
    print(generator)
    model = Net().cuda()
    test_acc = test_model(model, test_loader)
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    max_test_acc = 0
    accuracies = torch.zeros(opt['nb_epochs'])
    for epoch in range(opt['nb_epochs']):  # loop over the dataset multiple times
      running_loss = 0.0
#    for idx, data in enumerate(trainloader, 0):
#      for idx, (train_x, train_y) in enumerate(train_loader):
      for idx in range(250):
        
        #autoencoders
#        inputs_orig = train_x.cuda()
#        inputs = generator(inputs_orig.data)
#        labels = train_y.cuda()
      
        #acgan
        gen_images = sample_image(generator, 10)
        inputs = gen_images[0].cuda()
        labels = gen_images[1].cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if idx % 100 == 99:    # print every 2000 mini-batches
          print('[%d, %5d] loss: %.3f' %
                (epoch + 1, idx + 1, running_loss / 100))
          running_loss = 0.0
        
      test_acc = test_model(model, test_loader)
      if test_acc > max_test_acc:
        max_test_acc = test_acc
        best_model = model.float()
        
      print('Test accuracy: ' + str(test_acc))
      accuracies[epoch] = test_acc
    torch.save(accuracies, './results/MNIST/accuracies/'+model_name+'.pth')
    print('Finished Training')
