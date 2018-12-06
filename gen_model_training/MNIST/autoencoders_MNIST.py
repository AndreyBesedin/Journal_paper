import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os

def test_model(classif, autoencoder, test_loader):
  total = 0
  correct = 0
  std_real = 0
  std_fake = 0
  for idx, (test_x, test_y) in enumerate(test_loader):
    input_test = autoencoder(test_x.cuda())
    std_real = std_real + test_x.data.std()
    std_fake = std_fake + input_test.data.std()
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  print('Std for original data: ' + str(std_real))
  print('Std for fake data: ' + str(std_fake))
  return correct/total*100

def to_img(x):
  x = 0.5 * (x + 1)
  x = x.clamp(0, 1)
  x = x.view(x.size(0), 1, 28, 28)
  return x
#class autoencoder2(nn.Module):
  #def __init__(self):
    #global ls
    #super(autoencoder2, self).__init__()
    #self.encoder = nn.Sequential(
      #nn.Conv2d(1, 32, 5, 1, 2),
      #nn.BatchNorm2d(32),
      #nn.ReLU(True),
      #nn.MaxPool2d(2, stride=2), 
      #nn.Conv2d(32, 64, 5, 1, 2),
      #nn.BatchNorm2d(64),
      #nn.ReLU(True),
      #nn.MaxPool2d(2, stride=2),  # b, 8, 2, 2
      #nn.Conv2d(64, ls, 5, 1, 2),
      #nn.BatchNorm2d(ls),
      #nn.ReLU(True),
      #nn.MaxPool2d(7, stride=None)  # b, 8, 2, 2
    #)
    #self.decoder = nn.Sequential(
      #nn.ConvTranspose2d(ls, 32, 7, stride=1, padding=0),  # b, 16, 5, 5
      #nn.BatchNorm2d(32),
      #nn.ReLU(True),
      #nn.ConvTranspose2d(32, 64, 6, stride=2, padding=1),  # b, 8, 15, 15
      #nn.BatchNorm2d(64),
      #nn.ReLU(True),
      #nn.ConvTranspose2d(64, 16, 7, stride=1, padding=0),  # b, 1, 28, 28
      #nn.BatchNorm2d(16),
      #nn.ReLU(True),
      #nn.ConvTranspose2d(16, 1, 7, stride=1, padding=0),  # b, 1, 28, 28
      #nn.Tanh()
    #)
  #def forward(self, x):
    #batch_size = x.size(0)
    #x = x.view(batch_size, 1, 28,28)
    #x = self.encoder(x)
    #x = self.decoder(x)
    #return x

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
    self.bn2 = nn.BatchNorm2d(64)
    self.fc1  = nn.Linear(64*28*28, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, 1, 28,28)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = x.view(batch_size, 64*28*28)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

opts = {
  'root': '/home/besedin/workspace/Projects/Journal_paper/',
  'batch_size': 100,
  'mode': 'multi-class',
  'dataset': 'MNIST',
  'latent_space_size': 16,
  'learning_rate': 0.001,
  'number_of_epochs': 50,
  'betta': 10000, # trade-off between classification and MSE loss for mixed loss
  'loss': 'mixed', # possible losses: 'MSE', 'classif', 'mixed'
  'save_results': True,
  'mixed_classif_frequency': 1, # frequency of using the classif criterium in mixed approach (every N batches)
  }
ls = opts['latent_space_size']

def main(arg1):
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
      x = x.view(batch_size, 1, 28,28)
      x = self.encoder(x)
      x = self.decoder(x)
      return x

  
  global opts
  opts['betta'] = arg1
  print(opts)
  root = opts['root']
  dataset = opts['dataset']
  print('Loading data')
  
  
  img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  if not os.path.exists('./results/dc_img'):
    os.mkdir('./results/dc_img')

  train_dataset = datasets.MNIST(root=root+'datasets/',
    train=True,
    download=True,
    transform=img_transform)
  #  transform=transforms.ToTensor()) 
  test_dataset = datasets.MNIST(root=root+'datasets/',
    train=False,
    download=True,
    transform=img_transform)
  #train_dataset = datasets.EMNIST(root=root+'datasets/',
    #train=True,
    #split='balanced',
    #download=True,
    #transform=transforms.ToTensor()) 
  #test_dataset = datasets.EMNIST(root=root+'datasets/',
    #train=False,
    #split='balanced',
    #download=True,
    #transform=transforms.ToTensor())
  
  train_loader = DataLoader(train_dataset, shuffle=True,
    batch_size=opts['batch_size'])
  test_loader = DataLoader(test_dataset, shuffle=True,
    batch_size=opts['batch_size'])

  model = autoencoder2().cuda()
  print(model)
  classifier = torch.load(root + 'batch_training/models/MNIST_classifier.pt')
  classifier_const = torch.load(root + 'batch_training/models/MNIST_classifier.pt')

  criterion_classif = nn.MSELoss()
  criterion_AE = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=opts['learning_rate'],
                             weight_decay=1e-5)
  optimizer_class = torch.optim.Adam(model.parameters(), lr=opts['betta']*opts['learning_rate'],
                             weight_decay=1e-5)
  accuracies = torch.zeros(opts['number_of_epochs'])
  best_acc = 0
  test_batch = torch.FloatTensor(100, 1, 28, 28)
  for data in test_loader:
    test_batch, _ = data
    break
    
  for epoch in range(opts['number_of_epochs']):
    for idx_batch, data in enumerate(train_loader):
      img, _ = data
      print(img.mean())
      print(img.std())
      img = Variable(img).cuda()
      # Comuting the output of the classifier on original data (goal probabilities)
      # ===================forward=====================
      output = model(img)
    
      loss = 0
      if opts['loss']=='MSE' :
        loss = criterion_AE(output, img)
      elif opts['loss']=='classif':
        orig_classes = classifier(img)
        classification_reconstructed = classifier(output)
        loss = criterion_classif(classification_reconstructed, orig_classes)
      else:
        if idx_batch%opts['mixed_classif_frequency']==0:
          orig_classes = classifier(img)
          classification_reconstructed = classifier(output)
          loss_classif = criterion_classif(classification_reconstructed, orig_classes)
          optimizer.zero_grad()
          loss_classif.backward(retain_graph=True)
          optimizer_class.step()
        loss = criterion_AE(output, img)
        print(loss)
      #else:
        #orig_classes = classifier(img)
        #classification_reconstructed = classifier(output)
        #loss_AE = criterion_AE(output.data, img.data)
        #loss_classif = criterion_classif(classification_reconstructed, orig_classes)
        ##print('Loss AE: ' + str(loss_AE))
        ##print('Loss classif: ' + str(loss_classif))
        #loss = loss_AE + opts['betta']*loss_classif
      # ===================backward====================
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.data[0]))
    if epoch % 1 == 0:
      acc = test_model(classifier, model, test_loader)
      if acc > best_acc:
        best_acc=acc
        if opts['save_results'] and opts['loss']== 'mixed':
          torch.save(model.state_dict(), 'results/models/autoencoder_' +opts['mode'] + '_' + str(opts['latent_space_size']) + '_' +opts['loss']+ '_betta_'+str(opts['betta'])+'.pth')
        elif opts['save_results']:
          torch.save(model.state_dict(), 'results/models/autoencoder_' +opts['mode'] + '_' + str(opts['latent_space_size']) + '_' +opts['loss']+ '.pth')
      accuracies[epoch] = acc
      print('Accuracy on reconstructed testset: ' + str(acc))
      test_im = model(test_batch.cuda())
      pic = to_img(test_im.cpu().data)
  #    pic = to_img(output.cpu().data)
      epoch_str = str(epoch)
      if epoch<10: epoch_str = '0' + epoch_str
      save_image(pic, './results/dc_img/image_'+epoch_str+'.png')

#torch.save(model.state_dict(), './conv_autoencoder.pth')
  if opts['save_results'] and opts['loss']== 'mixed':
    torch.save(accuracies, 'results/accuracies/accuracies_' +opts['mode'] + '_' + str(opts['latent_space_size']) + '_' +opts['loss']+ '_betta_'+str(opts['betta'])+ '.pth')
  elif opts['save_results']:
    torch.save(accuracies, 'results/accuracies/accuracies_' +opts['mode'] + '_' + str(opts['latent_space_size']) + '_' +opts['loss']+'.pth')
  
if __name__=='__main__':
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
      self.bn1 = nn.BatchNorm2d(32)
      self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
      self.bn2 = nn.BatchNorm2d(64)
      self.fc1  = nn.Linear(64*28*28, 1024)
      self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
      batch_size = x.size(0)
      x = x.view(batch_size, 1, 28,28)
      x = F.relu(self.bn1(self.conv1(x)))
      x = F.relu(self.bn2(self.conv2(x)))
      x = x.view(batch_size, 64*28*28)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x
#  sys.exit(main())
  sys.exit(main(sys.argv[1]))
