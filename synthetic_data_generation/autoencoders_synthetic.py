import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
from torch.utils.data import sampler
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from progress.bar import Bar
from torchvision import datasets, transforms
from torch.autograd import Variable
from synthetic_data_generation import initialize_synthetic_sampler, sample_data_from_sampler
import os

root = '~/workspace/Projects/Journal_paper/'
dataset = 'synthetic'
print('Loading data')
opts = {
  'batch_size': 100,
  'mode': 'multi-class',
  'dataset': 'synthetic',
  'latent_space_size': 32,
  'test_every': 1,
  'learning_rate': 0.001,
  'number_of_epochs': 10000,
  'dim': 2048,
  'nb_classes': 10,
  'code_size': 64,
  'betta': 0.2,
  'samples_per_class_train': 10000,
  'samples_per_class_test': 10000,
  'predefined_sampler': False,
  'load_data': True,
  'add_noise': True,
  }
  
cuda_device = 0
torch.cuda.set_device(cuda_device)
code_size = opts['code_size']
nb_classes = opts['nb_classes']
trainset, testset, data_sampler = {}, {}, {}

if opts['load_data']:
  full_data = torch.load('./data/data_train_test_'+str(opts['nb_classes'])+'_classes_'+str(opts['samples_per_class_train'])+'_samples.pth')
  trainset = data_utils.TensorDataset(full_data['data_train'], full_data['labels_train'])
  testset = data_utils.TensorDataset(full_data['data_test'], full_data['labels_test'])
  #trainset = torch.load('./data/trainset_'+ str(opts['nb_classes']) +'_classes.pth')
  #testset = torch.load('./data/testset_'+ str(opts['nb_classes']) +'_classes.pth')
else:
  if opts['predefined_sampler']:
    data_sampler = torch.load('./models/data_sampler_'+ str(opts['nb_classes']) +'_classes.pth')
  else:
    data_sampler = initialize_synthetic_sampler(opts['dim'], opts['nb_classes'], 1.7)
  train_class_size, test_class_size = opts['samples_per_class_train'], opts['samples_per_class_test']
  trainset_ = sample_data_from_sampler(data_sampler, train_class_size)
  testset_ = sample_data_from_sampler(data_sampler, test_class_size)
  trainset = data_utils.TensorDataset(trainset_[0], trainset_[1])
  testset = data_utils.TensorDataset(testset_[0], testset_[1])  

train_loader = data_utils.DataLoader(trainset, batch_size=opts['batch_size'], shuffle = True)
test_loader = data_utils.DataLoader(testset, batch_size=opts['batch_size'], shuffle = False)

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    torch.nn.init.kaiming_normal_(m.weight.data)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)
  elif classname.find('Linear')!= -1:
    torch.nn.init.kaiming_normal_(m.weight.data)
    m.bias.data.fill_(0)

#class autoencoder(nn.Module):
  #def __init__(self):
    #def linear_block(in_, out_):
      #return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
##      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    #global code_size
    #super(autoencoder, self).__init__()
    #self.encoder = nn.Sequential(
      #linear_block(2048, 1024),
##      linear_block(3072, 1024),
      #linear_block(1024, 512),
      #linear_block(512, 128),
      #linear_block(128, 64),
      #linear_block(64, 64),
      #nn.Linear(64, code_size),
    #)
    #self.decoder = nn.Sequential(
      #linear_block(code_size, 64),
      #linear_block(64, 64),
      #linear_block(64, 128),
      #linear_block(128, 512),
      #linear_block(512, 1024),
      #nn.Linear(1024, 2048),
##      nn.Tanh()
    #)

  #def forward(self, x):
    #x = self.encoder(x)
    #x = self.decoder(x)
    #return x

class autoencoder(nn.Module):
  def __init__(self):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    global code_size
    super(autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(2048, 512),
    )
    self.decoder = nn.Sequential(
      nn.Linear(512, 2048),
#      nn.Tanh()
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
    
class Net(nn.Module):
  global nb_classes
  def __init__(self):
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


autoencoder_model = autoencoder().cuda()
#autoencoder_model.apply(weights_init)
classifier = torch.load('./models/batch_classifier_'+ str(opts['nb_classes']) +'_classes_'+str(opts['samples_per_class_train'])+'_samples.pth')

def test_model_on_gen(classif, autoenc, loader):
  total = 0
  correct = 0
  for idx, (test_X,  test_Y) in enumerate(loader):
    input_test = autoenc(test_X.cuda())
    outputs = classif(input_test)
#    outputs = classif(test_X.cuda())
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

def test_model(classif, loader):
  total = 0
  correct = 0
  for idx, (test_X, test_Y) in enumerate(loader):
    input_test = test_X.cuda()
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

criterion = nn.MSELoss().cuda()
criterion_classif = nn.MSELoss()
#optimizer = torch.optim.SGD(autoencoder_model.parameters(), lr=opts['learning_rate'], momentum=0.99)
optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999),
                             weight_decay=1e-5)
#optimizer_class = torch.optim.Adam(autoencoder_model.parameters(), lr=opts['betta']*opts['learning_rate'],
#                             weight_decay=1e-7)

acc = test_model(classifier, test_loader)
print('Accuracy of pretrained model on the original testset: ' + str(acc))
for epoch in range(opts['number_of_epochs']):
  bar = Bar('Training: ', max=int(opts['nb_classes']*opts['samples_per_class_train']/opts['batch_size']))
  for idx, (train_X, train_Y) in enumerate(train_loader):
    bar.next()
    inputs = train_X
    labels = train_Y.cuda()
    img = Variable(inputs).cuda()
    #orig_classes = classifier(img)
#    img = Variable(inputs).cuda()
    # ===================forward=====================
    output = autoencoder_model(img)
    #classification_reconstructed = classifier(output)
    #loss_classif = criterion_classif(classification_reconstructed, orig_classes)
    #optimizer_class.zero_grad()
    #loss_classif.backward(retain_graph=True)
    #optimizer_class.step()
    loss = criterion(output, img)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if idx%100==0:
      plt.plot(range(2048), img[0].cpu().detach().numpy(), label='in')
      plt.plot(range(2048), output[0].cpu().detach().numpy(), label='out')
      plt.legend()
      plt.savefig('imgs/epoch_'+str(epoch)+'_idx_'+str(idx)+'.png')
      plt.close()
      print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.item()))
    # ===================log========================
  bar.finish()
  print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.item()))
  if epoch % opts['test_every'] == 0:
    autoencoder_model.eval()
    acc = test_model_on_gen(classifier, autoencoder_model, train_loader)
    autoencoder_model.train()
    print('Accuracy on reconstructed testset: ' + str(acc))

#torch.save(model.state_dict(), './conv_autoencoder_LSUN.pth')
