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
from sup_functions import test_model, test_model_on_gen, weights_init
from models import Net, autoencoder
#from models import autoencoder2 as autoencoder

import os

root = '~/workspace/Projects/Journal_paper/'
dataset = 'synthetic'
print('Loading data')
opts = {
  'batch_size': 1000,
  'mode': 'multi-class',
  'dataset': 'synthetic',
  'test_every': 1,
  'learning_rate': 0.001,
  'number_of_epochs': 100,
  'dim': 2048,
  'nb_classes': 500,
  'code_size': 32,
  'betta': 0.1,
  'samples_per_class_train': 2000,
  'samples_per_class_test': 2000,
  'predefined_sampler': False,
  'load_data': True,
  'add_noise': True,
  'cuda_device': 0,
  }
  
torch.cuda.set_device(opts['cuda_device'])
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

autoencoder_model = autoencoder(code_size).cuda()
#autoencoder_model.apply(weights_init)
classifier_model = torch.load('./models/batch_classifier_'+ str(opts['nb_classes']) +'_classes_'+str(opts['samples_per_class_train'])+'_samples.pth')

criterion_AE = nn.MSELoss().cuda()
criterion_classif = nn.MSELoss().cuda()
optimizer_main = torch.optim.Adam(autoencoder_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999),
                             weight_decay=1e-5)

accuracies = []
best_acc = 0
acc = test_model(classifier_model, test_loader)
print('Accuracy of pretrained model on the original testset: ' + str(acc))
for epoch in range(opts['number_of_epochs']):
  bar = Bar('Training: ', max=int(opts['nb_classes']*opts['samples_per_class_train']/opts['batch_size']))
  for idx, (train_X, train_Y) in enumerate(train_loader):
    bar.next()
    inputs = train_X.cuda()
    labels = train_Y.cuda()
    optimizer_main.zero_grad()
    # ===================forward=====================
    outputs = autoencoder_model(inputs)
    
    orig_classes = classifier_model(inputs)
    classification_reconstructed = classifier_model(outputs)
    
    loss_classif = criterion_classif(classification_reconstructed, orig_classes)
    loss_AE = criterion_AE(outputs, inputs)
    loss = opts['betta']*loss_classif + loss_AE
    # ===================backward====================
    loss.backward()
    optimizer_main.step()
    
    if idx%100==0:
      #plt.plot(range(2048), inputs[0].cpu().detach().numpy(), label='in')
      #plt.plot(range(2048), outputs[0].cpu().detach().numpy(), label='out')
      #plt.legend()
      #plt.savefig('imgs/epoch_'+str(epoch)+'_idx_'+str(idx)+'.png')
      #plt.close()
      print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.item()))
    # ===================log========================
  bar.finish()
  print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.item()))
  if epoch % opts['test_every'] == 0:
    autoencoder_model.eval()
    acc = test_model_on_gen(classifier_model, autoencoder_model, test_loader)
    accuracies.append(acc)
    torch.save(accuracies, 'results/representivity_AE_' + str(opts['code_size']) + '_code_size_' + str(opts['nb_classes']) +'_classes_'+str(opts['samples_per_class_train'])+'_samples.pth')
    if acc>best_acc:
      best_acc=acc
      torch.save(autoencoder_model.state_dict(), 'models/AE_' +str(opts['code_size']) + '_code_size_' + str(opts['nb_classes']) +'_classes_'+str(opts['samples_per_class_train'])+'_samples.pth')
    autoencoder_model.train()
    print('Accuracy on reconstructed testset: ' + str(acc))

#torch.save(model.state_dict(), './conv_autoencoder_LSUN.pth')
