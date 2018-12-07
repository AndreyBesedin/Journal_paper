import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import matplotlib.pyplot as plt 
from torch.utils.data import sampler
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from progress.bar import Bar
from torchvision import datasets, transforms
from torch.autograd import Variable
#from synthetic_data_generation import initialize_synthetic_sampler, sample_data_from_sampler
from sup_functions import test_model, test_model_on_gen, weights_init
from models import Net, autoencoder
import new_models
#from models import autoencoder2 as autoencoder

import os

root = '~/workspace/Projects/Journal_paper/'
dataset = 'LSUN'
print('Loading data')
opts = {
  'batch_size': 1000,
  'mode': 'multi-class',
  'dataset': 'LSUN',
  'test_every': 1,
  'learning_rate': 0.001,
  'number_of_epochs': 100,
  'dim': 2048,
  'nb_classes': 30,
  'code_size': 32,
  'betta': 0.2,
  'add_noise': True,
  'cuda_device': 0,
  }
  
torch.cuda.set_device(opts['cuda_device'])
code_size = opts['code_size']
nb_classes = opts['nb_classes']
trainset, testset = {}, {}

trainset_ = torch.load('../datasets/LSUN/trainset.pth')
testset_  = torch.load('../datasets/LSUN/testset.pth')
trainset = data_utils.TensorDataset(trainset_[0], trainset_[1])
testset = data_utils.TensorDataset(testset_[0], testset_[1])

train_loader = data_utils.DataLoader(trainset, batch_size=opts['batch_size'], shuffle = True)
test_loader = data_utils.DataLoader(testset, batch_size=opts['batch_size'], shuffle = False)

autoencoder_model = autoencoder(code_size).cuda()
#autoencoder_model.apply(weights_init)
#classifier_model = torch.load(root+'batch_training/results/LSUN/models/LSUN_classifier_original.pth')

classifier_model_dict = torch.load('../pretrained_models/batch_classifier_LSUN_30_classes.pth')
classifier_model = new_models.Classifier_2048_features(30)
classifier_model.load_state_dict(classifier_model_dict)
classifier_model = classifier_model.cuda()
criterion_AE = nn.MSELoss().cuda()
criterion_classif = nn.MSELoss().cuda()
#optimizer = torch.optim.SGD(autoencoder_model.parameters(), lr=opts['learning_rate'], momentum=0.99)
optimizer_main = torch.optim.Adam(autoencoder_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999),
                             weight_decay=1e-5)

accuracies = []
best_acc = 0
acc = test_model(classifier_model, test_loader)
print('Accuracy of pretrained model on the original testset: ' + str(acc))
for epoch in range(opts['number_of_epochs']):
  bar = Bar('Training: ', max=int(opts['nb_classes']*100000/opts['batch_size']))
  for idx, (train_X, train_Y) in enumerate(train_loader):
    bar.next()
    inputs = train_X.cuda()
    labels = train_Y.cuda()
    optimizer_main.zero_grad()
    #optimizer_class.zero_grad()
    #
#    img = Variable(inputs).cuda()
    # ===================forward=====================
    outputs = autoencoder_model(inputs)
    
    orig_classes = classifier_model(inputs)
    classification_reconstructed = classifier_model(outputs)
    
    loss_classif = criterion_classif(classification_reconstructed, orig_classes)
    loss_AE = criterion_AE(outputs, inputs)
    #
    #loss_classif.backward(retain_graph=True)
    #
    loss = opts['betta']*loss_classif + loss_AE
    # ===================backward====================
    loss.backward()
    #optimizer_class.step()
    optimizer_main.step()
    
    if idx%100==0:
      #plt.plot(range(2048), inputs[0].cpu().detach().numpy(), label='in')
      #plt.plot(range(2048), outputs[0].cpu().detach().numpy(), label='out')
      #plt.legend()
      #plt.savefig('imgs/epoch_'+str(epoch)+'_idx_'+str(idx)+'.png')
      #plt.close()
      print('epoch [{}/{}], total loss:{:.4f}, classification loss: {:.4f}, AE loss: {:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.item(), loss_classif.item(), loss_AE.item()))
    # ===================log========================
  bar.finish()
  print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, opts['number_of_epochs'], loss.item()))
  if epoch % opts['test_every'] == 0:
    autoencoder_model.eval()
    acc = test_model_on_gen(classifier_model, autoencoder_model, test_loader)
    accuracies.append(acc)
    torch.save(accuracies, 'results/representivity_LSUN_' + str(opts['code_size']) + '_code_size_' + str(opts['nb_classes']) +'_classes.pth')
    if acc>best_acc:
      best_acc=acc
      torch.save(autoencoder_model.state_dict(), 'models/AE_LSUN_' +str(opts['code_size']) + '_code_size_' + str(opts['nb_classes']) +'_classes.pth')
    autoencoder_model.train()
    print('Accuracy on reconstructed testset: ' + str(acc))

#torch.save(model.state_dict(), './conv_autoencoder_LSUN.pth')
