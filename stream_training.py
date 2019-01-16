import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import synthetic_data_generation
import sup_functions
import models

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='MNIST | LSUN | Synthetic')
parser.add_argument('--experiment_type', default='generalizability', help='batch_classification | representativity | generalizability | incremental_stream | unordered_stream')
parser.add_argument('--generator_type', default='AE', help='AE | CGAN | ACGAN')
parser.add_argument('--code_size', default='32', help='Size of the code representation in autoencoder')
parser.add_argument('--root', default='/home/besedin/workspace/Projects/Journal_paper/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--image_size', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of training intervals')
parser.add_argument('--niter_generation', type=int, default=10, help='number of training epochs')
parser.add_argument('--niter_classification', type=int, default=10, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--betta1', type=float, default=0.2, help='trade-off coefficients for ae training, classification loss')
parser.add_argument('--betta2', type=float, default=1, help='trade-off coefficients for ae training, reconstruction loss')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default = 0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device to use')
parser.add_argument('--load_classifier', action='store_true', help='if enabled, load pretrained classifier with corresponding characteristics')
parser.add_argument('--load_gen_model', action='store_true', help='if enabled, load pretrained generative model')

parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--MNIST_classes', type=int, default=10, help='nb of classes from MNIST by default')
parser.add_argument('--LSUN_classes', type=int, default=30, help='nb of classes from MNIST by default')
parser.add_argument('--optimizer_classification', default='Adam', help='Adam | SGD')
parser.add_argument('--optimizer_generator', default='Adam', help='Adam | SGD')
#Synthetic data options
parser.add_argument('--nb_of_classes', default=10, type=int, help='number of classes in synthetic dataset')
parser.add_argument('--class_size', default=100, type=int, help='number of elements in each class')
parser.add_argument('--feature_size', default=2048, type=int, help='feature size in synthetic dataset')
parser.add_argument('--generate_data', action='store_true', help='generates a new dataset if enabled, loads predefined otherwise')
parser.add_argument('--real_storage_size', default=300, type=int, help='size of the storage of real data per class')
parser.add_argument('--fake_storage_size', default=2000, type=int, help='size of the storage of fake data per class')
parser.add_argument('--stream_batch_size', default=2000, type=int, help='size of the batch we receive from stream')
parser.add_argument('--max_class_duration', default=10, type=int, help='size of the batch we receive from stream')
parser.add_argument('--fake_to_real_ratio', default=40, type=int, help='size of the batch we receive from stream')
parser.add_argument('--max_stream_classes', default=3, type=int, help='Max number of simultaneous classes in the stream')
parser.add_argument('--max_interval_duration', default=10, type=int, help='Max duration of each interval environment')
parser.add_argument('--max_stream_intervals', default=500, type=int, help='Max duration of each interval environment')
opts = parser.parse_args()
opts.fake_storage_size = opts.stream_batch_size

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
      
print('Loading data')
if opts.cuda:
  torch.cuda.set_device(opts.cuda_device)
  
  
if opts.dataset=='MNIST_features':
  opts.nb_of_classes=opts.MNIST_classes
  opts.max_stream_classes = 2
  opts.max_stream_intervals = 500
elif opts.dataset=='LSUN':
  opts.nb_of_classes=opts.LSUN_classes
  opts.max_stream_classes = 6
  opts.max_stream_intervals = 1000
elif opts.dataset=='Synthetic':
  opts.max_stream_classes = 100
  opts.max_stream_intervals = 1000

AE_specific = ''
if opts.generator_type == 'AE':
  AE_specific = '_' + str(opts.code_size) + '_cl_loss_' + str(opts.betta1) + '_rec_loss_' +str(opts.betta2) +'_'
name_to_save = opts.dataset + '_' + opts.generator_type + AE_specific + str(opts.nb_of_classes) + '_classes.pth'
  
print(opts)
if opts.manual_seed is None:
  opts.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opts.manual_seed)
random.seed(opts.manual_seed)
torch.manual_seed(opts.manual_seed)

if torch.cuda.is_available() and not opts.cuda:
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
print('Loading data')

pretrain_on_classes = range(5)
if opts.dataset == 'LSUN':
  pretrain_on_classes = range(15)
elif opts.dataset == 'Synthetic':
  pretrain_on_classes = range(250)
  
trainset, testset = sup_functions.load_dataset(opts)
indices_train = sup_functions.get_indices_for_classes(trainset, pretrain_on_classes)
indices_test = sup_functions.get_indices_for_classes(testset, pretrain_on_classes)
train_loader_classif = data_utils.DataLoader(trainset, batch_size=opts.batch_size, sampler = SubsetRandomSampler(indices_train))
train_loader_gen = data_utils.DataLoader(trainset, batch_size=opts.batch_size, sampler = SubsetRandomSampler(indices_train))
test_loader = data_utils.DataLoader(testset, batch_size=opts.batch_size, shuffle = False, sampler = SubsetRandomSampler(indices_test))

gen_model = sup_functions.init_generative_model(opts)
#generative_optimizer_reconstruction = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta2, betas=(0.9, 0.999), weight_decay=1e-5)
generative_optimizer_classification = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta1, betas=(0.9, 0.999), weight_decay=1e-5)
#generative_criterion_reconstruction = nn.MSELoss()
generative_criterion_classification = nn.MSELoss()

classifier = sup_functions.init_classifier(opts)
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)
classification_criterion = nn.CrossEntropyLoss()

if opts.cuda:
  gen_model = gen_model.cuda()
#  generative_criterion_reconstruction = generative_criterion_reconstruction.cuda()
  generative_criterion_classification = generative_criterion_classification.cuda()
  classifier = classifier.cuda()
  classification_criterion = classification_criterion.cuda()

#print('Classification accuracy on the original testset: ' + str(sup_functions.test_classifier(classifier, test_loader)))

#-------------------------------------------------------------------------------------------------------------------------------------
# Training the original classifier on a half of all the classes on original data 
#-------------------------------------------------------------------------------------------------------------------------------------
pretrain = False
if pretrain:
  accuracies = {}
  best_models = {}
  accuracies['original_classification'] = []
  print('Training the classifier')

  max_test_acc = 0
  for epoch in range(opts.niter_classification):  # loop over the dataset multiple times
    print('Training epoch ' + str(epoch))
    r=torch.randperm(indices_train.shape[0])
    indices_train_new=indices_train[r[:, None]].squeeze()
#  bar = Bar('Training: ', max=int(opts['nb_classes']*opts['samples_per_class_train']/opts['batch_size']))
    train_loader_classif = data_utils.DataLoader(trainset, batch_size=opts.batch_size, sampler = SubsetRandomSampler(indices_train_new))
    classification_optimizer.zero_grad()
    generative_optimizer_classification.zero_grad()
    for idx, (train_X, train_Y) in enumerate(train_loader_classif):
      inputs = train_X.float().squeeze().cuda()
      labels = train_Y.cuda()
      orig_classes = classifier(inputs)
      orig_classes.require_grad = True
      classification_loss = classification_criterion(orig_classes, labels.long())
      classification_loss.backward()
      classification_optimizer.step()
      classification_optimizer.zero_grad()
      if idx%100==0:
        print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, opts.niter_classification,  classification_loss.item()))
    acc = sup_functions.test_classifier(classifier, test_loader)
    print('Test accuracy, original classification training: ' + str(acc))
    accuracies['original_classification'].append(acc)
    torch.save(accuracies, opts.root+'results/pre_stream_training_accuracy_' + name_to_save)
    if acc > max_test_acc:
      max_test_acc = acc
      best_models['original_classifier'] = classifier
      torch.save(best_models, opts.root + 'pretrained_models/pre_stream_training_' + name_to_save)
  

#-------------------------------------------------------------------------------------------------------------------------------------
# Training an autencoder using pretrained classifier for the loss computation
#-------------------------------------------------------------------------------------------------------------------------------------
  print('Training generative model')
  max_test_acc = 0
  accuracies['generative_model_progress'] = []
#classifier = best_models['original_classifier']
  for epoch in range(opts.niter_generation):  # loop over the dataset multiple times
    print('Training epoch ' + str(epoch))
    r=torch.randperm(indices_train.shape[0])
    indices_train_new=indices_train[r[:, None]].squeeze()
    train_loader_gen = data_utils.DataLoader(trainset, batch_size=opts.batch_size, sampler = SubsetRandomSampler(indices_train_new))
    generative_optimizer_classification.zero_grad()
    classification_optimizer.zero_grad()
    for idx, (train_X, train_Y) in enumerate(train_loader_gen):
      inputs = train_X.float()
      labels = train_Y
      if opts.cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
    #if opts.cuda:
      #inputs = inputs.cuda()
      #labels = inputs.cuda()
    # ===================forward=====================
      outputs = gen_model(inputs)
      orig_classes = classifier(inputs)
      orig_classes.require_grad=False
      classification_reconstructed = classifier(outputs)
      loss_gen = generative_criterion_classification(classification_reconstructed, orig_classes)
      loss_gen.backward()
      generative_optimizer_classification.step()
      generative_optimizer_classification.zero_grad()
      if idx%100==0:
        print('epoch [{}/{}], generators loss: {:.4f}'
          .format(epoch+1, opts.niter,  loss_gen.item()))

    # ===================backward====================
    
    gen_model.eval()
    acc = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
    print('Test accuracy on reconstructed: ' + str(acc))
    gen_model.train()
    accuracies['generative_model_progress'].append(acc)
    torch.save(accuracies, opts.root+'results/pre_stream_training_accuracy_' + name_to_save)
    if acc > max_test_acc:
      max_test_acc = acc
      best_models['generative_model'] = gen_model
      torch.save(best_models, opts.root + 'pretrained_models/pre_stream_training_' + name_to_save)


#-------------------------------------------------------------------------------------------------------------------------------------
# Training a new classifier on generated data only
#-------------------------------------------------------------------------------------------------------------------------------------

  accuracies['classification_on_reconstructed'] = []
  print('Retraining the classifier')

  new_classifier = sup_functions.init_classifier(opts)
  classification_optimizer = optim.Adam(new_classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)
  max_test_acc = 0

  for epoch in range(opts.niter_classification):  # loop over the dataset multiple times
    print('Training epoch ' + str(epoch))
    r=torch.randperm(indices_train.shape[0])
    indices_train_new=indices_train[r[:, None]].squeeze()
#  bar = Bar('Training: ', max=int(opts['nb_classes']*opts['samples_per_class_train']/opts['batch_size']))
    train_loader_classif = data_utils.DataLoader(trainset, batch_size=opts.batch_size, sampler = SubsetRandomSampler(indices_train_new))
    classification_optimizer.zero_grad()
    generative_optimizer_classification.zero_grad()
    for idx, (train_X, train_Y) in enumerate(train_loader_classif):
      inputs = gen_model(train_X.float().cuda())
      labels = train_Y.cuda()
      orig_classes = new_classifier(inputs)
      orig_classes.require_grad = True
      classification_loss = classification_criterion(orig_classes, labels.long())
      classification_loss.backward()
      classification_optimizer.step()
      classification_optimizer.zero_grad()
      if idx%100==0:
        print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, opts.niter_classification,  classification_loss.item()))
    acc = sup_functions.test_classifier_on_generator(new_classifier, gen_model, test_loader)
    print('Test accuracy, classification training on reconstructed data: ' + str(acc))
    accuracies['classification_on_reconstructed'].append(acc)
    torch.save(accuracies, opts.root+'results/pre_stream_training_accuracy_' + name_to_save)
    if acc > max_test_acc:
      max_test_acc = acc
      best_models['classifier_on_reconstructed'] = new_classifier
      torch.save(best_models, opts.root + 'pretrained_models/pre_stream_training_' + name_to_save)  

  print('Finished Training')
else:
  best_models = torch.load(opts.root + 'pretrained_models/pre_stream_training_' + name_to_save)
  gen_model = best_models['generative_model']
#generative_optimizer_reconstruction = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta2, betas=(0.9, 0.999), weight_decay=1e-5)
  generative_optimizer_classification = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta1, betas=(0.9, 0.999), weight_decay=1e-5)
#generative_criterion_reconstruction = nn.MSELoss()
  generative_criterion_classification = nn.MSELoss()

  new_classifier = best_models['classifier_on_reconstructed']
  classification_optimizer = optim.Adam(new_classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)
  classification_criterion = nn.CrossEntropyLoss()
  
#-------------------------------------------------------------------------------------------------------------------------------------
# Stream training, adding new classes etc
#-------------------------------------------------------------------------------------------------------------------------------------
# 

use_gen_models = False
if not use_gen_models:
  gen_model = Identity()
  generative_optimizer_classification = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta1, betas=(0.9, 0.999), weight_decay=1e-5)
  
original_trainset, original_testset = sup_functions.load_dataset(opts)
test_loader = data_utils.DataLoader(original_testset, batch_size=opts.batch_size, shuffle = False)  


pretrained_on_classes = range(5)
if opts.dataset == 'LSUN':
  pretrained_on_classes = range(15)
elif opts.dataset == 'Synthetic':
  pretrained_on_classes = range(250)

#pretrained_models = torch.load(opts.root+ 'pretrained_models/pre_stream_training_' + name_to_save)
#gen_model = pretrained_models['generative_model'].cuda()
#classifier = pretrained_models['classifier_on_reconstructed'].cuda()
#gen_model = sup_functions.init_generative_model(opts)
#classifier = sup_functions.init_classifier(opts)
accuracies = []
acc = sup_functions.test_classifier_on_generator(new_classifier, gen_model, test_loader)
print('Accuracy on the full testset before training on stream: ' + str(acc))
accuracies.append(acc)
max_test_acc = 0
#max_test_acc = accuracies[0]
historical_storage_full = {}
historical_storage_real = {}

print('Reconstructing historical data from codes')
for idx_class in pretrained_on_classes:
  indices_class = sup_functions.get_indices_for_classes(original_trainset, [idx_class])
  class_loader = data_utils.DataLoader(original_trainset, batch_size=opts.fake_storage_size, sampler = SubsetRandomSampler(indices_class))
  for (X, Y) in class_loader:
    inputs = X.clone().float() 
    rec = gen_model(inputs.cuda())
    historical_storage_full[str(idx_class)] = rec.data
    #historical_storage_full[str(idx_class)] = X[:]
    historical_storage_full[str(idx_class)][-opts.real_storage_size:] = X[-opts.real_storage_size:]
    historical_storage_real[str(idx_class)] = X[-opts.real_storage_size:]
    break
print('Finished reconstructing historical data')


#generative_optimizer_classification = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta1*0.05, betas=(0.9, 0.999), weight_decay=1e-5)
#generative_criterion_classification = nn.MSELoss()

classification_optimizer = optim.Adam(new_classifier.parameters(), lr=opts.lr*0.1, betas=(opts.beta1, 0.999), weight_decay=1e-5)
#classification_criterion = nn.CrossEntropyLoss()

#generative_criterion_classification = generative_criterion_classification.cuda()
#classification_criterion = classification_criterion.cuda()

stream_classes = []
res_to_save = {}
Stream = True
interval = 0
best_models = {}
def regroup_data_from_storage(historical_storage, opts):
  train_data = torch.FloatTensor(len(historical_storage)*opts.fake_storage_size, opts.feature_size)
  train_labels = torch.FloatTensor(len(historical_storage)*opts.fake_storage_size)
  for idx_class, label in enumerate(historical_storage.keys()):
    train_data[idx_class*opts.fake_storage_size:(idx_class+1)*opts.fake_storage_size] = historical_storage[label]
    train_labels[idx_class*opts.fake_storage_size:(idx_class+1)*opts.fake_storage_size] = int(label)
  return train_data, train_labels
  
for stream_intervals in range(opts.max_stream_intervals):   
  """ 
  Initialize the data buffer that will be used for training on current interval
  The buffer size is k*(size of the stream batch), where k is equal to the number of already seen classes
  until it reaches some predefined constant K
  """
  interval+=1
  print('Training interval ' + str(interval))
  new_data_classes = [random.randint(0, opts.nb_of_classes-1) for _ in range(random.randint(1, opts.max_stream_classes))] # randomly pick the streaming class
  #print('interval nb: ' + str(interval) + '; already seen classes:')
  #print(historical_storage_full.keys())
  for sub_int in range(random.randint(1, opts.max_interval_duration)):
    # Initialize the training dataset for the interval
    for idx_class in historical_storage_real.keys():
      historical_storage_full[str(idx_class)][-opts.real_storage_size:] = historical_storage_real[str(idx_class)][:]
    
    # Loading real data from stream
    print("Recieving data from the stream")
    for idx_class in new_data_classes:
      indices_class = sup_functions.get_indices_for_classes(original_trainset, [idx_class])
      class_loader = data_utils.DataLoader(original_trainset, batch_size=opts.fake_storage_size, sampler = SubsetRandomSampler(indices_class))
      for (X, Y) in class_loader:
        historical_storage_full[str(idx_class)] = X[:]
        historical_storage_real[str(idx_class)] = X[-opts.real_storage_size:]
        break
    
    stream_classes.append(new_data_classes)
    res_to_save['stream_classes'] = stream_classes
    train_data, train_labels = regroup_data_from_storage(historical_storage_full, opts)
    stream_trainset = TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(stream_trainset, batch_size=opts.batch_size, shuffle = True)
    
    print('Training generative model')
    generative_optimizer_classification.zero_grad()
    classification_optimizer.zero_grad()
    
    for idx, (train_X, train_Y) in enumerate(train_loader):
      inputs = train_X.float()
      if opts.cuda:
        inputs = inputs.cuda()
      # ===================forward=====================
      outputs = gen_model(inputs)
      orig_classes = new_classifier(inputs)
      orig_classes = orig_classes.data
      classification_reconstructed = new_classifier(outputs)
      loss_gen = generative_criterion_classification(classification_reconstructed, orig_classes)

      #loss_gen = generative_criterion_classification(classifier(gen_model(inputs)), orig_classes)
      loss_gen.backward()
      generative_optimizer_classification.step()
      generative_optimizer_classification.zero_grad()
      classification_optimizer.zero_grad()
      #if idx%100==0:
      #  print('epoch [{}/{}], generators loss: {:.4f}'
      #    .format(interval, opts.niter,  loss_gen.item()))

    # ===================backward====================
    
    print('Retraining the classifier')
    #classification_optimizer = optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)

    classification_optimizer.zero_grad()
    generative_optimizer_classification.zero_grad()
    for idx, (train_X, train_Y) in enumerate(train_loader):
      inputs = gen_model(train_X.float().cuda())
      labels = train_Y.cuda()
      orig_classes = new_classifier(inputs)
      orig_classes.require_grad = True
      classification_loss = classification_criterion(orig_classes, labels.long())
      classification_loss.backward()
      classification_optimizer.step()
      classification_optimizer.zero_grad()
      #if idx%100==0:
      #  print('interval [{}/{}], classification loss: {:.4f}'.format(interval, opts.niter,  classification_loss.item()))
        
  for idx_class in historical_storage_real.keys():
      rec = gen_model(historical_storage_full[str(idx_class)].cuda())
      historical_storage_full[str(idx_class)] = rec.data.float()      
  #classifier.eval()
  #gen_model.eval()
  acc = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
  #classifier.train()
  #gen_model.train()
  print('Test accuracy, classification training on reconstructed data: ' + str(acc))
  accuracies.append(acc)
  res_to_save['accuracies'] = accuracies
  torch.save(res_to_save, opts.root+'results/stream_training_accuracy_' + name_to_save)
  if acc > max_test_acc:
    max_test_acc = acc
    best_models['classifier'] = new_classifier
    best_models['generative_model'] = gen_model
    torch.save(best_models, opts.root + 'pretrained_models/stream_training_' + name_to_save)  

  print('Finished Training')
