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
parser.add_argument('--niter', type=int, default=100, help='number of training epochs')
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
opts = parser.parse_args()
opts.fake_storage_size = opts.stream_batch_size
print('Loading data')
if opts.cuda:
  torch.cuda.set_device(opts.cuda_device)
if opts.dataset=='MNIST':
  opts.nb_of_classes=opts.MNIST_classes
elif opts.dataset=='LSUN':
  opts.nb_of_classes=opts.LSUN_classes
  
AE_specific = ''
if opts.generator_type == 'AE':
  AE_specific = '_' + str(opts.code_size) + '_trade-off_' + str(opts.betta1) + '_'
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
orig_trainset, orig_testset = sup_functions.load_dataset(opts)

gen_model = sup_functions.init_generative_model(opts)
generative_optimizer_reconstruction = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta2, betas=(0.9, 0.999), weight_decay=1e-5)
generative_optimizer_classification = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta1, betas=(0.9, 0.999), weight_decay=1e-5)
generative_criterion_reconstruction = nn.MSELoss()
generative_criterion_classification = nn.MSELoss()

classifier = sup_functions.init_classifier(opts)
classification_optimizer = optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)
classification_criterion = nn.CrossEntropyLoss()

if opts.cuda:
  gen_model = gen_model.cuda()
  generative_criterion_reconstruction = generative_criterion_reconstruction.cuda()
  generative_criterion_classification = generative_criterion_classification.cuda()
  classifier = classifier.cuda()
  classification_criterion = classification_criterion.cuda()
  
print('Reconstructing data')
#trainset = sup_functions.reconstruct_dataset_with_AE(trainset, gen_model, bs = 1000, real_data_ratio=0)  
test_loader = data_utils.DataLoader(orig_testset, batch_size=opts.batch_size, shuffle = False)  

max_test_acc = 0
accuracies = []
stream_classes = []
Stream = True
epoch = 0
seen_classes = {}
real_data_storage = torch.FloatTensor(opts.nb_of_classes, opts.real_storage_size, opts.feature_size)
fake_data_storage = torch.FloatTensor(opts.nb_of_classes, opts.fake_storage_size, opts.feature_size)
while Stream:
  """ 
  Initialize the data buffer that will be used for training on current interval
  The buffer size is k*(size of the stream batch), where k is equal to the number of already seen classes
  until it reaches some predefined constant K
  """
  epoch+=1
  data_class = random.randint(0, opts.nb_of_classes-1) # randomly pick the streaming class
  seen_classes_prev = seen_classes.copy()
  seen_classes[str(data_class)] = True  # Add new class to seen classes or do nothing if already there
  print('Epoch nb: ' + str(epoch) + '; already seen classes:')
  print(seen_classes)
  # Initialize the training dataset for the interval
  fake_size = min(len(seen_classes), opts.fake_to_real_ratio) 
  
  train_data = torch.FloatTensor(fake_size*opts.stream_batch_size, opts.feature_size)
  train_labels = torch.FloatTensor(fake_size*opts.stream_batch_size)
  
  #train_data, train_labels = fill_from_storages(train_data, train_labels, fake_data_storage, real_data_storage, seen_classes_prev)

  #-----------------------------------------------------------------------------------------------
  # Inititalize the loader for corresponding class
  indices_real = (orig_trainset.tensors[1].long()==data_class).nonzero().long()
  real_data_loader = DataLoader(orig_trainset, batch_size=opts.stream_batch_size, sampler = SubsetRandomSampler(indices_real.squeeze()))
  class_duration = random.randint(1, opts.max_class_duration) # Number of stream batches we are going to receive
  
  received_batches = 0
  while received_batches < class_duration:
    for idx_stream, (real_batch, real_labels) in enumerate(real_data_loader):
      stream_classes.append(data_class)
      received_batches+=1
      fake_data_storage[data_class] = real_batch
      real_data_storage[data_class] = real_batch[-opts.real_storage_size:] #Storing a small portion of real data
      # Preparing the buffer for the interval
      for idx_key, key in enumerate(seen_classes.keys()):
        train_data[idx_key*opts.stream_batch_size:(idx_key + 1)*opts.stream_batch_size] = fake_data_storage[int(key)]
        train_data[(idx_key+1)*opts.stream_batch_size-opts.real_storage_size:(idx_key+1)*opts.stream_batch_size] = real_data_storage[int(key)]
        train_labels[idx_key*opts.stream_batch_size:(idx_key+1)*opts.stream_batch_size] = int(key)
        
      stream_trainset = TensorDataset(train_data, train_labels)
      train_loader = data_utils.DataLoader(stream_trainset, batch_size=opts.batch_size, shuffle = True)
      # Time to train models
      #sup_functions.train_classifier(classifier, train_loader, classification_optimizer, classification_criterion)
      
      # Generative model training
      #sup_functions.train_gen_model(gen_model, classifier, train_loader, generative_criterion_classification, generative_optimizer_classification,
 #generative_criterion_reconstruction, generative_optimizer_reconstruction,opts) 
      for idx, (train_X, train_Y) in enumerate(train_loader):
        inputs = train_X.float()
        labels = train_Y
        if opts.cuda:
          inputs = inputs.cuda()
          labels = labels.cuda()
    # ===================forward=====================
        orig_classes = classifier(inputs)
        #orig_classes.require_grad=True
        #classification_loss = classification_criterion(orig_classes, labels.long())
        #if opts.betta1!=0:
        outputs = gen_model(inputs)
        orig_classes.require_grad=False
        classification_reconstructed = classifier(outputs)
        generative_loss_class = generative_criterion_classification(classification_reconstructed, orig_classes)
        #print('Paco 5')
        generative_loss_class.backward()
        #print('Paco 6')
        #classification_optimizer.step()
        generative_optimizer_classification.step()
        #classification_optimizer.zero_grad()
        generative_optimizer_classification.zero_grad()
        #print('Paco 7')
        if idx%100==0:
          print('epoch [{}/{}], generative classification loss: {:.4f}'
            .format(epoch, opts.niter,  generative_loss_class.item()))
      for idx_classifier in range(10):
        for idx, (train_X, train_Y) in enumerate(train_loader):
          orig_classes = classifier(inputs)
          orig_classes.require_grad=True
          classification_loss = classification_criterion(orig_classes, labels.long())
          classification_loss.backward()
          classification_optimizer.step()
          classification_optimizer.zero_grad()
          if idx%100==0:
            print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, opts.niter,  classification_loss.item()))
      if received_batches >= class_duration: break
  # Reconstructing saved data with updated generator  
  for key in seen_classes.keys():
    fake_data_storage[int(key)] = gen_model(fake_data_storage[int(key)].data.cuda()).cpu().data
  
    
  # Testing phase in the end of the interval
  acc = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
  accuracies.append(acc)
  print('Test accuracy: ' + str(accuracies[-1]))
  
#for t in range(opts.niter):  # loop over the dataset multiple times
  #print('Training epoch ' + str(epoch))
  #sup_functions.train_classifier(classifier, train_loader, optimizer, criterion)
  #accuracies.append(sup_functions.test_classifier(classifier, test_loader))
  #if accuracies[-1] > max_test_acc:
    #max_test_acc = accuracies[-1]
    #best_classifier = classifier
    #torch.save(best_classifier, opts.root+'pretrained_models/generalizability_classifier_' + name_to_save + '.pth')      
  #print('Test accuracy: ' + str(accuracies[-1]))

  #torch.save(accuracies, opts.root+'results/generalizability_accuracy_' + name_to_save + '.pth' )
#print('Finished Training')
