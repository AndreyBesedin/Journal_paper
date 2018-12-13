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

original_trainset, original_testset = sup_functions.load_dataset(opts)
test_loader = data_utils.DataLoader(original_testset, batch_size=opts.batch_size, shuffle = False)  


pretrained_on_classes = range(5)
if opts.dataset == 'LSUN':
  pretrained_on_classes = range(15)
elif opts.dataset == 'Synthetic':
  pretrained_on_classes = range(250)

pretrained_models = torch.load(opts.root+ 'pretrained_models/pre_stream_training_' + name_to_save)
gen_model = pretrained_models['generative_model'].cuda()
classifier = pretrained_models['classifier_on_reconstructed'].cuda()
accuracies = []
acc = test_classifier_on_generator(classifier, gen_model, test_loader)
print('Accuracy of the pretrained classifier on full test data: ' + str(acc))
accuracies.append(acc)
max_test_acc = accuracies[0]
historical_storage_full = {}
historical_storage_real = {}

print('Reconstructing historical data from codes')
for idx_class in pretrained_on_classes:
  indices_class = sup_functions.get_indices_for_classes(original_trainset, [idx_class])
  class_loader = data_utils.DataLoader(original_trainset, batch_size=opts.fake_storage_size, sampler = SubsetRandomSampler(indices_class))
  for (X, Y) in class_loader:
    historical_storage_full[str(idx_class)] = gen_model(X.cuda())
    historical_storage_full[str(idx_class)][-opts.real_storage_size:] = X[-opts.real_storage_size:]
    historical_storage_real[str(idx_class)] = X[-opts.real_storage_size:]
    break
print('Finished reconstructing historical data')


generative_optimizer_classification = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta1, betas=(0.9, 0.999), weight_decay=1e-5)
generative_criterion_classification = nn.MSELoss()

classification_optimizer = optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)
classification_criterion = nn.CrossEntropyLoss()

if opts.cuda:
  gen_model = gen_model.cuda()
  generative_criterion_classification = generative_criterion_classification.cuda()
  classifier = classifier.cuda()
  classification_criterion = classification_criterion.cuda()

stream_classes = []
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
  
while Stream:
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
        historical_storage_full[str(idx_class)] = X.cuda()
        historical_storage_real[str(idx_class)] = X[-opts.real_storage_size:]
        break
    
    stream_classes.append(new_data_classes)
    train_data, train_labels = regroup_data_from_storage(historical_storage_full, opts) #TODO
    stream_trainset = TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(stream_trainset, batch_size=opts.batch_size, shuffle = True)
    
    print('Training generative model')
    generative_optimizer_classification.zero_grad()
    classification_optimizer.zero_grad()
    for idx, (train_X, train_Y) in enumerate(train_loader):
      inputs = train_X.float()
      labels = train_Y
      if opts.cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
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
    
    print('Retraining the classifier')
    classification_optimizer = optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)

    classification_optimizer.zero_grad()
    generative_optimizer_classification.zero_grad()
    for idx, (train_X, train_Y) in enumerate(train_loader_classif):
      inputs = gen_model(train_X.float().cuda())
      labels = train_Y.cuda()
      orig_classes = classifier(inputs)
      orig_classes.require_grad = True
      classification_loss = classification_criterion(orig_classes, labels.long())
      classification_loss.backward()
      classification_optimizer.step()
      classification_optimizer.zero_grad()
      if idx%100==0:
        print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, opts.niter_classification,  classification_loss.item()))
        
  for idx_class in historical_storage_real.keys():
    historical_storage_full[str(idx_class)] = gen_model(historical_storage_real[str(idx_class)])      
  acc = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
  print('Test accuracy, classification training on reconstructed data: ' + str(acc))
  accuracies.append(acc)
  torch.save(accuracies, opts.root+'results/stream_training_accuracy_' + name_to_save)
  if acc > max_test_acc:
    max_test_acc = acc
    best_models['classifier'] = classifier
    best_models['generative_model'] = gen_model
    torch.save(best_models, opts.root + 'pretrained_models/stream_training_' + name_to_save)  

  print('Finished Training')
