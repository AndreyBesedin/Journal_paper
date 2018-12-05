import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import synthetic_data_generation
import sup_functions
import models


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='MNIST | LSUN | Synthetic')
parser.add_argument('--root', default='/home/besedin/workspace/Projects/Journal_paper/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default = 0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device to use')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--MNIST_classes', type=int, default=10, help='nb of classes from MNIST by default')
parser.add_argument('--LSUN_classes', type=int, default=30, help='nb of classes from MNIST by default')
parser.add_argument('--optimizer', default='SGD', help='Adam, SGD')
parser.add_argument('--load_classifier', action='store_true', help='if enabled, load pretrained classifier with corresponding characteristics')

#Synthetic data options
parser.add_argument('--nb_of_classes', default=100, type=int, help='number of classes in synthetic dataset')
parser.add_argument('--class_size', default=100, type=int, help='number of elements in each class')
parser.add_argument('--feature_size', default=2048, type=int, help='feature size in synthetic dataset')
parser.add_argument('--generate_data', action='store_true', help='generates a new dataset if enabled, loads predefined otherwise')

opts = parser.parse_args()

print(opts)
for directory in ('results', 'pretrained_models', 'datasets'):
  if not os.path.exists(opts.root+directory):
    print('Creating a %s folder'.format(opts.root+directory))
    os.makedirs(opts.root+directory)
  
if opts.manual_seed is None:
  opts.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opts.manual_seed)
random.seed(opts.manual_seed)
torch.manual_seed(opts.manual_seed)

if torch.cuda.is_available() and not opts.cuda:
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
print('Loading data')
if opts.dataset=='MNIST':
  opts.nb_of_classes=opts.MNIST_classes
elif opts.dataset=='LSUN':
  opts.nb_of_classes=opts.LSUN_classes
  
opts.experiment_name = opts.dataset + '_' + opts.nb_of_classes + '_classes'

trainset, testset = sup_functions.load_dataset(opts)
train_loader = data_utils.DataLoader(trainset, batch_size=opts.batch_size, shuffle = True)
test_loader = data_utils.DataLoader(testset, batch_size=opts.batch_size, shuffle = False)
classifier = sup_functions.init_classifier(opts)
if opts.load_classifier:
  classifier_dict = torch.load(opts.root+'pretrained_models/batch_classifier_' + opts.experiment_name +'.pth')
  classifier.load_state_dict(classifier_dict)

criterion = nn.CrossEntropyLoss(); 
if opts.cuda: criterion = criterion.cuda()
optimizer = optim.SGD(classifier.parameters(), lr=opts.lr, momentum=0.99)
if opts.optimizer=='Adam':
  optimizer = optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)
        
max_test_acc = 0
accuracies = torch.FloatTensor(opts.niter)
for epoch in range(opts.niter):  # loop over the dataset multiple times
  print('Training epoch ' + str(epoch))
  sup_functions.train_classifier(classifier, train_loader, optimizer, criterion)
  accuracies[epoch] = sup_functions.test_classifier(classifier, test_loader)
  
  if accuracies[epoch] > max_test_acc:
    max_test_acc = accuracies[epoch]
    best_classifier = classifier
    torch.save(best_classifier.state_dict(), opts.root+'pretrained_models/batch_classifier_' + opts.experiment_name +'.pth')
      
  print('Test accuracy: ' + str(accuracies[epoch]))

  torch.save(accuracies,opts.root+'results/batch_classification_accuracy_' + opts.experiment_name + '.pth' )
print('Finished Training')
