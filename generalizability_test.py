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
parser.add_argument('--experiment_type', default='generalizability', help='batch_classification | representativity | generalizability | incremental_stream | unordered_stream')
parser.add_argument('--generator_type', default='AE', help='AE | CGAN | ACGAN')
parser.add_argument('--code_size', default='32', help='Size of the code representation in autoencoder')
parser.add_argument('--root', default='/home/besedin/workspace/Projects/Journal_paper/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--image_size', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--betta1', type=float, default=0.2, help='trade-off coefficients for ae training')
parser.add_argument('--betta2', type=float, default=1, help='trade-off coefficients for ae training')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default = 0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device to use')
parser.add_argument('--load_classifier', action='store_true', help='if enabled, load pretrained classifier with corresponding characteristics')
parser.add_argument('--load_gen_model', action='store_true', help='if enabled, load pretrained generative model')

parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--MNIST_classes', type=int, default=10, help='nb of classes from MNIST by default')
parser.add_argument('--LSUN_classes', type=int, default=30, help='nb of classes from MNIST by default')
parser.add_argument('--optimizer', default='Adam', help='Adam | SGD')
#Synthetic data options
parser.add_argument('--nb_of_classes', default=10, type=int, help='number of classes in synthetic dataset')
parser.add_argument('--class_size', default=100, type=int, help='number of elements in each class')
parser.add_argument('--feature_size', default=2048, type=int, help='feature size in synthetic dataset')
parser.add_argument('--generate_data', action='store_true', help='generates a new dataset if enabled, loads predefined otherwise')

opts = parser.parse_args()

if opts.cuda:
  torch.cuda.set_device(opts.cuda_device)
print('Loading data')

if opts.dataset=='MNIST' or opts.dataset=='MNIST_features':
  opts.nb_of_classes=opts.MNIST_classes
elif opts.dataset=='LSUN':
  opts.nb_of_classes=opts.LSUN_classes
  
AE_specific = ''
if opts.generator_type == 'AE':
#  AE_specific = '_' + str(opts.code_size) + '_trade-off_' + str(opts.betta1) + '_'
  AE_specific = '_' + str(opts.code_size) +'_cl_loss_' + str(opts.betta1) + '_rec_loss_' + str(opts.betta2) + '_'
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
trainset, testset = sup_functions.load_dataset(opts)
gen_model = torch.load(opts.root + 'pretrained_models/representativity_' + opts.dataset + '_' + opts.generator_type + AE_specific + str(opts.nb_of_classes) + '_classes.pth')
#gen_model = sup_functions.init_generative_model(opts)

classifier = sup_functions.init_classifier(opts)

classification_criterion = nn.CrossEntropyLoss(); 
classification_optimizer = optim.SGD(classifier.parameters(), lr=opts.lr, momentum=0.99)
if opts.optimizer=='Adam':
  optimizer = optim.Adam(classifier.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999), weight_decay=1e-5)

if opts.cuda:
  gen_model = gen_model.cuda()
  classification_criterion = classification_criterion.cuda()
  classifier = classifier.cuda()
  
print('Reconstructing data')
trainset = sup_functions.reconstruct_dataset_with_AE(trainset, gen_model, bs = 1000)  
testset = sup_functions.reconstruct_dataset_with_AE(testset, gen_model, bs = 1000)


  
train_loader = data_utils.DataLoader(trainset, batch_size=opts.batch_size, shuffle = True)
test_loader = data_utils.DataLoader(testset, batch_size=opts.batch_size, shuffle = False)


max_test_acc = 0
accuracies = []
for epoch in range(opts.niter):  # loop over the dataset multiple times
  print('Training epoch ' + str(epoch))
  for idx, (train_X, train_Y) in enumerate(train_loader):
    inputs = train_X.float().cuda()
    labels = train_Y.cuda()
    orig_classes = classifier(inputs)
    classification_loss = classification_criterion(orig_classes, labels.long())
    classification_loss.backward()
    classification_optimizer.step()
    classification_optimizer.zero_grad()
    if idx%100==0:
      print('epoch [{}/{}], classification loss: {:.4f}'.format(epoch, opts.niter,  classification_loss.item()))  
  acc = sup_functions.test_classifier(classifier, test_loader)
  accuracies.append(acc)
  if accuracies[-1] > max_test_acc:
    max_test_acc = accuracies[-1]
    best_classifier = classifier
    torch.save(best_classifier, opts.root+'pretrained_models/generalizability_classifier_' + name_to_save)      
  print('Test accuracy: ' + str(accuracies[-1]))

  torch.save(accuracies, opts.root+'results/generalizability_accuracy_' + name_to_save)
print('Finished Training')
