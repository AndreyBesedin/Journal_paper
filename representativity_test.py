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
parser.add_argument('--generator_type', default='AE', help='AE | CGAN | ACGAN')
parser.add_argument('--code_size', default='32', help='Size of the code representation in autoencoder')
parser.add_argument('--root', default='/home/besedin/workspace/Projects/Journal_paper/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.001')
parser.add_argument('--betta1', type=float, default=0.02, help='trade-off coefficients for ae training')
parser.add_argument('--betta2', type=float, default=1, help='trade-off coefficients for ae training')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default = 0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device to use')
parser.add_argument('--load_classifier', action='store_true', help='if enabled, load pretrained classifier with corresponding characteristics')
parser.add_argument('--load_gen_model', action='store_true', help='if enabled, load pretrained generative model')

parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--MNIST_classes', type=int, default=10, help='nb of classes from MNIST by default')
parser.add_argument('--LSUN_classes', type=int, default=30, help='nb of classes from MNIST by default')
parser.add_argument('--optimizer', default='Adam', help='Adam, SGD')
#Synthetic data options
parser.add_argument('--nb_of_classes', default=100, type=int, help='number of classes in synthetic dataset')
parser.add_argument('--class_size', default=100, type=int, help='number of elements in each class')
parser.add_argument('--feature_size', default=2048, type=int, help='feature size in synthetic dataset')
parser.add_argument('--generate_data', action='store_true', help='generates a new dataset if enabled, loads predefined otherwise')

opts = parser.parse_args()
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
trainset, testset = sup_functions.load_dataset(opts)
train_loader = data_utils.DataLoader(trainset, batch_size=opts.batch_size, shuffle = True)
test_loader = data_utils.DataLoader(testset, batch_size=opts.batch_size, shuffle = False)
opts.load_classifier = True
classifier = sup_functions.init_classifier(opts)
#classifier.eval()
gen_model = sup_functions.init_generative_model(opts)

criterion_AE = nn.MSELoss()
criterion_classif = nn.MSELoss()
optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta2, betas=(0.9, 0.999), weight_decay=1e-5)
optimizer_classif = torch.optim.Adam(gen_model.parameters(), lr=opts.lr*opts.betta1, betas=(0.9, 0.999), weight_decay=1e-5)

if opts.cuda:
  gen_model = gen_model.cuda()
  criterion_AE = criterion_AE.cuda()
  criterion_classif = criterion_classif.cuda()


print('Classification accuracy on the original testset: ' + str(sup_functions.test_classifier(classifier, test_loader)))
max_test_acc = 0
accuracies = []
#TODO add score computation as in the paper
for epoch in range(opts.niter):  # loop over the dataset multiple times
  opts.epoch = epoch
  print('Training epoch ' + str(epoch))
#  bar = Bar('Training: ', max=int(opts['nb_classes']*opts['samples_per_class_train']/opts['batch_size']))
  for idx, (train_X, train_Y) in enumerate(train_loader):
    inputs = train_X.float()
    labels = train_Y
    if opts.cuda:
      inputs = inputs.cuda()
      labels = labels.cuda()
    #if opts.cuda:
      #inputs = inputs.cuda()
      #labels = inputs.cuda()
    # ===================forward=====================
    
    if opts.betta1!=0:
      outputs = gen_model(inputs)
      orig_classes = classifier(inputs)
      orig_classes.require_grad=False
      classification_reconstructed = classifier(outputs)
      loss_classif = criterion_classif(classification_reconstructed, orig_classes)
      loss_classif.backward()
      optimizer_classif.step()
      optimizer_classif.zero_grad()
    if opts.betta2!=0:
      outputs = gen_model(inputs)
      loss_AE = criterion_AE(outputs, inputs)
      loss_AE.backward()
      optimizer_gen.step()
      optimizer_gen.zero_grad()
    
    if idx%100==0:
      if opts.betta1==0:
        print('epoch [{}/{}], AE loss: {:.4f}'
          .format(opts.epoch+1, opts.niter,  loss_AE.item()))
      elif opts.betta1==0:
        print('epoch [{}/{}], classification loss: {:.4f}'
          .format(opts.epoch+1, opts.niter,  loss_classif.item()))
      else:
        print('epoch [{}/{}], classification loss: {:.4f}, AE loss: {:.4f}'
          .format(opts.epoch+1, opts.niter, loss_classif.item(), loss_AE.item()))
    # ===================backward====================
    

  gen_model.eval()
  #classifier.eval()
  acc = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
  #classifier.train()
  gen_model.train()
  accuracies.append(acc)
  if accuracies[-1] > max_test_acc:
    max_test_acc = accuracies[-1]
    best_gen_model = gen_model
    torch.save(best_gen_model,opts.root + 'pretrained_models/representativity_' + name_to_save)
      
  print('Test accuracy: ' + str(accuracies[-1]))

  torch.save(accuracies, opts.root+'results/representativity_accuracy_' + name_to_save)
  
print('Finished Training')
