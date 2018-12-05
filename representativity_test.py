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
parser.add_argument('--generator_type', default='autoencoder', help='autoencoder | CGAN | ACGAN')
parser.add_argument('--code_size', default='32', help='Size of the code representation in autoencoder')
parser.add_argument('--root', default='/home/besedin/workspace/Projects/Journal_paper/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
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
parser.add_argument('--optimizer', default='SGD', help='Adam, SGD')
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
  
opts.experiment_name = str(opts.nb_of_classes) +'_classes'
if opts.dataset == 'Synthetic':
  opts.experiment_name = opts.experiment_name + '_' + str(opts.class_size) + '_samples'
  
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
classifier = sup_functions.init_classifier(opts)
gen_model = sup_functions.init_generative_model(opts)

print(gen_model)
criterion_AE = nn.MSELoss()
criterion_classif = nn.MSELoss()

if opts.cuda:
  gen_model = gen_model.cuda()
  criterion_AE = criterion_AE.cuda()
  criterion_classif = criterion_classif.cuda()

optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=opts.lr, betas=(0.9, 0.999),
                             weight_decay=1e-5)

print('Classification accuracy on the original testset: ' + str(sup_functions.test_classifier(classifier, test_loader)))
max_test_acc = 0
accuracies = []
#TODO add score computation as in the paper
for epoch in range(opts.niter):  # loop over the dataset multiple times
  print('Training epoch ' + str(epoch))
#  bar = Bar('Training: ', max=int(opts['nb_classes']*opts['samples_per_class_train']/opts['batch_size']))
  sup_functions.train_gen_model(gen_model, classifier, train_loader, optimizer_gen, criterion_AE, criterion_classif, opts)
  acc = sup_functions.test_classifier_on_generator(classifier, gen_model, test_loader)
  accuracies.append(acc)
  if accuracies[-1] > max_test_acc:
    max_test_acc = accuracies[-1]
    best_gen_model = gen_model
    torch.save(best_gen_model, opts.root+'pretrained_models/'+opts.dataset+'_' + opts.generator_type + str(opts.code_size)*(opts.generator_type=='autoencoder').real + '_' + opts.experiment_name + '.pth')
      
  print('Test accuracy: ' + str(accuracies[-1]))

  torch.save(accuracies, opts.root+'results/representativity_' + opts.dataset + '_' + opts.generator_type + str(opts.code_size)*(opts.generator_type=='autoencoder').real + '_' + opts.experiment_name + '.pth' )
  
print('Finished Training')
