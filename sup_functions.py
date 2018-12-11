import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from progress.bar import Bar
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models, synthetic_data_generation
import torchvision.transforms.functional as TF

def to_img(x):
  x = 0.5 * (x + 1)
  x = x.clamp(0, 1)
  x = x.view(x.size(0), 1, 28, 28)
  return x

def train_classifier(classifier_, train_loader_, optimizer_, criterion_):
  running_loss = 0.0
  for idx, (train_X, train_Y) in enumerate(train_loader_):
    inputs = train_X.float().cuda()
    labels = train_Y.float().cuda()
    # zero the parameter gradients
    optimizer_.zero_grad()

    # forward + backward + optimize
    outputs = classifier_(inputs)
    loss = criterion_(outputs, labels.long())
    loss.backward()
    optimizer_.step()

    # print statistics
    running_loss += loss.item()
    if idx % 50 == 49:    # print every 2000 mini-batches
      print('Iteration: %5d loss: %.3f' % (idx + 1, running_loss / 100))
      running_loss = 0.0
    
def train_gen_model(gen_model, classifier, train_loader, criterion_classif, optimizer_classif, criterion_AE, optimizer_gen, opts):
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
    optimizer_classif.zero_grad()
    optimizer_gen.zero_grad()
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
        print('AE loss: {:.4f}'
          .format(loss_AE.item()))
      elif int(opts.betta2)==0:
        print('classification loss: {:.4f}'
          .format(loss_classif.item()))
      else:
        print('classification loss: {:.4f}, AE loss: {:.4f}'
          .format(loss_classif.item(), loss_AE.item()))
    # ===================backward====================
  return 0
      
def test_classifier(classif_, data_loader_):
  total = 0
  correct = 0
  for idx, (test_X, test_Y) in enumerate(data_loader_):
    input_test = test_X.float().cuda()
    outputs = classif_(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

def test_classifier_on_generator(classif_, gen_model_, data_loader_):
  #TODO add cases of models other than AE
  total = 0
  correct = 0
  gen_model_.eval()
  for idx, (test_X,  test_Y) in enumerate(data_loader_):
    input_test = gen_model_(test_X.float().cuda())
    outputs = classif_(input_test)
#    outputs = classif_(test_X.cuda())
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  gen_model_.train()
  return correct/total*100

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

def reconstruct_dataset_with_AE(dataset, rec_model, bs = 100, real_data_ratio=0):
  data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
  data_size = dataset.tensors[0].shape[0]
  res_data = torch.zeros(dataset.tensors[0].shape)
  res_labels = torch.zeros(data_size)
  current_index = 0
  bar = Bar('Reconstructing data for absent classes:', max=int(data_size/bs))
  for idx, (train_x, train_y) in enumerate(data_loader):
      #call('nvidia-smi')
    bar.next()
    inputs = train_x.float().cuda()
    batch = rec_model(inputs)
    current_batch_size = batch.shape[0]
    res_data[current_index:current_index+current_batch_size] = batch.cpu().data
    res_labels[current_index:current_index+current_batch_size] = train_y
    current_index+=current_batch_size 
  bar.finish()
  return TensorDataset(res_data, res_labels)

def init_generative_model(opts):
  #TODO Add CGAN and ACGAN cases 
  gen_model = False
  if opts.dataset == 'MNIST':
    if opts.generator_type == 'AE':
      gen_model = models.autoencoder_MNIST(int(opts.code_size))
  elif opts.dataset == 'MNIST_features':
    gen_model = models.autoencoder_MNIST_512_features(int(opts.code_size))
  else:
    if opts.generator_type == 'AE':
      gen_model = models.autoencoder_2048(int(opts.code_size))
  if opts.load_gen_model:
    #TODO correct the name for loading
    print('Loading generator from')
    gen_model_state = torch.load(opts.root+'pretrained_models/'+opts.dataset+'_' + opts.generator_type + str(opts.code_size)*(opts.generator_type=='AE').real + '_' + opts.experiment_name + '.pth')
    gen_model.load_state_dict(gen_model_state)
  if opts.cuda:
    return gen_model.cuda()
  return gen_model

def init_classifier(opts):
  if opts.dataset == 'MNIST':
    opts.nb_of_classes = opts.MNIST_classes
    classifier = models.Classifier_MNIST_28x28(opts.nb_of_classes)
  elif opts.dataset == 'MNIST_features':
    opts.nb_of_classes = opts.MNIST_classes
    classifier = models.Classifier_MNIST_512_features(opts.nb_of_classes)    
  elif opts.dataset == 'LSUN':
    opts.nb_of_classes = opts.LSUN_classes
    classifier = models.Classifier_2048_features(opts.nb_of_classes)
  else:
    classifier = models.Classifier_2048_features(opts.nb_of_classes)
  if opts.load_classifier:
    print('Loading pretrained classifier')
    classifier_state = torch.load(opts.root+'pretrained_models/batch_classifier_' + opts.dataset + '_' + str(opts.nb_of_classes) + '_classes.pth')
    classifier.load_state_dict(classifier_state)
  if opts.cuda:
    return classifier.cuda()
  return classifier
  
class TensorDatasetMNIST():
  def __init__(self, *tensors, transform=None):
    self.transform = transform
    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    self.tensors = tensors

  def __getitem__(self, index):
    img = self.tensors[0][index].squeeze()
    img = Image.fromarray(img.numpy(), mode='L')
    if self.transform is not None:
      img = self.transform(img)
    return (img.reshape(1, 28, 28), self.tensors[1][index])

  def __len__(self):
    return self.tensors[0].size(0)
  
class TensorDataset2048():
  def __init__(self, *tensors, transform=None):
    self.transform = transform
    assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
    self.tensors = tensors

  def __getitem__(self, index):
    img = self.tensors[0][index].squeeze()
    img = Image.fromarray(img.reshape(2048,1).numpy(), mode='L')
    if self.transform is not None:
      img = self.transform(img)
    return (img.reshape(2048), self.tensors[1][index])

  def __len__(self):
    return self.tensors[0].size(0) 
  
def load_dataset(opts):
  print('Loading ' + opts.dataset + ' data')
  if not os.path.exists(opts.root+'datasets/'+opts.dataset):
    os.makedirs(opts.root+'datasets/'+opts.dataset)
  if opts.dataset=='MNIST':
    img_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.MNIST(root=opts.root+'datasets/MNIST/',
      train=True,
      download=True,
      transform=img_transform) 
    train_dataset = TensorDatasetMNIST(trainset.train_data.reshape(60000, 1, 28, 28), trainset.train_labels, transform=img_transform)
    testset = datasets.MNIST(root=opts.root+'datasets/MNIST/',
      train=False,
      download=True,
      transform=img_transform)
    test_dataset = TensorDatasetMNIST(testset.test_data.reshape(10000, 1, 28, 28), testset.test_labels, transform=img_transform)
  elif opts.dataset=='MNIST_features':
    tensor_train = torch.load(opts.root + 'datasets/MNIST/trainset.pth')
    tensor_test  = torch.load(opts.root + 'datasets/MNIST/testset.pth')
    train_dataset = TensorDataset(tensor_train[0], tensor_train[1])
    test_dataset = TensorDataset(tensor_test[0], tensor_test[1])
  elif opts.dataset=='LSUN':
    tensor_train = torch.load(opts.root + 'datasets/LSUN/trainset.pth')
    tensor_test  = torch.load(opts.root + 'datasets/LSUN/testset.pth')
    #train_dataset = TensorDataset2048(tensor_train[0], tensor_train[1], transform=img_transform)
    #test_dataset = TensorDataset2048(tensor_test[0], tensor_test[1], transform=img_transform)
    train_dataset = TensorDataset(tensor_train[0] - 0.46, tensor_train[1])
    test_dataset = TensorDataset(tensor_test[0] - 0.46, tensor_test[1])
    print('Trainset mean: ' + str(train_dataset.tensors[0].mean()))
    print('Trainset std: ' + str(train_dataset.tensors[0].std()))
    print('Testset mean: ' + str(test_dataset.tensors[0].mean()))
    print('Testset std: ' + str(test_dataset.tensors[0].std()))
  elif opts.dataset=='Synthetic':
    full_data = False
    if not opts.generate_data:
      try:
        if opts.first_half:
          full_data = torch.load(opts.root + 'datasets/Synthetic/data_train_test_'+str(opts.nb_of_classes)+'_classes_'+str(opts.class_size)+'_samples_first_half.pth')
        else:
          full_data = torch.load(opts.root + 'datasets/Synthetic/data_train_test_'+str(opts.nb_of_classes)+'_classes_'+str(opts.class_size)+'_samples.pth')
      except IOError:
        print('No data with corresponding characteristics found, creating new dataset')
        pass
    if not full_data:
      full_data = synthetic_data_generation.sample_big_data(opts.feature_size, opts.nb_of_classes, opts.class_size)
      torch.save(full_data, opts.root + 'datasets/Synthetic/data_train_test_'+str(opts.nb_of_classes)+'_classes_'+str(opts.class_size)+'_samples.pth')
    train_dataset = TensorDataset(full_data['data_train'], full_data['labels_train'])
    test_dataset = TensorDataset(full_data['data_test'], full_data['labels_test'])
  return train_dataset, test_dataset
