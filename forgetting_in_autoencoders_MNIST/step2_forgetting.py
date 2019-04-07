import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import AE_models
import data_buffer 
import models
import sup_functions

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

nb_of_classes = 10
code_size = 32 
training_epochs = 50

alpha1 = 0.3
alpha2 = 0.001

opts = {
  'batch_size': 100,
  'learning_rate': 0.001,
  'add_real': True,
  }

Classifier = models.Classifier_MNIST 
test_classifier = sup_functions.test_classifier
test_classifier_on_generator = sup_functions.test_classifier_on_generator
get_indices_for_classes = sup_functions.get_indices_for_classes

# Loading the datasets
trainset = torch.load('./data/trainset.pth')
testset = torch.load('./data/testset.pth')

original_trainset = TensorDataset(trainset[0], trainset[1])
trainset = TensorDataset(trainset[0], trainset[1])
testset = TensorDataset(testset[0], testset[1])

data_buffer = data_buffer.Data_Buffer(60, 100)
data_buffer.load_from_tensor_dataset(trainset)


# Initializing data loaders for first 5 classes
test_loader = DataLoader(testset, batch_size=opts['batch_size'], shuffle=False)

# Initializing classification model
classifier = Classifier(nb_of_classes)
classifier_dict = torch.load('./pretrained_models/classifier_10_MNIST.pth')
classifier.load_state_dict(classifier_dict)
classifier.cuda()

gen_model = models.AE_MNIST(code_size)
gen_dict = torch.load('./pretrained_models/AE_10_classes_32_code_size_MNIST.pth')
gen_model.load_state_dict(gen_dict)
gen_model.cuda()
#generative_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
#generative_criterion = nn.MSELoss()
#generative_criterion.cuda()

generative_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
generative_criterion_cl = nn.MSELoss()
generative_criterion_cl.cuda()
generative_criterion_rec = nn.MSELoss()
generative_criterion_rec.cuda()

acc = test_classifier(classifier, test_loader)
print('Classification accuracy prior to training: {:.4f}'.format(acc))

max_accuracy = 0
for epoch in range(training_epochs):  # loop over the dataset multiple times
  data_buffer.transform_data(gen_model)
  if opts['add_real']:
    data_buffer.add_batches_from_dataset(original_trainset, range(10), 1)
  data_buffer.make_tensor_dataset()
  trainset = data_buffer.tensor_dataset
  train_loader = DataLoader(trainset, batch_size=opts['batch_size'], shuffle=True)
  for idx, (train_X, train_Y, _) in enumerate(train_loader):
    inputs = train_X.cuda()
    # ===================forward=====================
    #reconstructions = gen_model(inputs)
    #orig_classes = classifier(inputs)
    #classification_reconstructed = classifier(reconstructions)
    #loss_gen = generative_criterion(classification_reconstructed, orig_classes)
    #loss_gen.backward()
    #generative_optimizer.step()
    #generative_optimizer.zero_grad()
    
    reconstructions = gen_model(inputs)
    orig_classes = classifier(inputs).detach()
    classification_reconstructed = classifier(reconstructions)
    loss_gen_rec = alpha2*generative_criterion_rec(reconstructions, inputs)
    loss_gen_cl = alpha1*generative_criterion_cl(classification_reconstructed, orig_classes)
    loss_gen = loss_gen_cl + loss_gen_rec
    loss_gen.backward()
    generative_optimizer.step()
    generative_optimizer.zero_grad()   
    if idx%100==0:
      print('epoch [{}/{}], generators loss: {:.4f}'
        .format(epoch+1, training_epochs,  loss_gen.item()))

    # ===================backward====================
    
  acc = test_classifier_on_generator(classifier, gen_model, test_loader)
  print('Test accuracy on reconstructed: ' + str(acc))
#  if acc > max_accuracy:
#    max_accuracy = acc
#    torch.save(gen_model.state_dict(), './pretrained_models/AE_10_classes_32_code_size_MNIST.pth')
    
