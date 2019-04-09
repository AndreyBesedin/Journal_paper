import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models
import sup_functions

from data_buffer import Data_Buffer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

nb_of_classes = 30
code_size = 32
training_epochs = 100
cuda_device = 0 
torch.cuda.set_device(cuda_device)
real_batches_to_add = 0
batches_per_class = 50
reconstruct_every = 1
opts = {
  'batch_size': 100,
  'learning_rate': 0.001,
  'betta1': 0.1, # Influence coefficient for classification loss in AE default 1e-2
  'betta2': 1, # Influence coefficient for reconstruction loss in AE
  }


Classifier = models.Classifier_LSUN
test_classifier = sup_functions.test_classifier
test_classifier_on_generator = sup_functions.test_classifier_on_generator
get_indices_for_classes = sup_functions.get_indices_for_classes

trainset = torch.load('../datasets/LSUN/trainset.pth')
testset = torch.load('../datasets/LSUN/testset.pth')

original_trainset = TensorDataset(trainset[0], trainset[1], torch.zeros(trainset[1].shape))
trainset = TensorDataset(trainset[0], trainset[1], torch.zeros(trainset[1].shape))
testset = TensorDataset(testset[0], testset[1])

# Loading the datasets
print('Reshaping data into readable format')
hist_buffer = Data_Buffer(real_batches_to_add or 1, opts['batch_size'])
hist_buffer.load_from_tensor_dataset(trainset, list(range(30)))
hist_buffer.make_tensor_dataset()
hist_data = hist_buffer.tensor_dataset

data_buffer = Data_Buffer(batches_per_class, opts['batch_size'])
data_buffer.load_from_tensor_dataset(trainset, list(range(30)))                                         
data_buffer.cuda_device = cuda_device

print('Ended reshaping')
# Initializing data loaders for first 5 classes

#train_loader = DataLoader(original_trainset, batch_size=opts['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=opts['batch_size'], shuffle=False)

# Initializing classification model
classifier = Classifier(nb_of_classes)
classifier_dict = torch.load('./pretrained_models/classifier_30_LSUN.pth')
classifier.load_state_dict(classifier_dict)
classifier.cuda()

gen_model = models.AE_LSUN(code_size)
gen_model_dict = torch.load('./pretrained_models/AE_32_LSUN.pth')
gen_model.load_state_dict(gen_model_dict)
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
acc_rec = test_classifier_on_generator(classifier, gen_model, test_loader)
print('Classification accuracy prior to training: {:.4f}'.format(acc))
print('Test accuracy on reconstructed testset: {}'.format(acc_rec))

accuracies = []
max_accuracy = 0
for epoch in range(training_epochs):  # loop over the dataset multiple times
#  if epoch % reconstruct_every == 0 and epoch>=1:
  print('Transforming data with the latest autoencoder')
  data_buffer.transform_data(gen_model)
  if real_batches_to_add>0:
    data_buffer.add_batches_from_dataset(hist_data, list(range(30)), real_batches_to_add)
  data_buffer.make_tensor_dataset()
  trainset = data_buffer.tensor_dataset
  train_loader = DataLoader(trainset, batch_size=10*opts['batch_size'], shuffle=True, drop_last=True)
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
    #loss_gen_rec = 0
    loss_gen_rec = opts['betta2']*generative_criterion_rec(reconstructions, inputs)
    loss_gen_cl = opts['betta1']*generative_criterion_cl(classification_reconstructed, orig_classes)
    loss_gen = loss_gen_cl + loss_gen_rec
    loss_gen.backward()
    generative_optimizer.step()
    generative_optimizer.zero_grad()   
    if idx%100==0:
      print('epoch [{}/{}], generators loss: {:.4f}'
        .format(epoch+1, training_epochs,  loss_gen.item()))

    # ===================backward====================
  #acc_test = test_classifier(classifier, test_loader)
  acc_test_rec = test_classifier_on_generator(classifier, gen_model, test_loader)
  accuracies.append(acc_test_rec)
  #print('Test accuracy on trainset: {}'.format(acc_train))
  print('Test accuracy on reconstructed testset: {}'.format(acc_test_rec))
  print('Learning rate for the experiment: {}'.format(opts['learning_rate']))
  #if acc_test_rec > max_accuracy:
  #  max_accuracy = acc_test_rec
#    torch.save(gen_model.state_dict(), './pretrained_models/AE_32_synthetic_data_forgetting.pth')
  torch.save(accuracies, './results/forgetting_AE_LSUN_{}_real_batches.pth'.format(real_batches_to_add)) 
print('End of training, max accuracy {}'.format(max_accuracy))    
print(opts)    
