import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import AE_models

from data_buffer import Data_Buffer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

nb_of_classes = 30
code_size = 32
training_epochs = 100
cuda_device = 1 
torch.cuda.set_device(cuda_device)
real_batches_to_add = 2
batches_per_class = 20
reconstruct_every = 2
opts = {
  'batch_size': 500,
  'learning_rate': 0.001,
  'betta1': 0.1, # Influence coefficient for classification loss in AE default 1e-2
  'betta2': 1, # Influence coefficient for reconstruction loss in AE
  }

 
class Classifier(nn.Module):
  def __init__(self, nb_classes):
    super(Classifier, self).__init__()
    self.fc1 = nn.Linear(2048, 784)
    self.fc2 = nn.Linear(784, 256)
    self.fc3 = nn.Linear(256, nb_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
 
def test_classifier(classif, data_loader):
  total = 0
  correct = 0
  for test_X, test_Y in data_loader:
    input_test = test_X.float().cuda()
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  return correct/total*100

def test_classifier_on_generator(classif, gen_model, data_loader):
  #TODO add cases of models other than AEi
  total = 0
  correct = 0
  gen_model.eval()
  for idx, (test_X,  test_Y) in enumerate(data_loader):
    input_test = gen_model(test_X.float().cuda())
    outputs = classif(input_test)
    _, predicted = torch.max(outputs.data, 1)
    labels = test_Y.long()
    total += labels.size(0)
    correct += (predicted.cpu().long() == labels).sum().item()
  gen_model.train()
  return correct/total*100

def get_indices_for_classes(data, data_classes):
  # Creates a list of indices of samples from the dataset, corresponding to given classes
  indices = torch.FloatTensor(list((data.tensors[1].long()==class_).tolist() for class_ in data_classes)).sum(0).nonzero().long().squeeze()
  return indices

trainset = torch.load('../datasets/LSUN/testset.pth')
testset = torch.load('../datasets/LSUN/testset.pth')

original_trainset = TensorDataset(trainset[0], trainset[1], torch.zeros(trainset[1].shape))
testset = TensorDataset(testset[0], testset[1])

# Loading the datasets
print('Reshaping data into readable format')
data_buffer = Data_Buffer(batches_per_class, opts['batch_size'])
data_buffer.add_batches_from_dataset(original_trainset, list(range(30)), batches_per_class)                                         
data_buffer.cuda_device = cuda_device

orig_buffer = Data_Buffer(real_batches_to_add, opts['batch_size'])
orig_buffer.add_batches_from_dataset(original_trainset, list(range(30)), real_batches_to_add)
orig_buffer.cuda_device = cuda_device
original_trainset = orig_buffer.make_tensor_dataset()

print('Ended reshaping')
# Initializing data loaders for first 5 classes

#train_loader = DataLoader(original_trainset, batch_size=opts['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=opts['batch_size'], shuffle=False)

# Initializing classification model
classifier = Classifier(nb_of_classes)
classifier_dict = torch.load('./pretrained_models/full_classifier_LSUN.pth')
classifier.load_state_dict(classifier_dict)
classifier.cuda()

gen_model = AE_models.AE_LSUN(code_size)
gen_model_state = torch.load('./pretrained_models/AE_32_LSUN_data.pth')
gen_model.load_state_dict(gen_model_state)
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
  data_buffer.add_batches_from_dataset(original_trainset, list(range(30)), real_batches_to_add)
  trainset = data_buffer.make_tensor_dataset()
  train_loader = DataLoader(trainset, batch_size=opts['batch_size'], shuffle=True, drop_last=True)
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
  if acc_test_rec > max_accuracy:
    max_accuracy = acc_test_rec
#    torch.save(gen_model.state_dict(), './pretrained_models/AE_32_synthetic_data_forgetting.pth')
  torch.save(accuracies, './results/forgetting_AE_LSUN_{}_real_batches_old_loss.pth'.format(real_batches_to_add)) 
print('End of training, max accuracy {}'.format(max_accuracy))    
print(opts)    
