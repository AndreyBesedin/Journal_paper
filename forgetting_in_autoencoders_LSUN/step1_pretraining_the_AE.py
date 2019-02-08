import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import AE_models

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

nb_of_classes = 30 
code_size = 32
training_epochs = 100
torch.cuda.set_device(0)
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
  #TODO add cases of models other than AE
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


# Loading the datasets
trainset = torch.load('../datasets/LSUN/trainset.pth')
testset = torch.load('../datasets/LSUN/testset.pth')

trainset = TensorDataset(trainset[0], trainset[1])
testset = TensorDataset(testset[0], testset[1])

# Initializing data loaders for first 5 classes

train_loader = DataLoader(trainset, batch_size=opts['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=opts['batch_size'], shuffle=False)

# Initializing classification model
classifier = Classifier(nb_of_classes)
classifier_dict = torch.load('./pretrained_models/full_classifier_LSUN.pth')
classifier.load_state_dict(classifier_dict)
classifier.cuda()

gen_model = AE_models.AE_LSUN(code_size)
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
  for idx, (train_X, train_Y) in enumerate(train_loader):
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
  acc_train = test_classifier_on_generator(classifier, gen_model, train_loader)
  acc_test = test_classifier_on_generator(classifier, gen_model, test_loader)
  print('Test accuracy on reconstructed trainset: {}'.format(acc_train))
  print('Test accuracy on reconstructed testset: {}'.format(acc_test))
  print('Learning rate for the experiment: {}'.format(opts['learning_rate']))
  if acc_test > max_accuracy:
    max_accuracy = acc_test
    torch.save(gen_model.state_dict(), './pretrained_models/AE_32_LSUN_data.pth')
      
print('End of training, max accuracy {}'.format(max_accuracy))    
print(opts)    
