import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

nb_of_classes = 500
code_size = 32
training_epochs = 200
torch.cuda.set_device(1)
opts = {
  'batch_size': 500,
  'learning_rate': 0.0005,
  'betta0': 0.1,
  'betta1': 0.01, # Influence coefficient for classification loss in AE default 1e-2
  'betta2': 3, # Influence coefficient for reconstruction loss in AE
  }

 
class Classifier_128_features(nn.Module):
  def __init__(self, nb_classes):
    super(Classifier_128_features, self).__init__()
    self.fc1 = nn.Linear(128, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, nb_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
 
class autoencoder_128_features(nn.Module):
  def __init__(self, code_size):
    def linear_block(in_, out_):
#      return nn.Sequential(nn.Linear(in_, out_), nn.ReLU(True))
      return nn.Sequential(nn.Linear(in_, out_), nn.BatchNorm1d(out_), nn.ReLU(True))
    super(autoencoder_128_features, self).__init__()
    self.encoder = nn.Sequential(
      linear_block(128, 128),
      linear_block(128, 192),
      linear_block(192, 128),
      nn.Linear(128, code_size),
    )
    self.decoder = nn.Sequential(
      linear_block(code_size, 128),
      linear_block(128, 192),
      linear_block(192, 128),
      nn.Linear(128, 128),
#      nn.Tanh()
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
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
full_data = torch.load('./data/Synthetic/data_train_test_500_classes_128_features_2000_samples.pth')
trainset = TensorDataset(full_data['data_train'], full_data['labels_train'])
testset = TensorDataset(full_data['data_test'], full_data['labels_test'])

# Initializing data loaders for first 5 classes
train_loader = DataLoader(trainset, batch_size=opts['batch_size'], shuffle=True)
test_loader = DataLoader(testset, batch_size=opts['batch_size'], shuffle=False)

# Initializing classification model
classifier = Classifier_128_features(nb_of_classes)
#classifier_dict = torch.load('./pretrained_models/full_classifier_synthetic_data.pth')
#classifier.load_state_dict(classifier_dict)
classifier.cuda()

gen_model = autoencoder_128_features(code_size)
gen_model.cuda()
full_model = nn.Sequential(gen_model, classifier)
full_model.cuda()
#generative_optimizer = torch.optim.Adam(gen_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
#generative_criterion = nn.MSELoss()
#generative_criterion.cuda()

classification_criterion = nn.CrossEntropyLoss()
classification_criterion.cuda()


training_optimizer = torch.optim.Adam(full_model.parameters(), lr=opts['learning_rate'], betas=(0.9, 0.999), weight_decay=1e-5)
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
    labels = train_Y.long().cuda()
    # ===================forward=====================
    reconstructions = gen_model(inputs)
    predictions_original = classifier(inputs)
    predictions_reconstructed = classifier(reconstructions)
    #loss_gen_rec = 0
    loss_cl = 0
    if idx%10==0:
      loss_cl = opts['betta0']*classification_criterion(predictions_original, labels)
    loss_gen_rec = opts['betta2']*generative_criterion_rec(reconstructions, inputs)
    loss_gen_cl = opts['betta1']*generative_criterion_cl(predictions_reconstructed, predictions_original.data)
    full_loss = loss_cl + loss_gen_cl + loss_gen_rec
    full_loss.register_hook(lambda grad: print(grad))
    full_loss.backward()
    print(full_loss.grad)
    training_optimizer.step()
    training_optimizer.zero_grad()   
    if idx%100==0:
      print('cl_loss: {}; gen_cl: {}; gen_rec: {}'.format(loss_cl, loss_gen_cl, loss_gen_rec))
      #print('epoch [{}/{}], generators loss: {:.4f}'
      #  .format(epoch+1, training_epochs,  full_loss.item()))

    # ===================backward====================
  acc_orig = test_classifier(classifier, test_loader)
#  acc_train = test_classifier_on_generator(classifier, gen_model, train_loader)
  acc_test = test_classifier_on_generator(classifier, gen_model, test_loader)
  print('Test accuracy on the original testset: {}'.format(acc_orig))
#  print('Test accuracy on reconstructed trainset: {}'.format(acc_train))
  print('Test accuracy on reconstructed testset: {}'.format(acc_test))
  print('Learning rate for the experiment: {}'.format(opts['learning_rate']))
  if acc_test > max_accuracy:
    max_accuracy = acc_test
    #torch.save(gen_model.state_dict(), './pretrained_models/AE_32_synthetic_data.pth')
      
print('End of training, max accuracy {}'.format(max_accuracy))    
print(opts)    
