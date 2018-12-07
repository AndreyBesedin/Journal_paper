import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils 
from synthetic_data_generation import initialize_synthetic_sampler, sample_data_from_sampler
from sup_functions import test_model, test_model_on_gen, weights_init, reconstruct_dataset_with_AE
from models import Net, autoencoder
import new_models
root = '~/workspace/Projects/Journal_paper/'
dataset = 'LSUN'
print('Loading data')


dim = 2048
nb_classes = 30
epochs = 20
train_class_size = 2000
test_class_size = 2000
batch_size = 100

trainset_ = torch.load('../datasets/LSUN/trainset.pth')
testset_  = torch.load('../datasets/LSUN/testset.pth')
trainset = data_utils.TensorDataset(trainset_[0], trainset_[1])
testset = data_utils.TensorDataset(testset_[0], testset_[1])

rec_model = autoencoder(32)
state = torch.load('./models/AE_LSUN_32_code_size_'+str(nb_classes)+'_classes.pth')
rec_model.load_state_dict(state)

print('Reconstructing data')
trainset = reconstruct_dataset_with_AE(trainset, rec_model.cuda(), bs = 1000, real_data_ratio=0)  
testset = reconstruct_dataset_with_AE(testset, rec_model.cuda(), bs = 1000, real_data_ratio=0)

train_loader = data_utils.DataLoader(trainset, batch_size=batch_size, shuffle = True)
test_loader = data_utils.DataLoader(testset, batch_size=batch_size, shuffle = False)
#rec_model = autoencoder(32)
#state = torch.load('./models/AE_32_code_size_500_classes_2000_samples.pth')
#rec_model.load_state(state)
pretrained_model_dict = torch.load('../pretrained_models/batch_classifier_LSUN_30_classes.pth')
pretrained_model = new_models.Classifier_2048_features(30)
pretrained_model.load_state_dict(pretrained_model_dict)
acc = test_model(pretrained_model.cuda(), train_loader)
print('Accuracy of pretrained model on erconstructed dataset: ' + str(acc))
print('Training')

model = Net(nb_classes).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
data_size = len(trainset[0])
max_test_acc = 0
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
#    for idx, data in enumerate(trainloader, 0):
    for idx, (train_X, train_Y) in enumerate(train_loader):
      inputs = train_X.cuda()
      labels = train_Y.cuda()

        # zero the parameter gradients
      optimizer.zero_grad()

        # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels.long())
      loss.backward()
      optimizer.step()

        # print statistics
      running_loss += loss.item()
      if idx % 50 == 49:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, idx + 1, running_loss / 100))
        running_loss = 0.0
        
    test_acc = test_model(model, test_loader)
    if test_acc > max_test_acc:
      max_test_acc = test_acc
      best_model = model.float()
      #torch.save(best_model, './results/synthetic/models/'+dataset+'_classifier_original.pth')
      
    print('Test accuracy: ' + str(test_acc))


torch.save(model, './models/LSUN_batch_classifier_'+ str(nb_classes) +'_classes_rec.pth')
#torch.save(data_sampler, './models/data_sampler_'+ str(nb_classes) +'_classes.pth')
print('Test accuracy')
print('Finished Training')
