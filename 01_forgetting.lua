require 'nn'
require 'torch'
require 'cunn'
require 'cudnn'
require 'gnuplot'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')


opt = {
  batch_size = 100,
  cuda = true,
  classifier_type = 'conv+lin',
  epoch_number = 2,
  learning_rate = 0.001,
}


trainset = torch.load('/home/besedin/workspace/Data/MNIST/t7_files/trainset.t7')
testset = torch.load('/home/besedin/workspace/Data/MNIST/t7_files/testset.t7')

classifier = nn.Sequential()
if opt.classifier_type == 'conv+lin' then
  classifier:add(nn.SpatialConvolution(1, 8, 2, 2, 1, 1)):add(nn.ReLU(true))
  classifier:add(nn.SpatialConvolution(8, 32, 3, 3, 1, 1)):add(nn.ReLU(true))
  classifier:add(nn.SpatialMaxPooling(3, 3, 3, 3))
  classifier:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1)):add(nn.ReLU(true))
  classifier:add(nn.SpatialConvolution(64, 128, 6, 6)):add(nn.ReLU(true)):add(nn.View(128))
  classifier:add(nn.Linear(128, 32)):add(nn.ReLU(true))
  classifier:add(nn.Linear(32, 10)):add(nn.LogSoftMax())
end

criterion = nn.ClassNLLCriterion()
input = torch.Tensor(opt.batch_size, 1, 28, 28)
labels = torch.Tensor(opt.batch_size)

if opt.cuda==true then
  require 'cunn'; require 'cudnn'
  criterion:cuda()
  classifier:cuda()
  input = input:cuda()
  labels = labels:cuda()
end

local fx = function(x)
  grad_params:zero()
  input:copy(batch.data)
  labels:copy(batch.labels) -- fake labels are real for generator cost
  local output = classifier:forward(input)
  local err = criterion:forward(output, labels)
  local df_do = criterion:backward(output, labels)
  classifier:backward(input, df_do)
  return err, grad_params
end

function get_batch(data, indices)
  local batch = {}
  batch.data = data.data:index(1, indices:long())
  batch.labels = data.labels:index(1, indices:long())
  return batch
end

function test_classifier(C_model, data)
  local confusion = optim.ConfusionMatrix(10)
  confusion:zero()
  for idx = 1, data.data:size(1), opt.batch_size do
    --xlua.progress(idx, data.data:size(1))
    local indices = torch.range(idx, math.min(idx + opt.batch_size-1, data.data:size(1)))
    local batch = get_batch(data, indices:long())
    local y = C_model:forward(batch.data:cuda())
    y = y:float()
    _, y_max = y:max(2)
    confusion:batchAdd(y_max:squeeze():float(), batch.labels:float())
  end
  confusion:updateValids()
  return confusion  
end

function trainset_by_classes(trainset)
  local counter = torch.zeros(10)
  for idx = 1, trainset.data:size(1) do counter[trainset.labels[idx]] = counter[trainset.labels[idx]]+1 end
  local result = {}; for idx = 1, 10 do result[idx] = torch.Tensor(counter[idx], 1, 28, 28) end
  counter:fill(1)
  for idx = 1, trainset.data:size(1) do
    result[trainset.labels[idx]][counter[trainset.labels[idx]]] = trainset.data[idx]
    counter[trainset.labels[idx]] = counter[trainset.labels[idx]] + 1
  end
  return result
end

function from_classes_to_single_t7(trainset_by_class, classes)
  local dataset_size = 0; local last_index = 0; local result = {}
  for idx_class = 1, #classes do dataset_size = dataset_size + trainset_by_class[classes[idx_class]]:size(1) end
  result.data = torch.Tensor(dataset_size, 1, 28, 28)
  result.labels = torch.Tensor(dataset_size)
  for idx_class = 1, #classes do
    result.data[{{last_index + 1, last_index + trainset_by_class[classes[idx_class]]:size(1)},{},{},{}}] = trainset_by_class[classes[idx_class]]
    result.labels[{{last_index + 1, last_index + trainset_by_class[classes[idx_class]]:size(1)}}]:fill(classes[idx_class])
    last_index = last_index + trainset_by_class[classes[idx_class]]:size(1)
  end
  return result
end

function add_confusion_to_results(confusion, results)
  to_plot = {}
  for idx = 1, 10 do 
    results[idx][#results[idx]+1] = confusion.valids[idx]*100
    to_plot[idx] = {'Class '..idx, torch.range(1, #results[idx]), torch.FloatTensor(results[idx])}
  end
  return results, to_plot
end

results = {}; 
to_plot = {}
count_res_point = 0
optim_state = {
  learningRate = opt.learning_rate,
}

trainset_by_class = trainset_by_classes(trainset)

params, grad_params = classifier:getParameters()
local data_size =  trainset.data:size(1)
confusion = test_classifier(classifier, testset); print(confusion)
full_confusions = {confusion}
for idx = 1, 10 do
  results[idx] = {confusion.valids[idx]*100} 
  to_plot[idx] = {'Class '..idx, torch.range(1, #results[idx]), torch.FloatTensor(results[idx])}
end
--gnuplot.plot(to_plot)

for epoch = 1, opt.epoch_number do
  local full_rand_indices = torch.randperm(data_size)
  for idx = 1, data_size, opt.batch_size do
    xlua.progress(idx, data_size)
    batch_indices = full_rand_indices[{{idx, idx + opt.batch_size - 1}}]
    batch = get_batch(trainset, batch_indices)
    --optim_state.t = 1
    optim.adam(fx, params, optim_state)
    if (idx-1)%(50*opt.batch_size)==0 then
      confusion = test_classifier(classifier, testset); print(confusion)
      full_confusions[#full_confusions+1] = confusion 
      results, to_plot = add_confusion_to_results(confusion, results)
--      gnuplot.plot(to_plot)
    end
  end
end

local scenarios = {
  {2, 3, 4, 5, 6, 7, 8, 9, 10},
  {3, 4, 5, 6, 7, 8, 9, 10},
  {4, 5, 6, 7, 8, 9, 10},
  {5, 6, 7, 8, 9, 10},
  {6, 7, 8, 9, 10},
  {7, 8, 9, 10},
  {1, 8, 9, 10},
  {9, 10},
  {10}
}

for scenario_number = 1, #scenarios do
  local classes = scenarios[scenario_number]
  trainset = from_classes_to_single_t7(trainset_by_class, classes)
  data_size =  trainset.data:size(1)
  epoch_number = math.floor(18e+4/data_size)
  for epoch = 1, epoch_number do
    local full_rand_indices = torch.randperm(data_size)
    for idx = 1, data_size - data_size%opt.batch_size, opt.batch_size do
      xlua.progress(idx, data_size)
      batch_indices = full_rand_indices[{{idx, idx + opt.batch_size - 1}}]
      batch = get_batch(trainset, batch_indices)
      --optim_state.t = 1
      optim.adam(fx, params, optim_state)
      if (idx-1)%(50*opt.batch_size)==0 then
        confusion = test_classifier(classifier, testset); print(confusion)
        full_confusions[#full_confusions+1] = confusion 
        results, to_plot = add_confusion_to_results(confusion, results)
--        gnuplot.plot(to_plot)
      end
    end
  end
end

gnuplot.plot(to_plot)
