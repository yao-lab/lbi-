from __future__ import print_function
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy
import numpy as np

#from models import *
from resnet_conv import ResNet18
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

# args = parser.parse_args()

__input_dir__ = "./"
__output_dir__ = "./small_model/"
if not os.path.isdir(__output_dir__):
    os.mkdir(__output_dir__)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root=r'./data', train=False, download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = ResNet18()

net = net.to(device)

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir(os.path.join(__input_dir__, 'checkpoint')), 'Error: no checkpoint directory found!'
checkpoint = torch.load(os.path.join(__input_dir__, 'checkpoint/ckpt_lbi_group_resume.t7'))
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss(size_average=True)


def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    return acc


def quantile(w, percent):
    with torch.no_grad():
        p = np.percentile(w.cpu().detach().numpy().reshape((1, -1)), percent)
        return torch.Tensor([p]).to(device)


def get_sparse_group(param, ts_norm, threshold):
    with torch.no_grad():
        W_sparse = copy.deepcopy(param.data)
        W_sparse[ts_norm < threshold, :, :, :] = 0
        # print("# selected filters", sum((ts_norm >= threshold).cpu().detach().numpy()))
        W_sparse.to(device)
        return W_sparse


def get_sparse(param, threhold):
    with torch.no_grad():
        W_sparse = copy.deepcopy(param.data)
        W_sparse[param < threhold] = 0
        # print("# selected units", (param >= threhold).sum().item())
        W_sparse.to(device)
        return W_sparse


# before pruning
accuracy = test()
print("Accuracy before pruning: {}".format(accuracy))

# slbi layer setting
slbi_param_names = [
    'layer3.0.conv1.weight',
    'layer3.0.conv2.weight',
    'layer3.1.conv1.weight',
    'layer3.1.conv2.weight',
    'layer4.0.conv1.weight',
    'layer4.0.conv2.weight',
    'layer4.1.conv1.weight',
    'layer4.1.conv2.weight',
]

# the SLBI_Param classes
slbi_params = dict()
q = 98.4375
print('-'*40)
print("Pruning rate: {}%".format(q))

total_num_weights = 0
pruned_num_weights = 0
params = []
for name, param in net.named_parameters():
    # print(name, param.numel())
    total_num_weights += param.numel()
    if name in slbi_param_names:
        # pruning
        pruned_num_weights += q / 100 * param.numel()
        print("Start pruning layer: {}".format(name))
        ts = param.data
        if ts.dim() == 4:  # CNN
            ts_reshape = torch.reshape(ts,(ts.shape[0],-1))
            ts_norm = torch.norm(ts_reshape,2,1)
            p = quantile(ts_norm, q)
            # this is ok, because W_star will be reconstricted during the next training step
            param.data = get_sparse_group(param, ts_norm, p)
        elif ts.dim() == 2:  # linear
            p = quantile(ts, q)
            param.data = get_sparse(param, p)
        else:
            assert 0, "ERROR: layer type is not CNN nor linear!"

        continue
print("Pruning ended.")
print('-'*40)
comp_rate = (1 - pruned_num_weights / total_num_weights) * 100
print("Total compression rate (#remain / #total): %0.2f" % comp_rate, '%')

# before pruning
accuracy = test()
print("Accuracy after pruning: {}".format(accuracy))

# Saving pruned model
print('Saving..')
state = {
    'net': net.state_dict(),
    'slbi_param_names': slbi_param_names,
}
torch.save(state, os.path.join(__output_dir__, 'small_model_slbi.t7'))

