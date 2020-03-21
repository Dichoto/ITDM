#%%
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
#os.chdir("path")
from MMDLoss import RBFMMD
from BalanceSampler_shuffle import BalancedBatchSampler
from itertools import cycle
#%%
class Net2C(nn.Module):
    def __init__(self):
        super(Net2C, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        embd = F.relu(self.fc1(x))
        x = self.fc2(embd)
        return x, embd
    
class Net5C(nn.Module):
    def __init__(self):
        super(Net5C, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=8, stride=1))
        self.fc = nn.Linear(512, 10)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        embd = out.reshape(out.size(0), -1)
        out = self.fc(embd)
        return out, embd
    

#%%
# Training settings
parser = argparse.ArgumentParser(description='PyTorch FMNIST Example')
parser.add_argument('--batch-size', type=int, default=150, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

trainset = datasets.KMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1903,), (0.3475,))
                   ]))    
testset = datasets.KMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1903,), (0.3475,))
                   ]))
train_loader = torch.utils.data.DataLoader(trainset, sampler=BalancedBatchSampler(trainset),
    batch_size=args.batch_size, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(testset, 
    batch_size=args.test_batch_size, shuffle=False,**kwargs)



#%%
model = Net2C().to(device)
#model = Net5C().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

criterion = nn.CrossEntropyLoss()
match_loss = RBFMMD(sigma=[1], use_est_width=True)

test_acc = list() # testing accuracy
test_ce = list() # testing CE loss value 
train_ce = list () # training CE loss value

for epoch in range(1, args.epochs + 1):
    # Training
    match_loader = torch.utils.data.DataLoader(trainset, sampler=BalancedBatchSampler(trainset),
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    match_iter = iter(match_loader)
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        data, target = data.to(device), target.to(device) # minibatch for cross-entroy
        xmatch, ymatch = match_iter.next() # minibatch for matching loss
        xmatch, ymatch = xmatch.to(device), ymatch.to(device)
        
        output, embd1 = model(data)
        _, embd2 = model(xmatch)
        
        optimizer.zero_grad()
        loss = criterion(output,target)
        #total_loss = loss + match_loss(embd1, embd2) # ours ITDM
        total_loss = loss # simply CE
        total_loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        if batch_idx % 50 == 0:
            train_ce.append(loss.item())
            model.eval()
            correct = 0
            test_loss = 0
            with torch.no_grad():
                for tedata, tetarget in test_loader:
                    tedata, tetarget = tedata.to(device), tetarget.to(device)
                    output, _ = model(tedata)
                    test_loss += criterion(output, tetarget)*tedata.shape[0]
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    correct += pred.eq(tetarget.view_as(pred)).sum().item()
                acc = correct / len(test_loader.dataset)
                test_acc.append(acc)
            test_loss /= len(test_loader.dataset)
            test_ce.append(test_loss.item())
    
    scheduler.step()
    
    train_ce.append(loss.item())
    # Testing after each epoch
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for tedata, tetarget in test_loader:
            tedata, tetarget = tedata.to(device), tetarget.to(device)
            output, _ = model(tedata)
            test_loss += criterion(output, tetarget)*tedata.shape[0] # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(tetarget.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = correct / len(test_loader.dataset)
    test_acc.append(acc)
    test_ce.append(test_loss.item())
    
    
    
#%%
if (args.save_model):
    torch.save(model.state_dict(),"kmnist_cnn.pt")
