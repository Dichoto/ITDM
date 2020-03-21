#%%
from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
# os.chdir("path")
from MMDLoss import RBFMMD
from BalanceSampler_shuffle import BalancedBatchSampler
from itertools import cycle
from models.vgg import VGG
import pandas as pd

#%%
# Training settings
parser = argparse.ArgumentParser(description='PyTorch STL Example')
parser.add_argument('--batch-size', type=int, default=150, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--save-path', type=str, default='trained/match_1_vgg_stl.pt',
                    help='For Saving the current Model')
parser.add_argument('--load-model', action='store_true', default=False,
                    help='If load trained model')

parser.add_argument('--model-path', type=str, default='trained/match_1_vgg_stl.pt',
                    help='Load model from path')

parser.add_argument('--match', action='store_true', default=True,
                    help='use the match reg')

parser.add_argument('--data', type=str, default='stl',
                    help='specify the dataset')


args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 24, 'pin_memory': True} if use_cuda else {}

if args.data == 'svhn':
    trainset = datasets.SVHN('../data', split='train', download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) 
                       ]))    
    testset = datasets.SVHN('../data', split='test',download=False, 
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) 
                       ]))
elif args.data == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

elif args.data == 'stl':
    trainset = datasets.STL10('../data', split='train', download=True,
                       transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) 
                       ]))    
    testset = datasets.STL10('../data', split='test',download=True, 
                            transform=transforms.Compose([
                            transforms.Resize(32),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) 
                       ]))



train_loader = torch.utils.data.DataLoader(trainset, sampler=BalancedBatchSampler(trainset),
    batch_size=args.batch_size, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(testset, 
    batch_size=args.test_batch_size, shuffle=True,**kwargs)



#%%
model = VGG('VGG13').to(device)
if args.load_model:
# =============================================================================
#     chk = torch.load(args.model_path)
# =============================================================================
    model = torch.load(args.model_path)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    train_acc=list()
    test_acc=list()
    train_loss_list = list()
    test_loss_list = list()
    
    criterion = nn.CrossEntropyLoss()
    match_loss = RBFMMD(sigma=[1,2,4,8,16], use_est_width=True)
    lbda = 0.8
    
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
            
            if args.match:
                m_loss = 0
                label_set = target.unique()
                for label in label_set:
                    label_loc = (target==label)
                    embd1_slice = embd1[label_loc]
                    embd2_slice = embd2[label_loc]
                    m_loss += match_loss(embd1_slice, embd2_slice)
                
                total_loss = loss + lbda * m_loss/10 # 10 classes in minibatch
            else:
                total_loss = loss
            total_loss.backward()
            optimizer.step()
    
            
            if batch_idx % 5 == 0:
                model.eval()
                
                train_pred = output.argmax(dim=1, keepdim=True)
                train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
                train_accuracy = train_correct / len(data)
                
                correct = 0
                test_loss = 0
                with torch.no_grad():
                    for tedata, tetarget in test_loader:
                        tedata, tetarget = tedata.to(device), tetarget.to(device)
                        output, _ = model(tedata)
                        test_loss += criterion(output, tetarget)*tedata.shape[0]
                        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                        correct += pred.eq(tetarget.view_as(pred)).sum().item()
                        
                    test_loss /= len(test_loader.dataset)
                    acc = correct / len(test_loader.dataset)
                    test_acc.append(acc)
                    train_loss_list.append(loss.item())
                    test_loss_list.append(test_loss.item())
                    train_acc.append(train_accuracy)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tTest Loss: {:.6f}\tTrain Accuracy: {:6f}\tAcc: {:.6f}'.format(
                          epoch, batch_idx * len(data), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item(), test_loss.item(), train_accuracy, acc))
    
        scheduler.step()
        # Testing
        model.eval()
        
        train_pred = output.argmax(dim=1, keepdim=True)
        train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
        train_accuracy = train_correct / len(data)
        
        
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for tedata, target in test_loader:
                tedata, target = tedata.to(device), target.to(device)
                output, _ = model(tedata)
                test_loss += criterion(output, target)*tedata.shape[0] # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        acc = correct / len(test_loader.dataset)
        test_acc.append(acc)
        train_loss_list.append(loss.item())
        test_loss_list.append(test_loss.item())
        train_acc.append(train_accuracy)
        
        # save to file # save at each epoch
        df_dict = {'trainloss': train_loss_list, 'test loss': test_loss_list, 
                   'trainacc': train_acc, 'testacc': test_acc}
        df = pd.DataFrame(df_dict)
        df.to_csv(args.save_path + '.csv')
#%%
model.eval()
test_loss = 0
correct = 0
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for tedata, target in test_loader:
        tedata, target = tedata.to(device), target.to(device)
        output, _ = model(tedata)
        test_loss += criterion(output, target)*tedata.shape[0] # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
acc = correct / len(test_loader.dataset)

    
if (args.save_model):
    torch.save(model,args.save_path)    
    