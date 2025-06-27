"""
The credit for this code belongs to the following github repository:
    https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training

This repo is an official implementation of the paper: Geometry-Aware Instance-Reweighted Adversarial Training
by Zhang et. al. (2021). We have used their adversarial attack scheme to directly test and build our other implementations
upon it. 
"""
import os
import sys
from logger import Logger

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split

from GAIR import GAIR
import attack_generator as attack
from gairat_config import GAIRATconfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.load_model import ModelZoo
from dataset.dataset import TaskDataset, SubsetWithTransform

import numpy as np


seed = GAIRATconfig.SEED
momentum = GAIRATconfig.MOMENTUM
weight_decay = GAIRATconfig.WEIGHT_DECAY
resume = GAIRATconfig.RESUME
out_dir = GAIRATconfig.OUT_DIR


torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

model = ModelZoo(GAIRATconfig.MODEL, GAIRATconfig.NUM_CLASSES).load_model().cuda()

model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=GAIRATconfig.LR_MAX, momentum=momentum, weight_decay=weight_decay)


# Learning schedules
if GAIRATconfig.LR_SCHEDULER == 'superconverge':
    lr_schedule = lambda t: np.interp([t], [0, GAIRATconfig.EPOCHS * 2 // 5, GAIRATconfig.EPOCHS], [0, GAIRATconfig.LR_MAX, 0])[0]
elif GAIRATconfig.LR_SCHEDULER == 'piecewise':
    def lr_schedule(t):
        if GAIRATconfig.EPOCHS >= 110:
            # Train Wide-ResNet
            if t / GAIRATconfig.EPOCHS < 0.5:
                return GAIRATconfig.LR_MAX
            elif t / GAIRATconfig.EPOCHS < 0.75:
                return GAIRATconfig.LR_MAX / 10.
            elif t / GAIRATconfig.EPOCHS < (11/12):
                return GAIRATconfig.LR_MAX / 100.
            else:
                return GAIRATconfig.LR_MAX / 200.
        else:
            # Train ResNet
            if t / GAIRATconfig.EPOCHS < 0.3:
                return GAIRATconfig.LR_MAX
            elif t / GAIRATconfig.EPOCHS < 0.6:
                return GAIRATconfig.LR_MAX / 10.
            else:
                return GAIRATconfig.LR_MAX / 100.
elif GAIRATconfig.LR_SCHEDULER == 'linear':
    lr_schedule = lambda t: np.interp([t], [0, GAIRATconfig.EPOCHS // 3, GAIRATconfig.EPOCHS * 2 // 3, GAIRATconfig.EPOCHS], [GAIRATconfig.LR_MAX, GAIRATconfig.LR_MAX, GAIRATconfig.LR_MAX / 10, GAIRATconfig.LR_MAX / 100])[0]
elif GAIRATconfig.LR_SCHEDULER == 'onedrop':
    def lr_schedule(t):
        if t < GAIRATconfig.LR_DROP_EPOCH:
            return GAIRATconfig.LR_MAX
        else:
            return GAIRATconfig.LR_ONE_DROP
elif GAIRATconfig.LR_SCHEDULER == 'multipledecay':
    def lr_schedule(t):
        return GAIRATconfig.LR_MAX - (t//(GAIRATconfig.EPOCHS//10))*(GAIRATconfig.LR_MAX/10)
elif GAIRATconfig.LR_SCHEDULER == 'cosine': 
    def lr_schedule(t): 
        return GAIRATconfig.LR_MAX * 0.5 * (1 + np.cos(t / GAIRATconfig.EPOCHS * np.pi))
    

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def train(epoch, model, train_loader, optimizer, Lambda):
    
    lr = 0
    num_data = 0
    train_robust_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        loss = 0
        data, target = data.cuda(), target.cuda()
        
        # Get adversarial data and geometry value
        x_adv, Kappa = attack.GA_PGD(model,data,target,GAIRATconfig.EPSILON,GAIRATconfig.STEP_SIZE,GAIRATconfig.NUM_STEPS,loss_fn="cent",category="Madry",rand_init=True)

        model.train()
        lr = lr_schedule(epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()
        
        logit = model(x_adv)

        if (epoch + 1) >= GAIRATconfig.BEGIN_EPOCH:
            Kappa = Kappa.cuda()
            loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
            # Calculate weight assignment according to geometry value
            normalized_reweight = GAIR(GAIRATconfig.NUM_STEPS, Kappa, Lambda, GAIRATconfig.WEIGHT_ASSIGNMENT_FUNCTION)
            loss = loss.mul(normalized_reweight).mean()
        else:
            loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
        
        train_robust_loss += loss.item() * len(x_adv)
        
        loss.backward()
        optimizer.step()
        
        num_data += len(data)

    train_robust_loss = train_robust_loss / num_data

    return train_robust_loss, lr


def adjust_Lambda(epoch):
    Lam = float(GAIRATconfig.LAMBDA)
    if GAIRATconfig.EPOCHS >= 110:
        # Train Wide-ResNet
        Lambda = GAIRATconfig.LAMBDA_MAX
        if GAIRATconfig.LAMBDA_SCHEDULE == 'linear':
            if epoch >= 60:
                Lambda = GAIRATconfig.LAMBDA_MAX - (epoch/GAIRATconfig.EPOCHS) * (GAIRATconfig.LAMBDA_MAX - Lam)
        elif GAIRATconfig.LAMBDA_SCHEDULE == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam-1.0
            elif epoch >= 110:
                Lambda = Lam-1.5
        elif GAIRATconfig.LAMBDA_SCHEDULE == 'fixed':
            if epoch >= 60:
                Lambda = Lam
    else:
        # Train ResNet
        Lambda = GAIRATconfig.LAMBDA_MAX
        if GAIRATconfig.LAMBDA_SCHEDULE == 'linear':
            if epoch >= 30:
                Lambda = GAIRATconfig.LAMBDA_MAX - (epoch/GAIRATconfig.EPOCHS) * (GAIRATconfig.LAMBDA_MAX - Lam)
        elif GAIRATconfig.LAMBDA_SCHEDULE == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam-2.0
        elif GAIRATconfig.LAMBDA_SCHEDULE == 'fixed':
            if epoch >= 30:
                Lambda = Lam
    return Lambda

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if GAIRATconfig.DATASET == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=GAIRATconfig.BATCH_SIZE, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=GAIRATconfig.BATCH_SIZE, shuffle=False, num_workers=0)

if GAIRATconfig.DATASET == "task":
    trainset = torch.load("./data/Train.pt", weights_only=False)
    trainset, testset = random_split(trainset, [80000, 20000])

    trainset_transformed = SubsetWithTransform(trainset, transform=transform_train)
    testset_transformed = SubsetWithTransform(testset, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset_transformed, batch_size=GAIRATconfig.BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset_transformed, batch_size=GAIRATconfig.BATCH_SIZE, shuffle=False, num_workers=0)

title = 'GAIRAT'
best_acc = 0
start_epoch = 0
if resume:
    # Resume directly point to checkpoint.pth.tar
    print ('==> GAIRAT Resuming from checkpoint ..')
    print(resume)
    assert os.path.isfile(resume)
    out_dir = os.path.dirname(resume)
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['test_pgd20_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
else:
    print('==> GAIRAT')
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])


## Training get started
test_nat_acc = 0
test_pgd20_acc = 0

for epoch in range(start_epoch, GAIRATconfig.EPOCHS):
    
    # Get lambda
    Lambda = adjust_Lambda(epoch + 1)
    
    # Adversarial training
    train_robust_loss, lr = train(epoch, model, train_loader, optimizer, Lambda)

    # Evalutions similar to DAT.
    _, test_nat_acc = attack.eval_clean(model, test_loader)
    _, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", random=True)


    print(
        'Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.2f | PGD20 Test Acc %.2f |\n' % (
        epoch,
        GAIRATconfig.EPOCHS,
        lr,
        test_nat_acc,
        test_pgd20_acc)
        )
         
    logger_test.append([epoch + 1, test_nat_acc, test_pgd20_acc])
    
    # Save the best checkpoint
    if test_pgd20_acc > best_acc:
        best_acc = test_pgd20_acc
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer' : optimizer.state_dict(),
            },filename='bestpoint.pth.tar')

    # Save the last checkpoint
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer' : optimizer.state_dict(),
            })
    
logger_test.close()