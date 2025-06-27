"""
The credit for this code belongs to the following github repository:
    https://github.com/hendrycks/pre-training

This repo is an official implementation of the paper: Using Pre-Training Can Improve Model Robustness and Uncertainty
by Hendrycks et. al. (2019). We have used their pretraining scheme to directly test and build our other implementations
upon it. 
"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

import os
import sys
import time
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.load_model import ModelZoo

import attacks
from imagenet_downsampled import ImageNetDS
from pretrain_config import PretrainConfig

torch.manual_seed(1)
np.random.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

state = {key: value for key, value in vars(PretrainConfig).items() if not key.startswith('__')}
print(state)

train_transform = trn.Compose([trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor()])
test_transform = trn.ToTensor()

train_data = ImageNetDS('./data/', 32, train=True, transform=train_transform)
test_data = ImageNetDS('./data/', 32, train=False, transform=test_transform)


train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=PretrainConfig.BATCH_SIZE, shuffle=True,
    num_workers=PretrainConfig.NUM_WORKERS, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=PretrainConfig.TEST_BS, shuffle=False,
    num_workers=PretrainConfig.NUM_WORKERS, pin_memory=True)

net = ModelZoo(model_name=PretrainConfig.MODEL, num_classes=PretrainConfig.NUM_CLASSES).load_model()

start_epoch = 0

# Restore model if desired
if PretrainConfig.LOAD != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(PretrainConfig.LOAD, PretrainConfig.DATASET + PretrainConfig.MODEL +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

if PretrainConfig.NUM_GPU > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(PretrainConfig.NUM_GPU)))

# if PretrainConfig.NUM_GPU > 0:
#     net.to(device)
#     torch.cuda.manual_seed(1)
net.to(device)
cudnn.benchmark = True  

if PretrainConfig.OPTIMIZER == "sgd":
    optimizer = torch.optim.SGD(
        net.parameters(), PretrainConfig.LR, momentum=PretrainConfig.MOMENTUM,
        weight_decay=PretrainConfig.DECAY, nesterov=True
    )

else:
    optimizer = torch.optim.Adam(
        net.parameters(), lr=PretrainConfig.LR,
        weight_decay=PretrainConfig.DECAY
    )

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        PretrainConfig.EPOCHS * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / PretrainConfig.LR))


adversary = attacks.PGD_linf(epsilon=8./255, num_steps=10, step_size=2./255).cuda()

# /////////////// Training ///////////////

def train():
    net.train()
    loss_avg = 0.0
    for bx, by in tqdm(train_loader):
        bx, by = bx.to(device), by.to(device)
        adv_bx = adversary(net, bx, by)
        logits = net(adv_bx * 2 - 1)  
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, by)
        loss.backward()
        optimizer.step()
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
    state['train_loss'] = loss_avg
    scheduler.step()

def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  #
            output = net(data * 2 - 1)
            loss = F.cross_entropy(output, target)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            loss_avg += float(loss.data)
    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if PretrainConfig.TEST:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(PretrainConfig.SAVE_PATH):
    os.makedirs(PretrainConfig.SAVE_PATH)
if not os.path.isdir(PretrainConfig.SAVE_PATH):
    raise Exception('%s is not a dir' % PretrainConfig.SAVE_PATH)

with open(os.path.join(PretrainConfig.SAVE_PATH, PretrainConfig.DATASET + PretrainConfig.MODEL +
                                  '_baseline_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

print('Beginning Training\n')

for epoch in range(start_epoch, PretrainConfig.EPOCHS):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    torch.save(net.state_dict(),
               os.path.join(PretrainConfig.SAVE_PATH, PretrainConfig.DATASET + PretrainConfig.MODEL +
                            '_baseline_epoch_' + str(epoch) + '.pt'))

    prev_path = os.path.join(PretrainConfig.SAVE_PATH, PretrainConfig.DATASET + PretrainConfig.MODEL +
                             '_baseline_epoch_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    with open(os.path.join(PretrainConfig.SAVE_PATH, PretrainConfig.DATASET + PretrainConfig.MODEL +
                                      '_baseline_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )