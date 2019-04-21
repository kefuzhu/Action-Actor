# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
# from Cityscapes_loader import CityscapesDataset
# from CamVid_loader import CamVidDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os

from loader.a2d_dataset import A2DDataset
from cfg.cfg_a2d import train as train_cfg
import torch.nn.functional as F
import argparse

import pdb
import torchfcn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_parameters(model,bias=False):
    modules_skipped = (
            nn.ReLU,
            nn.MaxPool2d,
            nn.Dropout2d,
            nn.Sequential,
            torchfcn.models.FCN32s
            )
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight

        elif isinstance(m,nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m,nn.ConvTranspose2d):
            if bias:
                assert m.bias is None
        elif isinstance(m,modules_skipped):
            continue
        else:
            raise ValueError('Unexpected modules: %s' %str(m))

def cross_entropy2d(input,target,weight=None,size_average=False):
    n,c,h,w = input.size()
    log_p = F.log_softmax(input,dim=1)

    log_p = log_p.transpose(1,2).transpose(2,3).contiguous()
    log_p = log_p[target.view(n,h,w,1).repeat(1,1,1,c) >=0]
    log_p = log_p.view(-1,c)

    mask = target>=0
    target = target[mask]
    loss = F.nll_loss(log_p,target,weight=weight,reduction='sum')
    if size_average:
        loss/=mask.data_sum()
    return loss

# Extract the learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if not os.path.exists('train_log/'):
        os.makedirs('train_log/')

    train_dataset = A2DDataset(train_cfg)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers = args.num_workers)

    # vgg_model = VGGNet(requires_grad=False, remove_fc=True)
    # fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)


    # if use_gpu:
    #     vgg_model = vgg_model.cuda()
    #     fcn_model = fcn_model.cuda()
    #     fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    #     print("Finish cuda loading...")

    model = torchfcn.models.FCN32s(n_class=args.num_cls).to(device)

    if os.path.exists(args.model_path+'net.ckpt'):
        model.load_state_dict(torch.load(os.path.join(args.model_path,'net.ckpt')))
        print('Load model parameters from previous training...')


    optimizer = torch.optim.SGD(
            [
                {'params':get_parameters(model,bias=False)},
                {'params':get_parameters(model,bias=True),
                'lr':1e-10*2, 'weight_decay':0},
            ],
            lr=1e-10,
            momentum=0.9,
            weight_decay = 0.00005)

    scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma =0.1)


    total_step = len(train_loader)
    data_size = total_step * args.batch_size

    start = time.time()    
    
    print('Start training...')
    log_file = 'train_log/log.txt'
    # Check if file already exists
    if os.path.isfile(log_file):
        raise(Exception('File already exists'))

    # Create log file
    with open(log_file, 'w') as f:
        try:
            for epoch in range(args.num_epochs):
                # Change of learning rate
                scheduler.step()
                print('-'*50)
                print('Current learning rate:{}\n'.format(get_lr(optimizer)))
                # Write log to file
                f.write('-'*50)
                f.write('\n')
                f.write('Current learning rate:{}\n'.format(get_lr(optimizer)))

                ts = time.time()

                running_loss = 0.0

                # pdb.set_trace()
                for i, batch_data in enumerate(train_loader):            
                    imgs = batch_data[0].to(device)
                    labels = batch_data[1].to(device)
                    # Feed forward
                    outputs = model(imgs)
                    # Calculate the loss
                    loss = cross_entropy2d(outputs,labels)
                    running_loss += loss
                    loss.backward()
                    optimizer.step()
                    # Zero the gradients
                    optimizer.zero_grad()
                    # pdb.set_trace()

                    if i % (total_step//10) == 0:
                        # pdb.set_trace()
                        print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch, args.num_epochs, i, total_step, loss.item()))
                        f.write("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n".format(epoch, args.num_epochs, i, total_step, loss.item()))
                
                print("Finished epoch {}, training time: {:.2f}, Loss:{:.4f}".format(epoch, (time.time() - ts)/60, running_loss.item()/data_size))
                f.write("Finished epoch {}, training time: {:.2f}, Loss:{:.4f}\n".format(epoch, (time.time() - ts)/60, running_loss.item()/data_size))
                torch.save(model.state_dict(), os.path.join(args.model_path, 'net.ckpt'))

            print('Finished training. Total training times spent: {:.2f} minutes'.format((time.time() - start)/60))
            f.write('Finished training. Total training times spent: {:.2f} minutes\n'.format((time.time() - start)/60))
            # Close the log file
            f.close()

        except(RuntimeError,BrokenPipeError,KeyboardInterrupt) as err:
            # Save model
            print('\nSave ckpt on exception ...\n')
            f.write('Save ckpt on exception ...\n')
            torch.save(model.state_dict(), os.path.join(args.model_path, 'net.ckpt'))
            print('Finished training. Total training times spent: {:.2f} minutes'.format((time.time() - start)/60))
            f.write('Finished training. Total training times spent: {:.2f} minutes\n'.format((time.time() - start)/60))
            # Close the log file
            f.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=44)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
main(args)

