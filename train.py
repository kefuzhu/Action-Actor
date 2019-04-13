from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader
from cfg.deeplab_pretrain_a2d import train as train_cfg
from cfg.deeplab_pretrain_a2d import val as val_cfg
from cfg.deeplab_pretrain_a2d import test as test_cfg
from network import net
import time

from tqdm import tqdm
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################
# Inception_v3 pre-trained on ImageNet #
########################################

def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4) # you can make changes

    #######################################################
    # Edit the inception_v3 model pre-trained on ImageNet #
    #######################################################
    # Our parameters
    freeze_layers = True
    n_class = args.num_cls
    # Load the model
    model = torchvision.models.inception_v3(pretrained='imagenet')
    ## Lets freeze the first few layers. This is done in two stages 
    # Stage-1 Freezing all the layers 
    if freeze_layers:
        for i, param in model.named_parameters():
            param.requires_grad = False

    # Edit the auxilary net 
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs, n_class)
    # Edit the primary net
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)
    
    # Stage-2 , Freeze all the layers till "Conv2d_4a_3*3"
    ct = []
    for name, child in model.named_children():
        if "Conv2d_4a_3x3" in ct:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)

    # Push model to GPU
    model.to(device)
    # Define Loss and optimizer
    print("[Using CrossEntropyLoss...]")
    criterion = nn.CrossEntropyLoss()
    print("[Using small learning rate with momentum...]")
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.001, momentum=0.9)
    # Define learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the models

    # Move model to GPU
    model.to(device)
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        t1 = time.time()
        data_size = 0
        for i, data in tqdm(enumerate(data_loader)):

            running_loss = 0.0
            running_corrects = 0

            images = data[0]
            labels = data[1]

            # Resize the image to 299x299? (Method 1)
            from skimage.transform import resize
            num_img = images.shape[0]
            num_channel = images.shape[1]
            desire_size = 299
            img_resize = np.empty([num_img,num_channel,desire_size,desire_size])
            for img in images:
                # Convert from Tensor to numpy
                img_npimg = img.numpy()
                # Resize the image
                img_npimg = resize(img_npimg,(num_channel,desire_size,desire_size))
                # Add the resized channel
                img_resize[i] = img_npimg
            # Convert numpy to Tensor
            images = torch.from_numpy(img_resize)

            # # Resize the image to 299x299? (Method 2)
            # from skimage.transform import resize
            # num_img = images[0]
            # num_channel = images[1]
            # desire_size = 299
            # images = torch.from_numpy(resize(images.numpy(),(num_img,num_channel,desire_size,desire_size)))

            # # mini-batch (Move input to GPU)
            # images = data[0].to(device)
            # labels = data[1].type(torch.FloatTensor).to(device)

            # mini-batch (Move input to GPU)
            images = images.type(torch.FloatTensor).to(device)
            labels = images.type(torch.FloatTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Current batch performance
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

            data_size += 4

            # Log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(model.state_dict(), os.path.join(args.model_path, 'net.ckpt'))

        print('data_size:{}.....'.format(data_size))
        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects / data_size
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        t2 = time.time()
        print('Training times spent: {} minutes'.format((t2 - t1)/60))

########################
# Self-defined Network #
########################

# def main(args):
#     # Create model directory for saving trained models
#     if not os.path.exists(args.model_path):
#         os.makedirs(args.model_path)

#     test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
#     data_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4) # you can make changes


#     # Define model
#     model = net()
#     # Push model to GPU
#     model.to(device)
#     # Define Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     # Define learning rate scheduler
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#     # Train the models
#     total_step = len(data_loader)
#     for epoch in range(args.num_epochs):
#         t1 = time.time()
#         running_loss = 0.0
#         running_corrects = 0
#         data_size = 0
#         for i, data in enumerate(data_loader):

#             # mini-batch
#             images = data[0].to(device)
#             labels = data[1].type(torch.FloatTensor).to(device)

#             # zero the parameter gradients 
#             optimizer.zero_grad()

#             # Forward, backward and optimize
#             outputs = model(images)

#             # print('outputs:{}'.format(outputs.type()))
#             # print('labels:{}'.format(labels.type()))

#             # print('outputs:{}'.format(outputs.long()))
#             # print('labels:{}'.format(labels.long()))

#             labels = torch.max(labels.long(), 1)[1]
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # Current batch prediction
#             _, preds = torch.max(outputs.data, 1)
#             # Current batch performance
#             running_loss += loss.data.item()
#             running_corrects += torch.sum(preds == labels.data)

#             data_size += 4


#             # Log info
#             if i % args.log_step == 0:
#                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                       .format(epoch, args.num_epochs, i, total_step, loss.item()))

#             # Save the model checkpoints
#             if (i + 1) % args.save_step == 0:
#                 torch.save(model.state_dict(), os.path.join(
#                     args.model_path, 'net.ckpt'))

#         print('data_size:{}.....'.format(data_size))
#         epoch_loss = running_loss / data_size
#         epoch_acc = running_corrects / data_size

#         print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

#         t2 = time.time()
#         print('Training times spent: {} minutes'.format((t2 - t1)/60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
main(args)