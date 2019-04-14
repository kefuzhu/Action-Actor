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

# Extract the learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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
    print("[Using BCEWithLogitsLoss...]")
    criterion = nn.BCEWithLogitsLoss()
    print("[Using small learning rate with momentum...]")

    ##############
    # Parameters #
    ##############
    # Number of steps per iteration
    total_step = len(data_loader)
    # Data size (Each step train on a batch of 4 images)
    data_size = total_step*4
    # Learning rate
    if args.lr:
        learning_rate = args.lr # 0.05
    else:
        raise(ValueError('Please provide learning rate'))
    # Decay rate of learning rate
    if args.lr_decay:
        lr_decay = args.lr_decay # 5
    else:
        raise(ValueError('Please provide rate of decay for learning rate'))
    # Number of times learning rate decay
    if args.lr_changes:
        lr_step = args.num_epochs//args.lr_changes
    else:
        raise(ValueError('Please provide number of decay times for learning rate'))
    # Define optimizer
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)
    # Define learning rate scheduler
    my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
    ###################
    # Train the model #
    ###################
    # File name to save model
    model_name = ''.join(['net_','epoch-',str(args.num_epochs),'_lr-',str(learning_rate),'_decay-',str(lr_decay),'_step-',str(lr_step),'.ckpt'])
    # Move model to GPU
    model.to(device)
    # Start time
    start = time.time()
    # Log file name
    log_file = ''.join(['train_log/net_','epoch-',str(args.num_epochs),'_lr-',str(learning_rate),'_decay-',str(lr_decay),'_step-',str(lr_step),'.txt'])
    # Check if file already exists
    if os.path.isfile(log_file):
        raise(Exception('File already exists'))
    # If file not exists
    else:
        # Create log file
        with open(log_file, 'w') as f:
            for epoch in range(args.num_epochs):
                # Change of learning rate
                my_lr_scheduler.step()
                print('-'*50)
                print('Current learning rate:{}\n'.format(get_lr(optimizer)))
                # Write log to file
                f.write('-'*50)
                f.write('\n')
                f.write('Current learning rate:{}\n'.format(get_lr(optimizer)))

                t1 = time.time()

                running_loss = 0.0
                running_corrects = 0
                for i, data in enumerate(data_loader):

                    # mini-batch (Move input to GPU)
                    images = data[0].type(torch.FloatTensor).to(device)
                    # labels = data[1].type(torch.int64).to(device)
                    labels = data[1].type(torch.FloatTensor).to(device)
                    
                    # Forward, backward and optimize
                    outputs,aux = model(images)
                    # labels = torch.max(labels.long(), 1)[1]
                    _, preds = torch.max(outputs.data, 1)
                    # Compute the loss = primay net loss + 0.3 * auxilary net loss
                    loss = criterion(outputs, labels) + 0.3*criterion(aux,labels)
                    # Backprop the loss to the network (Compute the gradient of loss w.r.t parameters with require_grad = True)
                    loss.backward()
                    # Update the parameters within network
                    optimizer.step()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Current batch performance
                    running_loss += loss
                    labels = torch.max(labels.long(), 1)[1]
                    running_corrects += torch.sum(preds == labels.data)
                    # Log info
                    if i % args.log_step == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item()))
                        # Write log to file
                        f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'.format(epoch, args.num_epochs, i, total_step, loss.item()))

                    # Save the model checkpoints
                    if (i + 1) % args.save_step == 0:
                        torch.save(model.state_dict(), os.path.join(args.model_path, model_name))

                epoch_loss = running_loss / data_size
                epoch_acc = running_corrects.item() / data_size # running_corrects is a torch.Tensor(), need .item() to extract value
                print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                # Write log to file
                f.write('Loss: {:.4f} Acc: {:.4f}\n'.format(epoch_loss, epoch_acc))

                t2 = time.time()
                print('Current epoch training time: {} minutes'.format((t2 - t1)/60))
                # Write log to file
                f.write('Current epoch training time: {} minutes\n'.format((t2 - t1)/60))

    # End time
    end = time.time()
    torch.save(model.state_dict(), os.path.join(args.model_path, model_name))
    print('Total training times spent: {} minutes'.format((end - start)/60))
    # Write log to file
    f.write('Total training times spent: {} minutes'.format((end - start)/60))
    # Close the log file
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=None, help='learning rate for training model')
    parser.add_argument('--lr_decay', type=float, default=None, help='rate of decay for learning rate')
    parser.add_argument('--lr_changes', type=int, default=None, help='number of decay times for learning rate')
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
