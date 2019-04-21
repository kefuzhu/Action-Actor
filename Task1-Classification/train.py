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
from utils.eval_metrics import Precision, Recall, F1

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract the learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def num_correct(preds,labels):
    '''
    Given two input torch.Tensor, compute the number of correct prediction
    '''
    if len(preds) != len(labels):
        raise(ValueError('Predictions and labels need to have same length'))
    # Initialize the counter for the number of correct prediction
    correct = 0
    # For each prediction
    for i in range(len(preds)):
        # If the prediction result is correct at every entry with the label
        # e.g.:
        # Prediction is correct if:
        #      prediction = [1,0,0,1,0]
        #      label = [1,0,0,1,0]
        # Prediction is NOT correct if:
        #      prediction = [0,0,0,1,0]
        #      label = [1,0,0,1,0]
        if torch.all(torch.eq(preds[i],labels[i])) > 0:
            # Increment the number of correct
            correct += 1

    return correct

########################################
# Inception_v3 pre-trained on ImageNet #
########################################

def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    test_dataset = a2d_dataset.A2DDataset(train_cfg, args.dataset_path)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4) # you can make changes

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
    # Data size
    data_size = total_step*args.batch_size
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
    model_name = ''.join(['net_','batchsize-',str(args.batch_size),'_epoch-',str(args.num_epochs),'_lr-',str(learning_rate),'_decay-',str(lr_decay),'_step-',str(lr_step),'.ckpt'])
    # Move model to GPU
    model.to(device)
    # Start time
    start = time.time()
    # Log file name
    log_file = ''.join(['train_log/net_','batchsize-',str(args.batch_size),'_epoch-',str(args.num_epochs),'_lr-',str(learning_rate),'_decay-',str(lr_decay),'_step-',str(lr_step),'.txt'])
    # Check if file already exists
    if os.path.isfile(log_file):
        raise(Exception('File already exists'))
    # If file not exists
    else:
        # Create log file
        with open(log_file, 'w') as f:
            # Write the configuration to log file
            f.write(str(args))
            try:
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
                        # Compute the loss = primay net loss + 0.3 * auxilary net loss
                        loss = criterion(outputs, labels) + 0.3*criterion(aux,labels)
                        # Backprop the loss to the network (Compute the gradient of loss w.r.t parameters with require_grad = True)
                        loss.backward()
                        # Update the parameters within network
                        optimizer.step()
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # Calculate the prediction from probability
                        preds = outputs.cpu().detach().numpy()
                        for pred in preds:
                            pred[pred >= 0.5] = 1
                            pred[pred < 0.5] = 0
                        # Convert it back to tensor type
                        preds = torch.Tensor(preds)
                        # Grab labels from GPU to CPU
                        labels = labels.cpu().detach()
                        # Current batch performance
                        running_loss += loss
                        # Calculate number of correct prediction for current batch
                        batch_correct = num_correct(preds,labels) 
                        # Add the correct prediction                       
                        running_corrects += batch_correct

                        if batch_correct > args.batch_size:
                            print('preds:{}'.format(preds.numpy()))
                            print('labels:{}'.format(labels.numpy()))
                            print('batch_correct:{}'.format(batch_correct))
                            raise(ValueError('WTF DUDE!'))

                        # running_corrects += torch.sum(preds == labels.data)                        # Log info
                        if i % args.log_step == 0:
                            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item()))
                            # Write log to file
                            f.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'.format(epoch, args.num_epochs, i, total_step, loss.item()))

                    # Compute the loss and correct number of prediction for current epoch
                    epoch_loss = running_loss / data_size
                    epoch_acc = running_corrects / data_size

                    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))                    
                    # Write log to file
                    f.write('Loss: {:.4f} Acc: {:.4f}\n'.format(epoch_loss, epoch_acc))

                    t2 = time.time()
                    print('Current epoch training time: {} minutes'.format((t2 - t1)/60))
                    # Write log to file
                    f.write('Current epoch training time: {} minutes\n'.format((t2 - t1)/60))
                    # Save model
                    torch.save(model.state_dict(), os.path.join(args.model_path, model_name))

                # End time
                end = time.time()
                print('Total training times spent: {} minutes'.format((end - start)/60))
                # Write log to file
                f.write('Total training times spent: {} minutes'.format((end - start)/60))


                # Close the log file
                f.close()
            except(RuntimeError,KeyboardInterrupt) as err:
                # Save model
                print('Save ckpt on exception ...')
                f.write('Save ckpt on exception ...')
                torch.save(model.state_dict(), os.path.join(args.model_path, model_name))
                print('Save ckpt done.')
                f.write('Save ckpt done.')
                # End time
                end = time.time()
                print('Total training times spent: {} minutes'.format((end - start)/60))
                # Write log to file
                f.write('Total training times spent: {} minutes'.format((end - start)/60))
                # Close the log file
                f.close()
                # Raise error message
                raise(err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate for training model')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='rate of decay for learning rate')
    parser.add_argument('--lr_changes', type=int, default=1, help='number of decay times for learning rate')
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=43)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
main(args)

# epoch_num = 100
# lr = 0.05
# lr_changes = 7
# step = epoch_num//lr_changes
# for i in range(lr_changes+1):
#     print('Epoch:{}.lr:{}'.format(i*step,lr))
#     lr = lr*0.5
    

