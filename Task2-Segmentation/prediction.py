from loader import a2d_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU ID
from torch.utils.data import Dataset, DataLoader

from cfg.cfg_a2d import train as train_cfg
from cfg.cfg_a2d import val as val_cfg
from cfg.cfg_a2d import test as test_cfg
import pickle
import time

import torchfcn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(args):
    
    val_dataset = a2d_dataset.A2DDataset(val_cfg)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=1)

    print('Loading model...')
    model = torchfcn.models.FCN32s(args.num_cls).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_path,'net.ckpt')))
    print('Model loaded :D')
    gt_list = []
    mask_list = []

    # Total number of steps per epoch
    total_step = len(val_loader)

    # model.eval()
    print('Predicting class label on pixel level...')
    start = time.time()
    with torch.no_grad():
        for i,data in enumerate(val_loader):

            images = data[0].to(device)
            gt = data[1].to(device)#[224,224]
            output = model(images)
            mask = output.data.max(1)[1].cpu().numpy()[:,:,:]
            mask_list.append(mask)
            gt_list.append(gt.cpu().numpy())
            # Print log every 1/10 of the total step
            if i % (total_step//10) == 0:
                print("Step [{}/{}]".format(i, total_step))

    print('Prediction took {} minutes'.format((time.time()-start)/60))

    with open('models/eval_mask_pred.pkl', 'wb') as f:
        print('Dumping mask file...')
        pickle.dump(mask_list,f)
    with open('models/eval_mask_gt.pkl','wb') as f:
        print('Dumping ground truth...')
        pickle.dump(gt_list,f)

    print('Finished prediction!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='../A2D', help='a2d dataset')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_cls', type=int, default=44)
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    print(args)
predict(args)
            



