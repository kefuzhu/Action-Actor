import os
import sys
sys.path.append('')
import random
import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import Dataset
import loader.transforms as tf
from cfg.cfg_a2d import train as train_cfg
from cfg.cfg_a2d import val as val_cfg
from cfg.cfg_a2d import test as test_cfg


class A2DDataset(Dataset):
    """A2DDataset for train and validation"""

    #num_class = 44
    num_class_orig = 80
    ignore_label = 255
    background_label = 0

    # 36+8=44 valid classes
    valid_cls = [0, # 0
        11, 12, 13, 15, 16, 17, 18, 19,    # 1-8
        21, 22, 26, 28, 29,    # 9-13
        34, 35, 36, 39,    # 14-17
        41, 43, 44, 45, 46, 48, 49,    # 18-24
        54, 55, 56, 57, 59,    # 25-29
        61, 63, 65, 66, 67, 68, 69,    # 30-36
        72, 73, 75, 76, 77, 78, 79] # 37-43
    num_valid_cls = len(valid_cls)
    convert_label = dict()
    convert_label_back = dict()
    for i, label in enumerate(valid_cls):
        convert_label[label] = i
        convert_label_back[i] = label

    label_80to44 = np.ones((num_class_orig))*255
    for label in range(num_class_orig):
        if label in convert_label:
            label_80to44[label] = convert_label[label]
    label_80to44 = label_80to44.astype(np.uint8)
    #print(label_80to44)
    label_44to80 = np.ones((num_valid_cls))*255
    for i in range(num_valid_cls):
        label_44to80[i] = convert_label_back[i]
    label_44to80 = label_44to80.astype(np.uint8)

    # official color map
    cmap = np.array([[0,0,0],   # 0
        #[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],    # 1-10
        [255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0],    # 1-10
        [52,1,1],[103,1,1],[154,1,1],[205,1,1],[255,1,1],   # 11-15
        [255,51,51],[255,103,103],[255,154,154],[255,205,205],[0,0,0],  # 16-20
        [52,46,1],[103,92,1],[154,138,1],[205,184,1],[255,230,1],   # 21-25
        [255,235,51],[255,240,103],[255,245,154],[255,250,205],[0,0,0], # 26-30
        [11,52,1],[21,103,1],[31,154,1],[41,205,1],[52,255,1],  # 31-35
        [92,255,51],[133,255,103],[174,255,154],[215,255,205],[0,0,0],  # 36-40
        [1,52,36],[1,103,72],[1,154,108],[1,205,143],[1,255,179],   #41-45
        [51,255,194],[103,255,210],[154,255,225],[205,255,240],[0,0,0], # 46-50
        [1,21,52],[1,41,103],[1,62,154],[1,82,205],[1,103,255], # 51-55
        [51,133,255],[103,164,255],[154,194,255],[205,225,255],[0,0,0], # 56-60
        [26,1,52],[52,1,103],[77,1,154],[103,1,205],[128,1,255],    # 61-65
        [154,51,255],[179,103,255],[205,154,255],[230,205,255],[0,0,0], #66-70
        [52,1,31],[103,1,62],[154,1,92],[205,1,123],[255,1,153],    #71-75
        [255,51,174],[255,103,194],[255,154,215],[255,205,235]  #76-79
    ])
    cmap = cmap.astype(np.uint8)

    # 80=1(bg) + 9+7(none) + 7(actor)*9(action)
    class_names = np.array([
        'background',
        'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 
        'none',
        'adult-climbing',
        'adult-crawling',
        'adult-eating',
        'adult-flying',
        'adult-jumping',
        'adult-rolling',
        'adult-running',
        'adult-walking',
        'adult-none',
        'none',
        'baby-climbing',
        'baby-crawling',
        'baby-eating',
        'baby-flying',
        'baby-jumping',
        'baby-rolling',
        'baby-running',
        'baby-walking',
        'baby-none',
        'none',
        'ball-climbing',
        'ball-crawling',
        'ball-eating',
        'ball-flying',
        'ball-jumping',
        'ball-rolling',
        'ball-running',
        'ball-walking',
        'ball-none',
        'none',
        'bird-climbing',
        'bird-crawling',
        'bird-eating',
        'bird-flying',
        'bird-jumping',
        'bird-rolling',
        'bird-running',
        'bird-walking',
        'bird-none',
        'none',
        'car-climbing',
        'car-crawling',
        'car-eating',
        'car-flying',
        'car-jumping',
        'car-rolling',
        'car-running',
        'car-walking',
        'car-none',
        'none',
        'cat-climbing',
        'cat-crawling',
        'cat-eating',
        'cat-flying',
        'cat-jumping',
        'cat-rolling',
        'cat-running',
        'cat-walking',
        'cat-none',
        'none',
        'dog-climbing',
        'dog-crawling',
        'dog-eating',
        'dog-flying',
        'dog-jumping',
        'dog-rolling',
        'dog-running',
        'dog-walking',
        'dog-none',
    ])
    def __init__(self, config):
        super(A2DDataset, self).__init__()
        dataset_path = '../A2D/'
        with open(
                os.path.join(dataset_path, 'list',
                             config.data_list + '.txt')) as f:
            self.img_list = [line.rstrip() for line in f]
        self.img_dir = os.path.join(dataset_path, 'pngs320H')
        self.gt_dir = os.path.join(dataset_path, 'Annotations/mat')
        self.config = config
        self.class_names = [A2DDataset.class_names[cls] for cls in A2DDataset.valid_cls]

    def untransform(self, img, lbl):

        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.config.input_mean
        img = img.astype(np.uint8)
        # go back to rgb
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        :param idx: index
        :return: img (c, h, w)
        :return: label (h, w), range: [0, 43]
        """
        vd_frame_idx = self.img_list[idx]
        # load img
        image_path = os.path.join(self.img_dir, vd_frame_idx + '.png')
        image = cv2.imread(image_path).astype(np.float32)
        # load gt
        gt_load_path = os.path.join(self.gt_dir, vd_frame_idx + '.mat')
        label_orig = h5py.File(gt_load_path)['reS_id'].value
        label_orig = np.transpose(label_orig)
        label = A2DDataset.label_80to44[label_orig]

        # flip
        if hasattr(self.config, 'flip') and self.config.flip:
            image, label = tf.group_random_flip([image, label])

        if hasattr(self.config, 'crop_policy'):
            target_size = self.config.crop_size
            if self.config.crop_policy == 'none':
                # resize
                image, label = tf.group_rescale([image, label],
                                                #self.config.scale_factor,
                                                target_size,
                                                [cv2.INTER_LINEAR, cv2.INTER_NEAREST])
            else:
                # resize -> crop -> pad 
                image, label = tf.group_rescale([image, label],
                                                self.config.scale_factor,
                                                [cv2.INTER_LINEAR, cv2.INTER_NEAREST])
                if self.config.crop_policy == 'random':
                    image, label = tf.group_random_crop([image, label], target_size)
                    image, label = tf.group_random_pad(
                        [image, label], target_size,
                        [self.config.input_mean, A2DDataset.background_label])
                elif self.config.crop_policy == 'center':
                    image, label = tf.group_center_crop([image, label], target_size)
                    image, label = tf.group_concer_pad(
                        [image, label], target_size,
                        [self.config.input_mean, A2DDataset.background_label])
                else:
                    ValueError('Unknown crop policy: {}'.format(
                        self.config.crop_policy))

        if hasattr(self.config, 'rotation') and random.random() < 0.5:
            image, label = tf.group_rotation(
                [image, label], self.config.rotation,
                [cv2.INTER_LINEAR, cv2.INTER_NEAREST],
                [self.config.input_mean, A2DDataset.background_label])
        # blur
        if hasattr(self.config,
                   'blur') and self.config.blur and random.random() < 0.5:
            image = tf.blur(image)


        # norm
        image = tf.normalize(image, self.config.input_mean,
                             self.config.input_std)

        # image tensor shape (c,h,w)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        # label tensor shape (h,w)
        label = torch.from_numpy(label).contiguous().long()

        return image, label


class A2DTestDataset(A2DDataset):
    """
    A2DTestDataset only for test
    """
    def __getitem__(self, idx):
        vd_frame_idx = self.img_list[idx]
        # load img
        image_path = os.path.join(self.img_dir, vd_frame_idx + '.png')
        image = cv2.imread(image_path).astype(np.float32)

        # norm
        image = tf.normalize(image, self.config.input_mean,
                             self.config.input_std)
        # image tensor shape (c,h,w)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        return image


if __name__ == '__main__':
    # train dataset
    dataset = A2DDataset(train_cfg)
    # val dataset
    # dataset = A2DDataset(val_cfg)
    # test datset
    # dataset = A2DTestDataset(test_cfg)
    img_list = dataset.img_list
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        img = data[0]
        label = data[1]
        print('{}/{}: image shape'.format(i + 1, len(dataset)),
              img.shape, 'lebel', label.shape, 'path', dataset.img_list[i])

