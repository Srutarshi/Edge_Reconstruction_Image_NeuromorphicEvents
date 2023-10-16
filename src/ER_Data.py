import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tforms


# Dataset
class ER_Dataset(Dataset):
    def __init__(self, image_list, mode, thresh):
        self.mode = mode
        self.thresh = thresh

        self.image_list = []
        self.loader = tforms.ToTensor()
        with open(image_list, 'r') as f:
            self.image_list = f.read().splitlines()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        sample = {}

        # Image
        im_name = os.path.join(os.getcwd(), 'data/images/', self.image_list[idx])
        if 'Edge_Reconstruction' not in im_name:
            im_name = os.path.join(im_name[0], 'Edge_Reconstruction/', im_name[1:])
        im = self.image_loader(im_name)
        sample['image'] = im

        sp_name = self.image_list[idx].split('/')

        # Label
        lb_name = os.path.join(os.getcwd(), 'data/labels', sp_name[0], sp_name[2], sp_name[3])
        if 'Edge_Reconstruction' not in lb_name:
            lb_name = os.path.join(lb_name[0], 'Edge_Reconstruction/', lb_name[1:])
        lb = self.image_loader(lb_name)
        if self.mode != 'run':
            sample['label'] = lb

        # Event
        sp_name[-1] = str(int(sp_name[-1][:-4]) - 1).zfill(6) + '.jpg'
        ev_name = os.path.join(os.getcwd(), 'data/images/', '/'.join(sp_name))
        if 'Edge_Reconstruction' not in ev_name:
            ev_name = os.path.join(ev_name[0], 'Edge_Reconstruction/', ev_name[1:])

        ev = self.image_loader(ev_name)
        ev = self.thresh * torch.sign(lb - ev)
        sample['event'] = ev

        return sample

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image)
        if image.size(0) > 1:
            image = image[[0],:,:]
        return 2 * image.float() - 1


# Dataset Creation
def create_datasets(train_list, test_list, val_list, mode, thresh):
    train_data = []
    test_data = []
    val_data = []

    if train_list:
         train_data = ER_Dataset(image_list = train_list,
                                 mode = mode,
                                 thresh = thresh)
    if test_list:
         test_data = ER_Dataset(image_list = test_list,
                                mode = mode,
                                thresh = thresh)
    if val_list:
         val_data = ER_Dataset(image_list = val_list,
                               mode = mode,
                               thresh = thresh)

    return {'train': train_data, 'test': test_data, 'val': val_data}


# Data Loader Creation
def create_dataloaders(datasets, n_workers, batch_size):
    train_loader = []
    test_loader = []
    val_loader = []

    if datasets['train']:
        train_loader = DataLoader(dataset = datasets['train'],
                                  num_workers = n_workers,
                                  batch_size = batch_size,
                                  shuffle = True)
    if datasets['test']:
        test_loader = DataLoader(dataset = datasets['test'],
                                 num_workers = n_workers,
                                 batch_size = batch_size,
                                 shuffle = False)
    if datasets['val']:
        val_loader = DataLoader(dataset = datasets['val'],
                                num_workers = n_workers,
                                batch_size = batch_size,
                                shuffle = False)

    return {'train': train_loader, 'test': test_loader, 'val': val_loader}

###