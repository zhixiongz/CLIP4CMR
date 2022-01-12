import torch.utils.data as data
import torch.nn as nn
import numpy as np
import os
import pickle
import h5py
from PIL import Image
import torch
import random
from torchvision import models, transforms
torch.manual_seed(1) # cpu
torch.cuda.manual_seed(1) #gpu
random.seed(1)
np.random.seed(1)

class xmediaPedes(data.Dataset):
    pklname_list = ['train.pkl', 'test.pkl']
    h5name_list = ['train.h5', 'test.h5']
    def __init__(self, root, split, transform=None):

        self.root = root
        self.split = split.lower()
        self.transform = transform

        if self.split == 'train':
            self.pklname = self.pklname_list[0]
            self.h5name = self.h5name_list[0]
            with open(os.path.join(self.root, self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.train_labels = data['labels']
                self.train_captions = data['text_ids']
                self.train_ids = data['ids']
            data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            self.train_images = data_h5py['images']

        elif self.split == 'valid':
            self.pklname = self.pklname_list[1]
            self.h5name = self.h5name_list[1]
            with open(os.path.join(self.root, self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.val_labels = data['labels']
                self.val_captions = data['text_ids']
                self.val_ids = data['ids']
            data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            self.val_images = data_h5py['images']

    def __getitem__(self, index):
        if self.split == 'train':
            img, caption, label, id = self.train_images[index], self.train_captions[index], self.train_labels[index], self.train_ids[index]
        else:
            img, caption, label, id = self.val_images[index], self.val_captions[index], self.val_labels[index], self.val_ids[index] # add:vgg=self.val_vgg[index]

        img = img.transpose((1, 2, 0))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label_one_hot = np.zeros(200)  # NUS-WIDE 10类， Pascal 20类
        label_one_hot[label - 1] = 1

        return img, caption, label_one_hot, id

    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        else:
            return len(self.val_labels)


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    img = torch.tensor([t.numpy() for t in batch[0]])
    caption = torch.tensor([t.numpy() for t in batch[1]])
    label_one_hot = torch.tensor([t.numpy() for t in batch[2]])
    id = torch.tensor([t.numpy() for t in batch[3]])
    del batch
    return img, caption, label_one_hot, id

def load_data(batch_size):
    transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    ])   # 这个是imgnet的均值和方差
    train_data = xmediaPedes('data/xmedia_bert', 'train', transform=transform)
    val_data = xmediaPedes('data/xmedia_bert', 'valid', transform=transform)
    train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size, shuffle=False)
    return train_loader, val_loader