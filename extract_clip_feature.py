from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import json
import clip
import torch


class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labs,
            ids
    ):
        self.images = images
        self.texts = texts
        self.labs = labs
        self.ids = ids

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        id = self.ids[index]
        return img, text, lab, id

    def __len__(self):
        count = len(self.texts)
        return count


def load_dataset(name, bsz=100):
    train_loc = 'data/'+name+'/train.pkl'
    test_loc = 'data/' + name + '/test.pkl'
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = data['label']
        train_texts = data['text']
        train_images = data['image']
        train_ids = data['ids']
    with open(test_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        test_labels = data['label']
        test_texts = data['text']
        test_images = data['image']
        test_ids = data['ids']
    imgs = {'train': train_images, 'test': test_images}
    texts = {'train': train_texts,  'test': test_texts}
    labs = {'train': train_labels, 'test': test_labels}
    ids = {'train': train_ids, 'test': test_ids}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x], ids=ids[x])
               for x in ['train', 'test']}

    shuffle = {'train': False, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=bsz,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    return dataloader




if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

    dataloaders = load_dataset('nus-wide')  # wiki/pascal/xmedianet/nus-wide
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    train_imgs, dev_imgs, test_imgs = [], [], []
    train_caps, dev_caps, test_caps = [], [], []
    train_labs, dev_labs, test_labs = [], [], []
    train_ids, dev_ids, test_ids = [], [], []

    for img, text, lab, id in test_loader:
        img = img.squeeze().to(device)
        text = text.squeeze().to(device)
        with torch.no_grad():
            image_features = model.encode_image(img)
            text_features = model.encode_text(text)
            print(image_features, image_features.shape)  # [bsz, 1024]
            print(text_features, text_features.shape)    # [bsz, 1024]
            image_features = image_features.detach().cpu().numpy()
            text_features = text_features.detach().cpu().numpy()
            test_imgs.append(image_features)
            test_caps.append(text_features)
            test_labs.append(lab.numpy())
            test_ids.append(id)
    test_imgs = np.concatenate(test_imgs)
    test_caps = np.concatenate(test_caps)
    test_labs = np.concatenate(test_labs)
    test_ids = np.concatenate(test_ids)
    test_data = {'image': test_imgs, "text": test_caps, "label": test_labs, 'ids': test_ids}
    with open('clip_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    print('Successfully process test data')

    for img, text, lab, id in train_loader:
        # print(img, img.shape)    # [bsz, 1, 3, 224, 224]
        # print(text, text.shape)  # [bsz, 1, 77]
        # print(lab, lab.shape)    # [bsz]
        img = img.squeeze().to(device)
        text = text.squeeze().to(device)
        with torch.no_grad():
            image_features = model.encode_image(img)
            text_features = model.encode_text(text)
            print(image_features, image_features.shape)  # [bsz, 1024]
            print(text_features, text_features.shape)    # [bsz, 1024]
            image_features = image_features.detach().cpu().numpy()
            text_features = text_features.detach().cpu().numpy()
            train_imgs.append(image_features)
            train_caps.append(text_features)
            train_labs.append(lab.numpy())
            train_ids.append(id)
    train_imgs = np.concatenate(train_imgs)
    train_caps = np.concatenate(train_caps)
    train_labs = np.concatenate(train_labs)
    train_ids = np.concatenate(train_ids)
    train_data = {'image': train_imgs, "text": train_caps, "label": train_labs, 'ids': train_ids}
    with open('clip_train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    print('Successfully process training data')