import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import copy
import time
import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
from model import model
from evaluate import fx_calc_map_label
from metrics import PAN, Triplet_Loss, Contrastive_Loss, Label_Regression_Loss, Modality_invariant_Loss, Proxy_Anchor
from torch.autograd import Function
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from extract_clip_feature import CustomDataSet

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
loss_fct = nn.CrossEntropyLoss()


def one_hot(x, num_class):

	return torch.eye(num_class)[x,:]


def train(model, loader, optimizer, num_class, choose_loss='PAN', modality_imbalanced=False):
    model.train()
    running_loss = 0.0
    for img, text, labels, id in loader:  
        optimizer.zero_grad()
        text = text.to(device)
        img = img.to(device)
        label_realvalue = labels.int().type(torch.long).to(device)  # nuswide, wiki, xmedia
        # label_realvalue = (labels - 1).int().type(torch.long).to(device)  # pascal
        centers, img_feature, text_feature, img_predict, text_predict = model(img, text)
        centers = centers[:img_feature.shape[1]]  # multiple GPUs
        if modality_imbalanced:     # i.e. 100%I, 30%T
            bsz = int(img_feature.shape[0]/10)
            text_feature = text_feature[:3*bsz]
            text_label = label_realvalue[:3*bsz]

        if choose_loss == 'CL':   # Contrastive loss
            loss = Contrastive_Loss(img_feature, text_feature, label_realvalue)
        elif choose_loss == 'ML':     # Modality-invariant loss
            loss = Modality_invariant_Loss(img_feature, text_feature, label_realvalue)
        elif choose_loss == 'TL':      # Triplet loss
            loss = Triplet_Loss(img_feature, text_feature, label_realvalue) \
                   + Triplet_Loss(text_feature, img_feature, label_realvalue)
        elif choose_loss == 'LRL':     # Label regression loss
            label_onehot = one_hot(label_realvalue, num_class).to(device)
            loss = Label_Regression_Loss(img_predict, text_predict, label_onehot)
        elif choose_loss == 'CEL':     # Cross-entropy loss
            loss = loss_fct(img_predict, label_realvalue) + loss_fct(text_predict, label_realvalue)
        elif choose_loss == 'PCL':     # Prototype contrastive loss
            loss = PAN(img_feature, torch.t(centers), label_realvalue) \
                   + PAN(text_feature, torch.t(centers), label_realvalue)
	elif choose_loss == 'PNCA':     # Prototype contrastive loss, sample anchor
            loss = Proxy_NCA(img_feature, torch.t(centers), label_realvalue) \
                   + Proxy_NCA(text_feature, torch.t(centers), label_realvalue)
	elif choose_loss == 'P_Anchor':     # sample contrastive loss, prototypical anchor
            loss = Proxy_Anchor(img_feature, torch.t(centers), label_realvalue) \
                   + Proxy_Anchor(text_feature, torch.t(centers), label_realvalue)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    running_loss = 0.0
    t_imgs, t_txts, t_labels = [], [], []
    with torch.no_grad():
        for img, text, labels, id in loader:
            text = text.to(device)
            img = img.to(device)
            labels = labels.int().to(device)
            _, img_feature, text_feature, img_predict, text_predict = model(img, text)
            t_imgs.append(img_feature.cpu().numpy())
            t_txts.append(text_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())

    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))

    return i_map, t_map, t_imgs, t_txts, t_labels


def figure_plt(Train_Loss, Valid_Loss, png_path):
    plt.figure()
    Epoch = len(Train_Loss)
    X = range(1, Epoch + 1)
    plt.plot(X, Train_Loss, label='Train loss')
    plt.plot(X, Valid_Loss, label='Valid loss')
    plt.legend()
    # plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(png_path)
    # plt.show()


def load_dataset(name, bsz):
    train_loc = 'data/'+name+'/clip_train.pkl'
    test_loc = 'data/'+name+'/clip_test.pkl'
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

    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=bsz,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    return dataloader


if __name__ == '__main__':
    batch_size = 300
    dataloaders = load_dataset('nus-wide', batch_size)  # wiki, nus-wide, pascal, xmedianet
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    print(len(train_loader), len(test_loader))

    num_class = 10

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:3" if USE_CUDA else "cpu")

    MAX_EPOCH = 500
    temperature = 1.0
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0
    early_stop = 10

    model_ft = model(num_class=num_class).to(device)
    model_ft = nn.DataParallel(model_ft, device_ids=[3, 1])
    model_ft.to(device)

    params_to_update = list(model_ft.parameters())
    total = sum([param.nelement() for param in params_to_update])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # Observe that all parameters are being optimized
    optimizer_all = optim.Adam(params_to_update, lr=lr, betas=betas)
    for state in [1]:
        print('...Training is beginning...', state)
        # Train and evaluate
        train_loss_history = []
        test_loss_history = []
        i_map = []
        t_map = []
        best_map = 0.0
        no_up = 0  # early stop
        best_model_wts = copy.deepcopy(model_ft.state_dict())  

        for epoch in range(MAX_EPOCH):
            print('==============================')
            start_time = time.time()
            train_loss = train(model_ft, train_loader, optimizer_all, num_class=num_class)
            print('Train loss: ', train_loss)

            img2text, text2img, t_imgs, t_txts, t_labels = evaluate(model_ft, test_loader)
            i_map.append(img2text)
            t_map.append(text2img)

            time_elapsed = time.time() - start_time
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')

            if (img2text + text2img) / 2. > best_map:
                best_map = (img2text + text2img) / 2.
                print('New Best model')
                no_up = 0
                best_model_wts = copy.deepcopy(model_ft.state_dict())
                torch.save(model_ft.state_dict(), 'result/nuswide_pan.pt')
                np.savez('result/nuswide/{}.npz'.format(best_map), image=t_imgs, text=t_txts, label=t_labels)
            else:
                no_up += 1
            if no_up >= early_stop:
                break
        print('==============================')
        print(f'Best average mAP: {best_map:.4f}, Epoch: {epoch+1-early_stop}')
        # print(i_map)  # for visualization
        # print(t_map)  # for visualization

