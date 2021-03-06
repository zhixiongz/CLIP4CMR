import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertConfig, BertTokenizer, BertModel
from torchvision.models import alexnet, resnet18, resnet50, inception_v3, vgg19
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import pickle
import math
import clip

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=2048, mindum_dim=512, out_dim=128, dropout_prob=0.1):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = F.relu(self.denseL1(x))
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=768, mindum_dim=512, out_dim=128, dropout_prob=0.1):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = F.relu(self.denseL1(x))
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out


class model(nn.Module):
    def __init__(self, num_class, img_dim=1024, text_dim=1024, mid_dim=256, feature_dim=1024, init_weight=True):
        super(model, self).__init__()

        self.imgnn = ImgNN(input_dim=img_dim, mindum_dim=mid_dim, out_dim=feature_dim)
        self.textnn = TextNN(input_dim=text_dim, mindum_dim=mid_dim, out_dim=feature_dim)

        self.n_classes = num_class
        self.feat_dim = feature_dim
        self.predictLayer = nn.Linear(self.feat_dim, self.n_classes, bias=True)  # ?????????bias??????????????????????????????Proxy-NCA, Normlized Softmax
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes), requires_grad=False)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        nn.init.kaiming_normal_(self.predictLayer.weight.data, mode='fan_out')

    def forward(self, img, text):
        # the normalization of class proxies, the more obvious its effect is in the higher-dimensional representation space
        self.predictLayer.weight.data = l2norm(self.predictLayer.weight.data, dim=-1)
        self.centers.data = l2norm(self.centers.data, dim=0)

        img_features = self.imgnn(img.float())
        img_features = l2norm(img_features, dim=1)
        img_pred = self.predictLayer(img_features)

        text_features = self.textnn(text.float())
        text_features = l2norm(text_features, dim=1)
        text_pred = self.predictLayer(text_features)

        return self.centers, img_features, text_features, img_pred, text_pred


