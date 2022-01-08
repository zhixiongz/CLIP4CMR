"""
评价指标是mAP：precision只考虑了准确率（查准率），但是没有考虑召回率（查全率）。准确率高召回率就低
AP综合考虑准确率和召回率，衡量的是学出来的模型在每个类别上的好坏，
mAP衡量的是学出的模型在所有类别上的好坏，得到AP后mAP的计算就变得很简单了，就是取所有AP的平均值。
"""
import numpy as np
import scipy.spatial
import torch
from collections import Counter


def calc_precision_at_K(order, label, K):
    ranks = order[:, :K]
    tmp = 0
    for i in range(ranks.shape[0]):
        r_label = label[i]
        query = ranks[i]
        query_label = label[query]
        res = Counter(query_label)
        n_pos = res[r_label]
        tmp = tmp + n_pos / K
    return tmp / order.shape[0]


def fx_calc_map_label(image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort() # [batch, batch]
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)


def fx_calc_recall(image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')

    ord = dist.argsort() # [batch, batch]
    ranks = np.zeros(image.shape[0])

    # R@K
    for i in range(image.shape[0]):
        q_label = label[i]
        r_labels = label[ord[i]]
        ranks[i] = np.where(r_labels == q_label)[0][0]
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Prec@K
    for K in [1, 2, 4, 8, 16]:
        prec_at_k = calc_precision_at_K(ord, label, K)
        print("P@{} : {:.3f}".format(k, 100 * prec_at_k))

    return r1, r5, r10


def calc_mean(label, res, num_class):
    num_list = [0 for i in range(10)]
    value_list = [0 for i in range(10)]
    for i in range(len(res)):
        num_list[label[i]] += 1
        value_list[label[i]] += res[i]
    for i in range(num_class):
        if num_list[i] != 0:
            value_list[i] = value_list[i]/num_list[i]
            value_list[i] = round(value_list[i], 4)
        else:
            value_list[i] = 0
    return value_list
