# !/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np


def sigmoid(z_arr):

    """ sigmoid函数 """

    return np.array([1.0/(1.0 + math.exp(-z)) for z in z_arr])


def cross_entropy(labels, predicts):

    """ 计算交叉熵 """

    sample_num = labels.shape[0]
    cross_ent_sum = 0
    for i in range(sample_num):
        cross_ent_sum -= (labels[i]*math.log(predicts[i]) + (1 - labels[i])*math.log(1-predicts[i]))
    return float(cross_ent_sum/sample_num)


def gradient_decent(features_mat, error, weights, alpha=0.01):

    """ 梯度下降，更新权重 """

    gradient = features_mat.transpose()*error
    return weights - alpha*gradient


def judge(value, threshold=0.5):
    if value >= threshold:
        return 1
    else:
        return 0


if __name__ == "__main__":
    # 数据初始化
    features = np.mat([[1.5, 2],
                       [1,   1],
                       [1,   3],
                       [2.2, 2.5],
                       [0.5, 4.5],
                       [5,   2],
                       [2,   5.6],
                       [7,   1.2],
                       [3.5, 3],
                       [6,   3.5]])
    b = np.ones(features.shape[0])
    features = np.insert(features, 2, values=b, axis=1)  # 扩充一列偏置
    labs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # 训练
    W = np.mat([0.1, 0.2, 0.3]).transpose()  # 初始权重，也可以随机初始化
    loss = 100
    while loss > 0.1:
        preds = sigmoid(features*W)
        loss = cross_entropy(labs, preds)
        err = np.mat(preds - labs).transpose()
        W = gradient_decent(features, err, W)
        print('Loss: %f' % loss)
    print(W)

    # 测试
    test = np.mat([[3,   4],
                   [1,   3.7],
                   [2,   2.1],
                   [0.5, 2.2],
                   [6,   0.2],
                   [7,   4.5],
                   [2.6, 1.2],
                   [2,   7]])
    b = np.ones(test.shape[0])
    test = np.insert(test, 2, values=b, axis=1)
    test_labels = [1, 0, 0, 0, 1, 1, 0, 1]
    print('Labels  : ', test_labels)
    print('Results : ', [judge(_) for _ in sigmoid(test*W)])