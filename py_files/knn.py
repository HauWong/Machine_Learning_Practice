# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def calculate_features_distance(feature1, feature2):

    """ 计算特征间的距离 """

    feature_num = feature1.shape[0]
    sum_of_square = 0
    for i in range(feature_num):
        sum_of_square += np.power((feature1[i] - feature2[i]), 2)
    return np.sqrt(sum_of_square)


def classify(samples_group, target, k):

    """ 利用knn算法分类 """

    if samples_group.shape[1] != target.shape[0]+1:
        raise ValueError('Excepted shape of target %d but get %d' %
                         (samples_group.shape[1], target.shape[0]))

    # 计算目标向量与所有样本特征向量的距离
    distances = []
    labels = []
    for sample in samples_group:
        labels.append(sample[0])
        feature = sample[1:]
        distances.append(calculate_features_distance(feature, target))

    # 对所有距离升序排序并统计前k个样本标签出现的数量
    idx_sorted = np.argsort(distances)
    labels_counts = np.zeros(np.max(labels)+1, dtype='uint16')
    for i in range(k):
        if i >= idx_sorted.shape[0]:
            break
        labels_counts[labels[idx_sorted[i]]] += 1

    return np.argmax(labels_counts)


if __name__ == '__main__':
    g = np.array([[0, 3, 104, 50],
                  [2, 2, 100, 10],
                  [0, 1, 81, 46],
                  [1, 101, 10, 60],
                  [1, 99, 5, 58],
                  [3, 98, 2, 12],
                  [3, 67, 10, 5],
                  [2, 6, 98, 11],
                  [2, 10, 96, 8],
                  [0, 5, 89, 45],
                  [1, 100, 12, 45],
                  [3, 101, 8, 4]])
    t = np.array([8, 92, 4])
    print(classify(g, t, 4))