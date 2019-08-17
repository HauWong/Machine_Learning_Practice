# !/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np


def split_dataset(dataset, feature_idx, value):

    """ 根据给定特征及特征值划分子数据集 """

    res_dataset = []
    for sample in dataset:
        if sample[feature_idx] == value:
            res_dataset.append(sample)
    return np.array(res_dataset)


def calculate_entropy(dataset):

    """ 计算数据集的熵 """

    sample_num = dataset.shape[0]
    label_counts = {}
    for sample in dataset:
        cur_label = sample[0]
        if cur_label not in label_counts.keys():
            label_counts[cur_label] = 1
        else:
            label_counts[cur_label] += 1

    entropy = 0
    for values in label_counts.values():
        prob = values/sample_num
        entropy -= prob*math.log(prob, 2)
    return entropy


def find_best_feature(dataset):

    """ 找到当前最优特征 """

    sample_num, sample_length = dataset.shape
    base_entropy = calculate_entropy(dataset)
    max_info_gain = 0
    res_feature_idx = 0
    for i in range(1, sample_length):
        cur_entropy = 0
        cur_value_set = set(dataset[:, i])
        for value in cur_value_set:
            sub_dataset = split_dataset(dataset, i, value)
            weight = sub_dataset.shape[0]/sample_num
            cur_entropy+= weight*calculate_entropy(sub_dataset)
        info_gain = base_entropy- cur_entropy
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            res_feature_idx = i
    return res_feature_idx


def generate_tree(dataset):

    """ 生成决策树 """

    best_feature = find_best_feature(dataset)
    tree = {best_feature: {}}
    labels = set(dataset[:, 0])
    if len(labels) == 1:
        return dataset[0, 0]
    else:
        values = set(dataset[:, best_feature])
        for value in values:
            sub_dataset = split_dataset(dataset, best_feature, value)
            tree[best_feature][value] = generate_tree(sub_dataset)
    return tree


if __name__ == '__main__':
    data_set = np.array([[4, 1, 1],
                         [4, 1, 1],
                         [5, 1, 0],
                         [5, 0, 1],
                         [5, 0, 1]])
    print(generate_tree(data_set))