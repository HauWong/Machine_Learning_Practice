# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def calculate_prior_prob(dataset, idx):

    """  计算先验概率"""

    sample_num = dataset.shape[0]
    prob_dict = {}
    for sample in dataset:
        label = sample[idx]
        if label not in prob_dict.keys():
            prob_dict[label] = 1
        else:
            prob_dict[label] += 1

    for label in prob_dict.keys():
        prob_dict[label] /= sample_num

    return prob_dict


def calculate_post_prob(dataset, category, feature_idx, value):

    """ 计算某类别中某特征值的后验概率 """

    cond = dataset[:, 0] == category
    arr = dataset[np.where(cond)]
    sample_num = arr.shape[0]
    matched_sample_num = np.sum(arr[:, feature_idx] == value)
    return float(matched_sample_num/sample_num)


class BayesClassifier(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.category_prior_prob_dict = calculate_prior_prob(dataset, 0)
        self.features_prior_prob_list = []
        for i in range(1, dataset.shape[1]):
            self.features_prior_prob_list.append(calculate_prior_prob(dataset, i))

    def predict(self, features_list):

        res_dict = {}
        for label in self.category_prior_prob_dict.keys():
            features_post_prob = 1
            for f_idx in range(len(features_list)):
                features_post_prob *= calculate_post_prob(self.dataset, label, f_idx+1, features_list[f_idx])
            category_prior_prob = self.category_prior_prob_dict[label]
            prob = features_post_prob*category_prior_prob
            res_dict[label] = prob

        return res_dict


if __name__ == '__main__':

    data_set = np.array([['不嫁', '帅', '不好', '矮', '不上进'],
                        ['不嫁', '不帅', '好', '矮', '上进'],
                        ['嫁', '帅', '好', '矮', '上进'],
                        ['嫁', '不帅', '好', '高', '上进'],
                        ['嫁', '不帅', '好', '高', '不上进'],
                        ['不嫁', '帅', '不好', '矮', '上进'],
                        ['不嫁', '帅', '不好', '矮', '上进'],
                        ['嫁', '帅', '好', '高', '不上进'],
                        ['嫁', '不帅', '好', '中', '上进'],
                        ['嫁', '帅', '好', '中', '上进'],
                        ['嫁', '不帅', '不好', '高', '上进'],
                        ['不嫁', '帅', '好', '矮', '不上进'],
                        ['不嫁', '帅', '好', '矮', '不上进'],
                        ['不嫁', '不帅', '不好', '高', '不上进']])
    test = np.array([['不帅', '不好', '矮', '不上进'],
                     ['不帅', '好', '矮', '不上进'],
                     ['不帅', '不好', '高', '不上进'],
                     ['帅', '不好', '高', '上进']])
    b_c = BayesClassifier(data_set)
    for s in test:
        print(b_c.predict(s))
