# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def move(data_mat):

    """ 将数据平移至以平均值为中心 """

    means = np.mean(data_mat, axis=1)
    res = []
    for i in range(means.shape[0]):
        res.append((data_mat[i, :]-means[i]).tolist()[0])
    return np.mat(res)


def pca(data_mat, principal_num):

    """ 主成分分析 """

    moved_mat = move(data_mat)
    cov_mat = np.cov(moved_mat)
    eig_values, eig_vectors = np.linalg.eig(cov_mat)
    idx_sorted = np.argsort(-eig_values)
    vectors_sorted = np.mat([eig_vectors[i, :] for i in idx_sorted[:principal_num]])
    return vectors_sorted, vectors_sorted*moved_mat


if __name__=='__main__':
    data = np.mat([[-1, -1,  0,  2,  0],
                     [-2,  0,  0,  1,  1]])
    print(pca(data, 1))