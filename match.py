#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   match.py
@Time    :   2020/12/15 17:17:09
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   二分图匹配
'''

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment



def get_sim(boys, girls, model):
    # 计算相似度
    # 首先按照男女划分成两个DataFrame
    
    boys_len = boys.shape[0]
    girls_len = girls.shape[0]
    sim_mat = np.zeros((boys_len, girls_len))
    for i in range(boys_len):
        for j in range(girls_len):
            sim_mat[i, j] = model.predict(boys.iloc[i], girls.iloc[j])
    return sim_mat

def get_match_result(sim, threshold = 0.):
    # 获得匹配结果
    # 阈值可以过滤掉匹配得分过低的对
    boy_ind, girl_ind = linear_sum_assignment(-sim)
    pairs = []
    # 过滤
    for b, g in zip(boy_ind, girl_ind):
        if sim[b, g] >= threshold:
            pairs.append((b, g))
    # 返回匹配结果下标
    return pairs