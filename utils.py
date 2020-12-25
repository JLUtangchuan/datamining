#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/12/21 16:42:29
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   一些工具函数
'''
import pandas as pd

def load_excel(file_name):
    origin_df = pd.read_excel(file_name)
    df = origin_df.iloc[:,6:14]
    df.columns = ['name', 'sex', 'email', 'weixin', 'a1', 'a2', 'a3', 'a4']
    # TODO 加入数据预处理工作：类别转编码等
    
    # 这里就按照男女分开
    boys = df[df['sex'] == 1]
    girls = df[df['sex'] == 2]
    # print('Boys', boys.shape[0], 'Girls', girls.shape[0])
    return boys, girls

def get_paired_info(boys, girls, sim, pairs):
    """拿到pair的邮件,相似度得分可以计算一个相似度排名
    """
    infos = []
    scores = []
    num = len(pairs)
    print("一共有%d对匹配成功!!!" % num)
    for i, j in pairs:
        boy = boys.iloc[i]
        girl = girls.iloc[j]
        infos.append((boy, girl))
        scores.append(sim[i, j])
    
    return infos, scores

def show_match_info():
    # 展示部分信息,导出到一个文件中或者直接输出出来
    pass

