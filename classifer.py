#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   classifer.py
@Time    :   2020/12/21 17:02:30
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   封装分类器
'''
import random

class Classifer():
    def __init__(self, *args, **kwargs):
        # 这里进行算法的初始化
        # model = SVM
        # 
        pass
    
    def predict(self, boy, girl):
        # 这里返回结果 0,1 或者一个得分
        # 这里应该得分越大表示匹配度越高
        # score = model.fit() ???
        return random.random()

def get_model():
    model = Classifer()
    return model
