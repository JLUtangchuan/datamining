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
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from scipy.special import softmax


class Classifer():
    def __init__(self, model_name,X_train, X_test, y_train, y_test):
        # 这里进行算法的初始化
        self.model_name = model_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        if self.model_name == 'LR':
            self.model = LogisticRegression(C=1, random_state=0)
        elif self.model_name == 'RF':
            self.model = RandomForestClassifier(n_estimators=300,oob_score=True)
        elif self.model_name == 'SVC':
            self.model = SVC(class_weight='balanced')
        elif self.model_name == 'Xgboost':
            self.model = GradientBoostingClassifier()
        else:
            pass
        # 训练
        self.model.fit(X_train, y_train)
    def get_all_resutls(self):
        # 返回训练Accuracy，测试Accuracy，测试Precision和Recall
        predict_train = self.model.predict(self.X_train)
        predict_test = self.model.predict(self.X_test)
        return metrics.accuracy_score(y_train, predict_train),metrics.accuracy_score(y_test, predict_test),metrics.precision_score(y_test, predict_test),metrics.recall_score(y_test,predict_test)

    def predict(self, boy, girl):
        # 这里返回结果 0,1 或者一个得分
        # 这里应该得分越大表示匹配度越高
        # score = model.fit() ???
        # feature_vec拼接boy和girl，拼接后将self.X_train[0].reshape(-1, feature_num))替换为feature_vec
        # feature_vec shape:1*特征数
        predict_01_p = softmax(self.model.predict_proba(self.X_train[0].reshape(-1, feature_num)))
        predict_01 = self.model.predict(self.X_train[0].reshape(-1, feature_num))
        # 返回一个得分即匹配成功的得分
        return predict_01_p[0][1]

def get_model(model_name,X_train, X_test, y_train, y_test):
    model = Classifer(model_name,X_train, X_test, y_train, y_test)
    return model


if __name__ == '__main__':
    df = pd.read_csv('data/data.csv', encoding="ISO-8859-1")
    df.zipcode = df.zipcode.astype('category')
    df.goal = df.goal.astype('category')
    df.date = df.date.astype('category')
    df.go_out = df.go_out.astype('category')
    df.career_c = df.career_c.astype('category')
    df.field_cd = df.field_cd.astype('category')
    df.length = df.length.astype('category')
    df_X, y = df.drop(['match'], axis=1), df['match']
    df_X = df_X.dropna(axis=1,thresh=6600)
    df_X = df_X.fillna(df.median())
    print(np.sum(df_X.isnull().sum()))
    
    print(df_X[['field','from','zipcode','career']].head())
    df_X.drop(['field','from','zipcode','career'],axis=1,inplace=True)
    # print(np.sum(df_X.isnull().sum()))
    X = StandardScaler().fit_transform(df_X)
    # print(X)
    pca = PCA().fit(X)
    pca_reduced = PCA(n_components=65).fit_transform(X)
    pca_reduced.shape
    X_train, X_test, y_train, y_test = train_test_split(pca_reduced,y,random_state=23)
    
    # 筛选特征
    # 手动筛选的特征
    feature_num=65
    # df_X_selectd=df_X[['attr2_1','attr_o']]

    # X_train, X_test, y_train, y_test = train_test_split(df_X_selectd,y,random_state=23)
    model_name='RF'
    model=get_model(model_name,X_train, X_test, y_train, y_test)
    train_acc,test_acc,test_prec,recall=model.get_all_resutls()
    print(train_acc)
    print(test_acc)
    print(test_prec)
    print(recall)
    
    # 获取boy和girl的特征
    boy=0
    girl=0
    # 输出拼接后得分
    print(model.predict(boy,girl))