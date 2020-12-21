#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/12/21 16:42:58
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   程序入口
'''

import pandas as pd
import numpy as np

from classifer import get_model
from match import get_sim, get_match_result
from utils import load_excel, get_paired_info
from message import send_mails
# 全局变量
excel_file = './test.csv'


def main():
    # 主函数
    boys, girls = load_excel(excel_file)
    model = get_model()
    sim = get_sim(boys, girls, model)
    print(sim)
    pairs = get_match_result(sim)
    print(pairs)
    infos, scores = get_paired_info(boys, girls, sim, pairs)
    send_mails(infos, scores)

if __name__ == "__main__":
    main()
