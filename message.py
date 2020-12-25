#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   message.py
@Time    :   2020/12/21 17:58:13
@Author  :   Tang Chuan 
@Contact :   tangchuan20@mails.jlu.edu.cn
@Desc    :   发邮件相关
'''



import smtplib
import json
import poplib
from email import parser
from email.mime.text import MIMEText
from email.header import Header


def sendEmail(msg_from, msg_to, auth_id, title, content):
    """发送邮件：目前只支持qq邮箱自动发送邮件

    Args:
        msg_from ([type]): [description]
        msg_to ([type]): [description]
        auth_id ([type]): [description]
        title ([type]): [description]
        content ([type]): [description]
    """
    msg = MIMEText(content)
    msg['Subject'] = title
    msg['From'] = msg_from
    msg['To'] = msg_to
    try:
        s = smtplib.SMTP_SSL("smtp.qq.com",465)
        s.login(msg_from, auth_id)
        s.sendmail(msg_from, msg_to, msg.as_string())
        print("发送成功", msg_to)
    except s.SMTPException:
        print("发送失败")
    finally:
        s.quit()

def send_mails(infos, scores):
    # 读取信息                       
    filename = './info.json'
    with open(filename, 'r') as f:
        dic = json.load(f)
    msg_from = dic['msg_from'] #发送方邮箱
    passwd = dic['passwd']  #填入发送方邮箱的授权码
    subject = "快速匹配"
    for (b, g), s in zip(infos, scores):
        email_b = b['email']
        weixin_b = b['weixin']
        email_g = g['email']
        weixin_g = g['weixin']

        content_g = f"恭喜您,为您匹配到了一个ta, 你们的得分是{s:.2f}, ta的微信号是 {weixin_b}, 邮箱是 {email_b}"
        content_b = f"恭喜您,为您匹配到了一个ta, 你们的得分是{s:.2f}, ta的微信号是 {weixin_g}, 邮箱是 {email_g}"
        # 这里可以发挥想象, 可以将对方的一些信息也自动发送过去
        
        # print(content_g)
        sendEmail(msg_from, email_g, passwd, subject, content_g)
        sendEmail(msg_from, email_b, passwd, subject, content_b)
    
