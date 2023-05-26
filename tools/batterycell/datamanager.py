#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: charles zhou
"""
import os
import shutil
#import tqdm
import random
import glob


def write_txt(txt_path, image_list, cls):
    label = class_table.index(cls)
    with open(txt_path, 'a+') as t:
        for image in image_list:
            image_name = cls + '/' + image + ' '+str(label) + '\n'
            t.write(image_name)
    t.close()


def move_images(image_root_dir, image_list, save_folder, label):
    save_path = os.path.join(save_folder, label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_path = os.path.join(image_root_dir, label)
    for image in image_list:
        image_name = os.path.join(image_path, image)
        save_name = os.path.join(save_path, image)
        shutil.copy(image_name, save_name)



# class_table = ['0BG', '1DianHei', '2KuaiHei', '3ShuiYin', '4GuoKe', '5YinLie',
#                '6HuaLanYin', '7BaiSeShuiYin', '8MoShang', '9LiangDianHei',
#                '10PiDaiYin', '11GunLunYin', '12LiangDian', '13QueKou']
#class_table = ['0-YL', '1-BB', '2-BianBuYin', '3-Yin','4-HeiBan','5-HeiXian','6-ZangWu', '7-TongXinYuan','8-HuaShang','9-GuoKe','10-Other']
# class_table = ['BG', '白色水印', '点黑',  '过刻', '滚轮印', '块黑', '亮点 ', '亮点黑', '皮带印', '缺口', '水印', '隐裂']
class_table = ['0-Other', '1-YL', '2-BB','3-Other']

if __name__ == '__main__':
    image_root_dir = r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\data'
    ratio = 0.8
    classes = os.listdir(image_root_dir)
    random.seed(40)

    for cls in classes:
        try:
            class_path = os.path.join(image_root_dir, cls)
            images = os.listdir(class_path)
            image_count = len(images)
            train_count = int(image_count * ratio)
            train_image = random.sample(images, train_count)
            val_image = []
            for t in images:
                if t not in train_image:
                    val_image.append(t)
            meta = os.path.join(os.path.dirname(image_root_dir), 'meta')
            if not os.path.exists(meta):
                os.makedirs(meta)
            train_path = os.path.join(meta, 'train.txt')
            val_path = os.path.join(meta, 'val.txt')

            write_txt(train_path, train_image, cls)
            write_txt(val_path, val_image, cls)



            train_save_path = os.path.join((image_root_dir + '_split'), 'train')
            val_save_path = os.path.join((image_root_dir + '_split'), 'val')
            move_images(image_root_dir, train_image, train_save_path, cls)
            move_images(image_root_dir, val_image, val_save_path, cls)
        except Exception as e:
            print(e)
            continue

