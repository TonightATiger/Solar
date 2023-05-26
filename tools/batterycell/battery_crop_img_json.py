import os
import glob
import cv2 as cv
import numpy as np
import json
import tqdm


class_dict = {
            'background' : '0BeiJing', 
            '点黑' : '1DianHei', 
            '块黑' : '2KuaiHei', 
            '水印' : '3ShuiYin', 
            '过刻' : '4GuoKe',
            '隐裂' : '5YinLie', 
            '花篮印' : '6HuaLanYin', 
            '白色水印' : '7BaiSeShuiYin', 
            '磨伤' : '8MoShang',
            '亮点黑' : '9LiangDianHei', 
            '皮带印' : '10PiDaiYin', 
            '滚轮印' : '11GunLunYin', 
            '亮点' : '12LiangDian',
            '白点' : '13BaiDian',
            '缺口' : '14QueKou'
}


def read_json(file_path):
    with open(file_path, encoding='utf-8') as json_file:
        return json.load(json_file)


def transPoly2Rect(polygon):
    pts_ar = np.array(polygon)
    xmin = int(min(pts_ar[:, 0]))
    xmax = int(max(pts_ar[:, 0]))
    ymin = int(min(pts_ar[:, 1]))
    ymax = int(max(pts_ar[:, 1]))
    return xmin, ymin, xmax, ymax


def checkOverlap(boxa, boxb):
    x1, y1, w1, h1 = boxa[:4]
    x2, y2, w2, h2 = boxb[:4]
    if (x1 > x2 + w2):
        return 0
    if (y1 > y2 + h2):
        return 0
    if (x1 + w1 < x2):
        return 0
    if (y1 + h1 < y2):
        return 0
    colInt = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return overlap_area / (area1 + area2 - overlap_area)

def unionBox(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    h = max(a[0] + a[2], b[0] + b[2]) - x
    w = max(a[1] + a[3], b[1] + b[3]) - y
    labels = a[4] + b[4]
    return [x, y, w, h, labels]

def intersectionBox(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return ()
    return [x, y, w, h]

def rectMerge_sxf(rects: []):
    # rects => [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
    rectList = rects.copy()
    rectList.sort()
    new_array = []
    complete = 1
    i = 0
    while i < len(rectList):
        j = i + 1
        succees_once = 0
        while j < len(rectList):
            boxa = rectList[i]
            boxb = rectList[j]
            if checkOverlap(boxa, boxb):  # intersectionBox(boxa, boxb)
                complete = 0
                new_array.append(unionBox(boxa, boxb))  # 合并有交集的框
                succees_once = 1
                rectList.remove(boxa)
                rectList.remove(boxb)
                break
            j += 1
        if succees_once:
            continue
        i += 1
    new_array.extend(rectList)
    if complete == 0:
        complete, new_array = rectMerge_sxf(new_array)
    return complete, new_array


if __name__ == '__main__':

    # image_path = r"G:\test"
    image_paths = [
        r'D:\datasets\seg_20230210\2.8_254',
        r'D:\datasets\seg_20230210\2.8_806',
        r'D:\datasets\seg_20230210\2.9_899',
        r'D:\datasets\seg_20230210\2.10_857',
        # r'G:\test'
    ]
    for image_path in tqdm.tqdm(image_paths):
        # image_path = r'G:\2.8_254'

        crop_W = 32
        crop_H = 32

        border_w = 8
        border_h = 8

        pad_W = 10
        pad_H = 10

        images = glob.glob(os.path.join(image_path, '*.bmp'))
        for img_path in images:
            # save_path = image_path + '_crop_iamge'
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            json_path = img_path.replace('.bmp', '.json')
            img = cv.imread(img_path)
            jf = read_json(json_path)
            img_data = cv.copyMakeBorder(img, int(crop_W/2) + border_h, int(crop_W/2) 
                                        + border_h, int(crop_H/2) + border_w, int(crop_H/2) 
                                        + border_w, cv.BORDER_CONSTANT, value=(0, 0, 0))
            labels = jf['shapes']
            bbox = []
            
            for lab in labels:
                points =  lab['points']
                label = class_dict[lab['label']]
                # if label not in ['SY', 'BSSY', 'PDY', 'GLY']:
                xmin, ymin, xmax, ymax = transPoly2Rect(points)
                # 上下扩展
                h = xmax - xmin
                w = ymax - ymin

                w_label = int(w+pad_W)+1
                h_label = int(h+pad_H)+1
                x = int(xmin-(w_label-w)/2)+1
                y = int(ymin-(h_label-h)/2)+1
                box = [x, y, w_label, h_label]
                box.append([label])
                bbox.append(box)
                    # print('1111',x, y, w_label, h_label)
            # _, res = rectMerge_sxf(bbox)
            res = bbox.copy()

            count = 0
            try:
                for idx, bline in enumerate(res):
                    x = bline[0]
                    y = bline[1]
                    w1 = bline[2]
                    h1 = bline[3]
                    labels = bline[4]
                    
                    # cx = x + int(h1/2)
                    # cy = y + int(w1/2)
                    # print('2222', x, y, w1, h1)
                    if w1 < crop_W:
                        y = max(int(y - ((crop_W-w1)/2)), 0)
                        w1 = crop_W
                    if h1 < crop_H:
                        x = max(int(x - ((crop_H-h1)/2)), 0)
                        h1 = crop_H
                    # print('3333', x, y, w1, h1)
                    img_crop = img[abs(y):abs(y)+w1, abs(x):abs(x)+h1]
                    tempH = img_crop.shape[0]
                    tempW = img_crop.shape[1]
                    ratio = 0
                    if tempH >= tempW:
                        ratio = crop_H/tempH
                    else:
                        ratio = crop_W/tempW
                        
                    Img_ = cv.resize(img_crop, (None, None), fx=ratio, fy=ratio)
                    tempH_ = int((crop_H-Img_.shape[0])/2)
                    tempW_ = int((crop_W-Img_.shape[1])/2)
                    Img_ = cv.copyMakeBorder(Img_, tempH_, tempH_, tempW_, tempW_, cv.BORDER_CONSTANT, value=(0))
                    
                    Img_ = cv.resize(Img_, (crop_W, crop_H))
                    
                    save_root = os.path.join(os.path.dirname(image_path), 'classfier_0301_unmerge_32')
                    for label in labels:
                        save_path = os.path.join(save_root, label)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        cv.imwrite(os.path.join(save_path, os.path.basename(img_path).split('.bmp')[0] + '_' + str(idx) + '.bmp'), Img_)
                    count += 1
            except Exception as e:
                print(e, img_path)



                # img_crop = img[y:y + crop_H, x:x+crop_W]
                
