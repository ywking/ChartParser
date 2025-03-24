import math
import torch, cv2, os, sys, json
import numpy as np
import tqdm as tq

from PIL import Image, ImageDraw, ImageFont
from numpy import unravel_index
from scipy.interpolate import interp1d
# from skimage.morphology import skeletonize
from scipy import ndimage

def connect_lines(img):
    #img = cv2.imread('line_join_test2.png', 0)     # grayscale image
    #img1 = cv2.imread('line_join_test2.png', 1)    # color image
    
    th = cv2.threshold(img.astype(np.uint8), 150, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)) #(19, 19)
    img = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel)
    
    cnts1 = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts1[0]     # storing contours in a variable


    for i in range(len(cnts)):
        min_dist = max(img.shape[0], img.shape[1])

        cl = []

        ci = cnts[i]
        ci_left = tuple(ci[ci[:, :, 0].argmin()][0])
        ci_right = tuple(ci[ci[:, :, 0].argmax()][0])
        ci_top = tuple(ci[ci[:, :, 1].argmin()][0])
        ci_bottom = tuple(ci[ci[:, :, 1].argmax()][0])
        ci_list = [ci_bottom, ci_left, ci_right, ci_top]

        for j in range(i + 1, len(cnts)):
            cj = cnts[j]
            cj_left = tuple(cj[cj[:, :, 0].argmin()][0])
            cj_right = tuple(cj[cj[:, :, 0].argmax()][0])
            cj_top = tuple(cj[cj[:, :, 1].argmin()][0])
            cj_bottom = tuple(cj[cj[:, :, 1].argmax()][0])
            cj_list = [cj_bottom, cj_left, cj_right, cj_top]

            for pt1 in ci_list:
                for pt2 in cj_list:
                    dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))    
                    if dist < min_dist:
                        min_dist = dist             
                        cl = []
                        cl.append([pt1, pt2, min_dist])
                        
        # if len(cl) > 0:
        #     cv2.line(img, cl[0][0], cl[0][1], (255, 255, 255), thickness = 5)

    img = img//255
    img = skeletonize(img).astype(np.uint8)
    img = img * 255

    return img

def locate_offset(kps, mask, xy):
    result = []
    mask = mask.cpu()
    # if xy == 'x':
    #     kps = sorted(kps, key=lambda x:x[1])
    for (x,pre_y) in kps:
        ymin = 999999
        if x>510:
            continue
        # y = np.where(mask[:,x]==1)[0]
        y = np.where((mask[:,x+1]==1) | (mask[:,x-1]==1) | (mask[:,x]==1))[0]
        lines = []
        if len(y) == 0:
            continue
        y0 = y[0]
        if len(y) == 1:
            lines =[y0]
        t_sum = [y0]
        for p in y[1:]:
            if abs(p-y0)<=2:
                t_sum.append(p)
                y0 = p
            else:
                
                lines.append(sum(t_sum)//len(t_sum))
                t_sum = [p]
                y0 = p
           
        lines.append(sum(t_sum)//len(t_sum))
        for point in lines:
            if abs(pre_y-point)<15:
                ymin = point
                # if xy == 'y':
                #     result.append([x,point])
                #     break
                # elif xy == 'x':
                result.append([x,(point+pre_y)//2])
                break
    return result

def process_kps(kps, mask):
    """
    根据曲线检测结果过滤误检的点和修正关键点的位置
    """
    # mask =  connect_lines(mask*255)//255
    kps = sorted(kps, key=lambda x:x[0])
    result = []
    # 处理Y轴偏移
    result = locate_offset(kps, mask, 'y')
    # result = locate_offset(kps, mask, 'x')
    return result