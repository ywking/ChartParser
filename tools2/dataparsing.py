"""
time:2023-4-3
function: 解析图表中的数据，将关键点映射为原始数据，关键点匹配分类


"""

# import paddlehub as hub 
import time

# ocr = hub.Module(name="chinese_ocr_db_crnn_server")
import re



def OCR_parsing(img):
    """
    img，输入原图路径
    返回：x,y轴数据每个刻度对应数值大小

    """

    results = ocr.recognize_text(paths=[img],visualization=True,
                            box_thresh=0.5,           # 检测文本框置信度的阈值；
                            text_thresh=0.5)          # 识别中文文本置信度的阈值；
    return results



# from shapely import geometry
import requests
import json
import cv2
# from azure_ocr import result_ocr
import math
import numpy as np

def ocr_result(image_path):
    subscription_key = ""
    vision_base_url = "https://westus2.api.cognitive.microsoft.com/vision/v2.0/"
    ocr_url = vision_base_url + "read/core/asyncBatchAnalyze"
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'language': 'unk', 'detectOrientation': 'true'}
    # print(image_path)
    image_data = open(image_path, "rb").read()
    response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    op_location = response.headers['Operation-Location']
    analysis = {}
    while "recognitionResults" not in analysis.keys():
        time.sleep(3)
        binary_content = requests.get(op_location, headers=headers, params=params).content
        analysis = json.loads(binary_content.decode('ascii'))
    line_infos = [region["lines"] for region in analysis["recognitionResults"]]
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                word_infos.append(word_info)
    return word_infos

def check_intersection(box1, box2):
    if (box1[2] - box1[0]) + ((box2[2] - box2[0])) > max(box2[2], box1[2]) - min(box2[0], box1[0]) \
            and (box1[3] - box1[1]) + ((box2[3] - box2[1])) > max(box2[3], box1[3]) - min(box2[1], box1[1]):
        Xc1 = max(box1[0], box2[0])
        Yc1 = max(box1[1], box2[1])
        Xc2 = min(box1[2], box2[2])
        Yc2 = min(box1[3], box2[3])
        intersection_area = (Xc2-Xc1)*(Yc2-Yc1)
        return intersection_area/((box2[3]-box2[1])*(box2[2]-box2[0]))
    else:
        return 0

def if_inPoly(polygon, Points):
    """
    Determine if a point is inside a rectangle
    
    """
    line = geometry.LineString(polygon)
    point = geometry.Point(Points)
    polygon = geometry.Polygon(line)
    return polygon.contains(point)


def findlagend(word_infos, image):
    """
    word_infos:lists the result of OCR,which contains text,bbox
    retrun:legand:pixel mean value
    Classify the text to find the legend.
    """
    img = cv2.imread(image)
    res = {}
    # plot_area = cls_info[5][0:4]
    for wordinfo in word_infos:
        text = wordinfo['text']
        bbox = list(map(int,wordinfo['boundingBox']))
        h = bbox[5]-bbox[1]
        w = bbox[4]-bbox[0]
        if w<h or bbox[0] < 50:
            continue

        h_center = (bbox[1]+bbox[5])//2
        left = img[h_center-2:h_center+2,bbox[0]-30:bbox[0]-10,:]
        right = img[h_center-2:h_center+2,bbox[4]+10:bbox[4]+30,:]
        left = cv2.cvtColor(left, cv2.COLOR_RGB2HSV)
        right = cv2.cvtColor(right, cv2.COLOR_RGB2HSV)
        # left = 255 if len(left[left<250]) == 0 else left[left<250].mean()
        # right = 255 if len(right[right<255]) == 0 else right[right<250].mean()
        # if h_center>plot_area[1] and h_center<plot_area[3] \
        #     and bbox[0]>plot_area[0] and bbox[4]<plot_area[2] and text.isalnum():
            # cv2.imwrite('right.jpg',img[h_center-2:h_center+2,bbox[4]+10:bbox[4]+30,:])
            # if right.mean()
        res[text] = left if left.mean() < right.mean() else right
    return res


def Clusterpoint(image, keys, lagendcls):
    """
    点聚类:
    将关键点图像转为hsv颜色空间，根据颜色空间进行聚类
    lagendcls： legend文本分类
    keys： 文本检测结果

    """
    result = {}
    img = np.asarray(image)
    from sklearn.cluster import KMeans
    k = len(lagendcls)
    i = 0
    # for key in lagendcls.keys():
    #     result[i] = []
    #     i+= 1
    a = []
    plotarea_key = []
    for key in keys:
        point = key['bbox']
        # if point[0]>plot_area[0] and point[0]<plot_area[2] \
        #     and point[1]>plot_area[1] and point[1]<plot_area[3]:
        plotarea_key.append(key)
            # at = np.ones((4,20,3))*255
        at = img[int(point[1])-5:int(point[1])+5,int(point[0])-5:int(point[0])+5,:]
        hsv = cv2.cvtColor(at, cv2.COLOR_BGR2HSV)
        # for text in result.keys():
        # pointpixel = img[int(point[1]-2):int(point[1]+2),int(point[0]-10):int(point[0]+10),:].mean()
        a.append(hsv)
        # else:

    y_pred = KMeans(n_clusters=k,algorithm='elkan').fit_predict(a)
    indexs = {}
    result = {}
    for i,key in  enumerate(y_pred):
        result[key].append(a[i]) 
        if key not in indexs.keys():
            indexs[key] = []
        indexs[key].append(i)
    lines = {}
    mind = 999999
    line = 0
    for text in lagendcls:
        for key in result.keys():
            d = abs(np.array(result[key])-lagendcls[text]).min()
            if d <mind:
                mind = d
                line = key
        lines[text] = [plotarea_key[i] for i in indexs[line]]
        # draw_group(lines[text],image)
        mind = 999999
    return lines

def try_math(image_path, cls_info):
    title_list = [1, 2, 3]
    title2string = {}
    max_value = 1
    min_value = 0
    max_y = 0
    min_y = 1
    word_infos = result_ocr(image_path)
    legendcls = findlagend(word_infos, image_path,cls_info)
    for id in title_list:
        if id in cls_info.keys():
            predicted_box = cls_info[id]
            words = []
            for word_info in word_infos:
                word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
                if check_intersection(predicted_box, word_bbox) > 0.5:
                    words.append([word_info["text"], word_bbox[0], word_bbox[1]])
            words.sort(key=lambda x: x[1]+10*x[2])
            word_string = ""
            for word in words:
                word_string = word_string + word[0] + ' '
            title2string[id] = word_string
    if 5 in cls_info.keys():
        plot_area = cls_info[5]
        y_max = plot_area[1]
        y_min = plot_area[3]
        x_board = plot_area[0]
        dis_max = 10000000000000000
        dis_min = 10000000000000000
        for word_info in word_infos:
            word_bbox = [word_info["boundingBox"][0], word_info["boundingBox"][1], word_info["boundingBox"][4], word_info["boundingBox"][5]]
            word_text = word_info["text"]
            word_text = re.sub('[^-+0123456789.]', '',  word_text)
            word_text_num = re.sub('[^0123456789]', '', word_text)
            word_text_pure = re.sub('[^0123456789.]', '', word_text)
            if len(word_text_num) > 0 and word_bbox[2] <= x_board+10:
                dis2max = math.sqrt(math.pow((word_bbox[0]+word_bbox[2])/2-x_board, 2)+math.pow((word_bbox[1]+word_bbox[3])/2-y_max, 2))
                dis2min = math.sqrt(math.pow((word_bbox[0] + word_bbox[2]) / 2 - x_board, 2) + math.pow(
                    (word_bbox[1] + word_bbox[3]) / 2 - y_min, 2))
                y_mid = (word_bbox[1]+word_bbox[3])/2
                if dis2max <= dis_max:
                    dis_max = dis2max
                    max_y = y_mid
                    max_value = float(word_text_pure)
                    if word_text[0] == '-':
                        max_value = -max_value
                if dis2min <= dis_min:
                    dis_min = dis2min
                    min_y = y_mid
                    min_value = float(word_text_pure)
                    if word_text[0] == '-':
                        min_value = -min_value
        print(min_value)
        print(max_value)
        delta_min_max = max_value-min_value
        delta_mark = min_y - max_y
        delta_plot_y = y_min - y_max
        delta = delta_min_max/delta_mark
        if abs(min_y-y_min)/delta_plot_y > 0.1:
            print(abs(min_y-y_min)/delta_plot_y)
            print("Predict the lower bar")
            min_value = int(min_value + (min_y-y_min)*delta)

    return title2string, round(min_value, 2), round(max_value, 2), legendcls


def get_data(points,mask, legend):
    points = sorted(points, key=lambda x:(x[0],x[1]))
    res = [points[0]]
    for i in range(1,len(points)):
        k = (points[i][1]-points[1])


if __name__ == "__main__":

    result = OCR_parsing('/root/data1/ywj/images/123446.png')

    print(result)