'''
Author: yangwenjin 1183235940@qq.com
Date: 2025-12-22 15:32:16
LastEditors: yangwenjin 1183235940@qq.com
LastEditTime: 2025-12-22 15:32:21
FilePath: /ChartParser-master/lib/dataset/convert.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

id_dict_single = {'chart_title':0,'axis_title':1,
            'tick_label':2,'legend_label':3,
            'mark_label':4,'value_label':5,
            'legend_title':6,'other':7,'tick_grouping':8}
id_dict = {'chart_title':0,'axis_title':1,
            'tick_label':2,'legend_label':3,
            'mark_label':4,'value_label':5,
            'legend_title':6,'other':7,'tick_grouping':8}
# id_dict = {'car': 0, 'bus': 1, 'truck': 2}

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
