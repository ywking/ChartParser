'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-12-22 13:57:22
LastEditors: yangwenjin 1183235940@qq.com
LastEditTime: 2025-12-22 16:08:57
FilePath: /YOLOP/lib/models/Chart_kp_line_text.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
from lib.config import cfg
sys.path.append(os.getcwd())

from lib.utils import initialize_weights,weights_init

from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv,Header,Point_Head
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from lib.models.horglass import PivotDetectin, Heatmap


from lib.models.DANet import AffinityAttention

# The lane line and the driving area segment branches without share information with each other and without link
ChartParser = [
[24, 29, 40],   #Det_out_idx, kp_det_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, AffinityAttention, [512,512,[5, 9, 13]]],
# [ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16         #Encoder


# 文本检测
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [cfg.det_class, [[16,9,12,23,25,15],
        [18,36,38,22,65,22],
        [32,58,19,106,119,25]], [128, 256, 512]]], #Detection head 24


# LineEX 文本 [[17,8,7,29,24,14],[18,25,32,17,40,23],[27,52,73,23,399,398]
# """ chart2019 lineEX anchor 9 [16,9,12,23,25,15],[18,36,38,22,65,22],[32,58,19,106,119,25]
# [5,7,8,8,14,8,8,19,24,8,8,26],
#                                         [10,34,44,13,14,48,86,9,14,67,6,171],
#                                         [25,55,168,9,35,64,262,11,10,376,379,16]], [128, 256, 512]
# """
# """
# lineEX
# [15,9,8,24,22,11,18,21,28,15,11,42],
# [22,33,41,19,33,26,14,74,57,23,32,50],
# [82,24,16,133,125,28,48,74,322,369,439,397]], [128, 256, 512]
# chart2019:[5,7,8,8,14,8,8,19,24,8,8,26],\
#                                         [10,34,44,13,14,48,86,9,14,67,6,171],
#                                         [25,55,168,9,35,64,262,11,10,376,379,16]
# """



# lineEX 图像元素 [211, 311] [284, 327] [385, 307] [315, 394] [414, 345] [353, 421] [439, 387] [449, 412] [461, 433]
#              目标检测               [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]]
#         文本检测框            [8,14,19,15,45,15], [23,35,17,55,26,53], [24,82,46,101,78,113]  
#            (640*480)          [8,6,10,8,20,8],[10,17,8,26,14,28],[19,47,39,55,232,9]
#  512x512 [12, 7] [8, 18] [7, 26] [10, 26] [12, 32] [14, 50] [23, 57] [171, 9] [34, 61]
# chart2019 [5, 7] [8, 8] [14, 8] [8, 19] [24, 8] [8, 26] [10, 34] [44, 13] [14, 48] [86, 9] [14, 67] [6, 171] [25, 55] [168, 9] [35, 64] [262, 11] [10, 376] [379, 16]
# 关键点检测[tricks, points]
[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #27
[ [-1,2], Concat, [1]],     #28         #Encoder
[-1,PivotDetectin,[cfg.key_points_class, False,3,5]], #29
# [-1, Heatmap, [cfg.key_points_class,False]],

# [ -1, Conv, [256, 128, 3, 1]],    #28
# [ -1, Upsample, [None, 2, 'nearest']],  #29
# [ -1, Conv, [128, 64, 3, 1]],    #30
# [ -1, BottleneckCSP, [64, 32, 1, False]],    #31
# [ -1, Upsample, [None, 2, 'nearest']],  #32
# [-1, Header, [32, 2, 3, 1]], #33
# [ -1, Conv, [8, 2, 3, 1]], #33 heatmap 预测
# [ -1, Conv, [8, 2, 3, 1]], #34  偏执预测


# 曲线分割 
[ 16, Conv, [256, 128, 3, 1]],   #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, BottleneckCSP, [128, 64, 1, False]],  #32
[ [-1,2], Concat, [1]],     #33         #Encoder
[ -1, Conv, [128, 64, 3, 1]],    #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ [-1,0], Concat, [1]],     #36        #Encoder
[ -1, Conv, [96, 64, 3, 1]],    #37
[ -1, BottleneckCSP, [64, 32, 1, False]],    #38
[ -1, Upsample, [None, 2, 'nearest']],  #39
[ -1, Conv, [32, 2, 3, 1]], #40  line segmentation head

]


class BuildModel(nn.Module):
    def __init__(self, block_cfg):
        super(BuildModel, self).__init__()
        pass


class ChartParserModel(nn.Module):
    def __init__(self, block_cfg):
        super(ChartParserModel, self).__init__()
        layers, save= [], []
        self.nc = cfg.det_class
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            # with torch.no_grad():
            # model_out = self.forward(torch.zeros(1, 3, s, s))
            # detects, _, _= model_out
            Detector.stride = torch.tensor([8, 16, 32])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchorsor the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        # weights_init(self)
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            # t = x.copy() if isinstance(x, list) else x.clone()
            x = block(x)
            if i in self.seg_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                if i==50:
                    out.append(m(x))
                    continue
                if isinstance(x, (tuple,dict,list)):
                    # for l in x:
                    #     # print(x[x==1])
                    #     # print(m(l)[m(l)==1])
                    out.append(x)
                else:
                    # print(x)
                    out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out
            
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs): 
    m_block_cfg = ChartParser
    model = ChartParserModel(m_block_cfg)
    
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from torchsummaryX  import summary
    # from torchstat import stat
    from torchvision import transforms
    import cv2
    model = get_net2(False)
    img = cv2.imread('Chart2019512x512/image/val/0.png')
    print(model)
    # img = transforms.ToTensor()(img)
    # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0)
    # input_ = torch.randn((1, 3, 512, 512))
    # gt_ = torch.rand((1, 2, 256, 256))
    # metric = SegmentationMetric(2)
    # out = model(img)
    # summary(model,torch.rand((1, 3, 512, 512)))
    # stat(model,input_size=(3,512,512))
 
