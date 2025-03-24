import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout
import math
from lib.config import cfg

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

class Chart2019Dataset(Dataset):
    """
    A general Dataset for some common function
     训练文本检测，曲线关键点检测，曲线分割任务数据读取
    """
    def __init__(self, cfg, is_train, inputsize=512, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        # img_root = Path(cfg.DATASET.DATAROOT)
        # label_root = Path(cfg.DATASET.LABELROOT)
        # mask_root = Path(cfg.DATASET.MASKROOT)
        # lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        # self.img_root = img_root / indicator
        # self.label_root = label_root / indicator
        # self.mask_root = mask_root / indicator
        # self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.img_root = cfg.DATASET.DATAROOT
        self.mask = cfg.DATASET.MASK
        self.label_root = cfg.DATASET.LABELROOT
        self.lane_root = cfg.DATASET.LANEROOT
        self.instance_line = cfg.DATASET.INSTANCELINE
        self.mask_list = []
        if isinstance(cfg.DATASET.MASKROOT,list):
            for p in cfg.DATASET.MASKROOT:
                self.mask_list.extend((Path(p)/ indicator).iterdir())
        else:
            self.mask_list = (Path(self.mask_root) / indicator).iterdir()
        random.shuffle(self.mask_list)
        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
       
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # seg_label = cv2.imread(data["mask"], 0)
        if self.cfg.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])
        else:
            seg_label = cv2.imread(data["mask"], 0)
        # lane_label = cv2.imread(data["lane"], 0)
        #print(lane_label.shape)
        # print(seg_label.shape)                                  
        # print(lane_label.shape)
        # print(seg_label.shape)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0*r), int(h0*r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0*r), int(h0*r)), interpolation=interp)
            # lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        # resized_shape= (640,480)
        (img, seg_label), ratio, pad = letterbox((img, seg_label), resized_shape, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(shapes)
        
        det_label = data["label"]
        labels=[]
        # 关键点
        points = data['lane'].copy()
        points[:, 1] =  w * points[:, 1]  + pad[0]  # pad width
        points[:, 2] =  h* points[:, 2] + pad[1]  # pad height 
        # print(batch_hm.shape, points)
        # 关键点end
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
        if self.is_train:
            combination = (img, seg_label)
            (img, seg_label), labels,points = random_perspective(
                combination=combination,
                targets=labels,
                points=points,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR
            )
            #print(labels.shape)
            if random.random() < 0.4:
                augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)
            # img, seg_label, labels = cutout(combination=combination, labels=labels)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # # random left-right flip
            # lr_flip = True
            # if lr_flip and random.random() < 0.2:
            #     img = np.fliplr(img)
            #     seg_label = np.fliplr(seg_label)
            #     # lane_label = np.fliplr(lane_label)
            #     if len(labels):
            #         labels[:, 1] = w - labels[:, 1]
            #         labels[:,3] = w - labels[:, 3]
            #     if len(points):
            #         points[:,1] = w-points[:,1]

            # # # random up-down flip
            # # ud_flip = True
            # elif random.random() > 0.8:
            #     img = np.flipud(img)
            #     seg_label = np.flipud(seg_label)
            #     # lane_label = np.filpud(lane_label)
            #     if len(labels):
            #         labels[:, 2] = h - labels[:, 2]
            #         labels[:, 4] = h - labels[:, 4]
            #     if len(points):
            #         points[:,2] = h-points[:,2]
        
        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width
        # labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

        # # Normalize coordinates 0 - 1
        # labels[:, [2, 4]] /= img.shape[0]  # height
        # labels[:, [1, 3]] /= img.shape[1]  # width
        
    ## --------------------------------------绘制高斯热力图--------------------------------------------------------

        # point_dict = {'trick':0,'point':1}
        batch_hm        = np.zeros((int(h/4), int(w/4), cfg.key_points_class), dtype=np.float32)
        # batch_wh        = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg       = np.zeros((int(h/4), int(w/4), 2), dtype=np.float32)
        batch_reg_mask  = np.zeros((int(h/4), int(w/4)), dtype=np.float32)
        points[:, 1] = 1/4  * points[:, 1]  + pad[0]  # pad width
        points[:, 2] =  1/4* points[:, 2] + pad[1]  # pad height 
        # print(batch_hm.shape, points)
        for i in range(len(points)):
            cls_id  = int(points[i][0])
            ct = points[i][1:]
            if cfg.key_points_class == 1 and int(cls_id) !=2:
                continue
            elif cfg.key_points_class == 1 and int(cls_id) ==2:
                cls_id = 0
                
        
            # h, w, _ = img.shape
            # if h > 0 and w > 0:
            # radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            # radius = max(0, int(radius))
            radius = 5
            #-------------------------------------------------#
            #   计算真实框所属的特征点
            #-------------------------------------------------#
            # ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            # print(ct_int)
            #----------------------------#
            #   绘制高斯热力图
            #----------------------------#
    
            batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
            
            #---------------------------------------------------#
            #   计算宽高真实值
            #---------------------------------------------------#
            # batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
            #---------------------------------------------------#
            #   计算中心偏移量
            #---------------------------------------------------#
            
            
            batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
            #---------------------------------------------------#
            #   将对应的mask设置为1
            #---------------------------------------------------#
            batch_reg_mask[ct_int[1], ct_int[0]] = 1
     # --------------------------------------绘制高斯热力图结束--------------------------------------------------------
        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])

        if self.cfg.num_seg_class == 3:
            _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
            _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        else:
            _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)
        # _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
        # _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)
#        _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        # # seg1[cutout_mask] = 0
        # # seg2[cutout_mask] = 0
        
        # seg_label /= 255
        # seg0 = self.Tensor(seg0)
        if self.cfg.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        # seg1 = self.Tensor(seg1)
        # seg2 = self.Tensor(seg2)
        # lane1 = self.Tensor(lane1)
        # lane2 = self.Tensor(lane2)

        # seg_label = torch.stack((seg2[0], seg1[0]),0)
        if self.cfg.num_seg_class == 3:
            seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]),0)
            
        # lane_label = torch.stack((lane2[0], lane1[0]),0)
        # _, gt_mask = torch.max(seg_label, 0)
        # _ = show_seg_result(img, gt_mask, idx, 0, save_dir='debug', is_gt=True)
        

        target = [labels_out,torch.tensor(batch_hm),torch.tensor(batch_reg), torch.tensor(batch_reg_mask), seg_label]
        img = self.transform(img)

        return img, target, data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, seg_label,batch_hm, batch_reg, batch_reg_mask= [], [], [], [], []
        for i, l in enumerate(label):     # label=[labels_out, seg_label,batch_hm, batch_reg,batch_reg_mask]
            l_det, b_hm,b_reg, b_regmask, l_seg = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            batch_hm.append(b_hm)
            seg_label.append(l_seg)
            # batch_hm.append(b_hm)
            batch_reg.append(b_reg)
            batch_reg_mask.append(b_regmask)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(batch_hm, 0),\
             torch.stack(batch_reg, 0),  torch.stack(batch_reg_mask, 0), torch.stack(seg_label, 0)], paths, shapes

