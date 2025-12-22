import numpy as np
import json
import os
import random

# from .Chart2019Dataset import Chart2019Dataset
from .Chart2019Dataset import Chart2019Dataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm

single_cls = False       # 文本检测分类



class Chart2019(Chart2019Dataset):


    def __init__(self, cfg, is_train, inputsize, transform=None,number_process=20):
        super().__init__(cfg, is_train, inputsize, transform)
        self.number_process = number_process
        self.db = self._get_db()
        self.cfg = cfg
        

    def get_data(self,start, end):

        print('building database...')
        gt_db = []
        height, width = self.shapes
        # random.shuffle(self.mask_list)
        t = 0
        for mask in tqdm(list(self.mask_list[start:end])):
           
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask), str(self.label_root)).replace(".png", ".txt")
            image_path = mask_path.replace(str(self.mask), str(self.img_root)).replace(".png", ".png")
            lane_path = mask_path.replace(str(self.mask), str(self.lane_root)).replace(".png", ".txt")
            try:
                if not os.path.exists(label_path):
                    gt = np.zeros((5, 5))
                else:
                    with open(label_path, 'r') as f:
                        # label = json.load(f)
                        lines = f.readlines()
                        gt = []
                        # gt = np.zeros((len(lines), 5))
                        for idx, line in enumerate(lines):
                            line = line.strip().split(',')
                            x1 = float(line[0])
                            y1 = float(line[1])
                            x2 = float(line[2])
                            y2 = float(line[3])
                            cls_id = line[-1]
                            if int(cls_id) >7 or int(cls_id)==3:
                                continue
                            # if int(cls_id) > 4:
                            #     cls_id = str(int(cls_id)-1)
                            box = convert((width, height), (x1, x2, y1, y2))
                            gt.append([float(cls_id), box[0], box[1], box[2], box[3]])
                            # gt[idx][0] = cls_id
                            # 
                            # gt[idx][1:] = list(box)
                        gt = np.array(gt)
            except Exception as e:
                print(e)
            # 关键点
            with open(lane_path, 'r') as f:
                # label = json.load(f)
                # print(lane_path)
                lines = f.readlines()
                point = np.zeros((len(lines), 3))
                for idx, line in enumerate(lines):
                    line = line.strip().split(',')
                    x1 = float(line[0])/width
                    y1 = float(line[1])/height
                    cls_id = line[-1]
                    point[idx][0] = int(cls_id)
                    # box = convert((width, height), (x1, x2, y1, y2))
                    point[idx][1:] = list([x1,y1])
            rec = [{
                'image': image_path,
                'label': gt,  # 文本框
                'mask': mask_path, # 曲线分割路径
                'lane': point  # 关键点分割
            }]

            gt_db += rec
        print('database build finish')
        return gt_db
    def mutil_process(self):
        result = []
        import multiprocessing as mp
        num_process =  self.number_process
        # image_list = file['images']
        # img_length = len(image_list)
        n = len(self.mask_list)//num_process 
        pool = mp.Pool(num_process)
        # results = pool.starmap(lineEXdata_process, [[i*n, i*n+n, i] for i in range(num_process+1)])
        result = pool.starmap(self.get_data, [[i*n, i*n +n ] for i in range(num_process+1)])
        pool.close()
        for data in result:
            result += data
        return result

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        return self.get_data(0, len(self.mask_list))

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
