import os
from yacs.config import CfgNode as CN
import random
def random_int_list(start, stop, length):

    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))

    length = int(abs(length)) if length else 0

    random_list = []

    for i in range(length):

        random_list.append(random.randint(start, stop))

    return random_list


_C = CN()

_C.LOG_DIR = 'abuly/encoder_Det/'
_C.GPUS = (0,1,2,3)     
_C.WORKERS = 20
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 40
_C.AUTO_RESUME =False       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 2  # 线类别：曲线和背景
_C.key_points_class = 1 # 关键点类型 
_C.det_class = 8
_C.DTYPE= 'float32'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']
# _C.MODEL.PRETRAINED = "runs/center_threeHead_radio_3_lineEXdata_offset1x_8cat_anchor9/Chart2019/_2023-06-25-20-59/epoch-9.pth"
_C.MODEL.PRETRAINED = ""
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.IMAGE_SIZE = [512,512]  # idth * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)


# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.05  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
_C.LOSS.LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.2 # lane line iou loss gain
import glob

# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = 'image'       # the path of images folder
_C.DATASET.LABELROOT = 'det_text'      # the path of det_annotations folder
# material = ["material/mask" for i in range(2)]
# chartoccr = ["ChartOCR512x512/mask"]
chart2019 = ["Chart2019512x512_lineEX/mask"]
# icpr = [ 'ICPR/mask' for i in range(15) ]
# fs = ['Fsdaata/mask' for i in range(2)]
# lineEX_idex = [0, 3, 4, 5, 7, 8, 12, 14, 18, 19, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 
#                37, 40, 41, 42, 44, 46, 47, 48, 49, 51, 52, 54, ]
# lineex = [f'LineEXdata512x512/mask/{i}' for i in lineEX_idex]
#  material +  chart2019+ icpr +lineex  
# _C.DATASET.MASKROOT =         material       # the path of da_seg_annotations folder
# lineEX_idex = set([ random.randint(0,65) for i in range(50)])
# material +  chart2019+ icpr +lineex  + fs +chartoccr
_C.DATASET.MASKROOT =  chart2019   # 训练关键点检测
# _C.DATASET.MASKROOT += [f'LineEXdata512x512/mask/{i}' for i in lineEX_idex] 
_C.DATASET.MASK = 'mask'
_C.DATASET.LANEROOT = 'points'               # the path of ll_seg_annotations folder
_C.DATASET.INSTANCELINE = 'instance_mask'     
_C.DATASET.DATASET = 'Chart2019'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'png'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [512,512] #(h,w)

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add


# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 1000

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 12
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
_C.TRAIN.SEG_ONLY = False          
_C.TRAIN.DET_ONLY = False           
_C.TRAIN.ENC_SEG_ONLY = False      
_C.TRAIN.ENC_DET_ONLY = False       
_C.TRAIN.ENC_KP_ONLY = False                
_C.TRAIN.SEG_DET = False            
_C.TRAIN.KP_DET = True  
_C.TRAIN.KP = False  






_C.TRAIN.PLOT = True                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 12
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True
_C.TEST.NMS_CONF_THRESHOLD  = 0.6
_C.TEST.NMS_IOU_THRESHOLD  = 0.4
_C.datamode = 'chartocr' # LineEX, chart2019 chartocr, material


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir
    if args.datamode:
        cfg.datamode = args.datamode
    if args.conf_thres:
        cfg.TEST.NMS_CONF_THRESHOLD = args.conf_thres

    if args.iou_thres:
        cfg.TEST.NMS_IOU_THRESHOLD = args.iou_thres
    


    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )
    #
    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()
