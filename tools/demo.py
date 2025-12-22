import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import sys
sys.path.append('./')
from utils import checkifbackground
# sys.path.append('DeepRule')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
from tools2.decode_heatmap import decode_hm,show_heatmap,decode_line
# from tools2.azure_ocr import result_ocr
from tools2.dataparsing import *

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

# from tools2.azure_ocr import result_ocr

# from tools2.kp_legend_grouping import LineMatchData
from lib.utils.line_process import process_kps
# from DeepRule.test_pipe_type_cloud import chartocrpivot, chartocr_det

label_dict = {0:'Y_title',  1:'chart_title', 2:'x_title',
              4:'x_trick', 5:'y_trick', 6:'legend_marker', 7:'legend_txt'}



def chartocr_legend_plotarea(img):
    det = np.array(chartocr_det(img))
    legendbox = det[det[...,5]==0]
    inplot = det[det[...,5]==5]
    outplot = det[det[...,5]==4]
    return legendbox,inplot, outplot

def cac_IOU(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA ) * max(0, yB - yA )

    boxAArea = abs((boxA[2] - boxA[0] ) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0] ) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def chart_parsering(img, device='cuda'):
    img = cv2.imread(img)
    print('开始图像处理')
    model = get_net2(cfg)
    # model = get_net_kpline(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    checkpoint_dict = {key.replace('module.model.', 'model.'):value for key,value in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    model.eval()
    # print(img.shape)
    h,w,c = img.shape
    ori_img = img.copy()
    img = cv2.resize(img, (512,512))
    img_t = img.copy()
    img = transform(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    c,h,w =img.shape
    # img_name = os.path.basename(path)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t = cv2.imwrite('./t.png', ori_img)
    with torch.no_grad():
        out = model(img)
    if len(out)==3:
        det_out, hm,ll_seg_out = out
    elif len(out) == 2:
        det_out = []
        hm,ll_seg_out= model(img)
    else:
        det_out = []
        hm = model(img)[0]
        ll_seg_out = hm[-1]['seg']
    decodehm = decode_hm(hm[-1]['hm'],hm[-1]['reg'],0.3,"cpu") ## 归一化结果
    x_cords = np.array((decodehm[0][:,0]*4).cpu()).astype(int) # 坐标点为128*128的坐标点， 放大四倍还原为512*512的
    y_cords =  np.array((decodehm[0][:,1]*4).cpu()).astype(int)
    pred_kp = []
    # timage = cv2.imread
    for x,y in zip(x_cords,y_cords):
        if(checkifbackground((x,y),img_t)):
            continue
        pred_kp.append((x,y))

    if len(det_out) >0:
        inf_out, _ = det_out
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)

        det=det_pred[0]
    ll_predict = ll_seg_out
    # ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')

    _, ll_seg_mask = torch.max(ll_predict, 1)
    # cv2.imwrite(f'tools/seg/chart2019/{i}_fused.png',ll_predict.squeeze().cpu().numpy()*255)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
    det_result = []

    if len(det_out):
        det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img.shape).round()
        for *xyxy,conf,cls in reversed(det):
            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
            det_result.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls), conf])
    return det_result, ll_seg_mask, pred_kp


# from paddleocr import PaddleOCR, draw_ocr
# ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)

def single_img_demo(img, device='cpu'):
    print('开始图像处理')
    # img.save('t.png')
    # ori_path = 't.png'
    name = os.path.basename(img)
    # print(img)
    img =cv2.imread(img)

    # img = np.array(img)
    ori_img = img.copy()
    ori_h, ori_w,_ = img.shape
    model = get_net(cfg)
    print(device)
    
    checkpoint = torch.load(opt.weights, map_location= device, )
    checkpoint_dict = {key.replace('module.model.', 'model.'):value for key,value in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint_dict)
    model = model.to(device)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    model.eval()
    # print(img.shape)
    h,w,c = img.shape
    ori_img = img.copy()
    img = cv2.resize(img, (512,512))
    img_t = img.copy()
    img = transform(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    c,h,w =img.shape
    # img_name = os.path.basename(path)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t = cv2.imwrite('./t.png', ori_img)
    with torch.no_grad():
        out = model(img)
    if len(out)==3:
        det_out, hm,ll_seg_out = out
    elif len(out) == 2:
        det_out = []
        hm,ll_seg_out= model(img)
    else:
        det_out = []
        hm = model(img)[0]
        ll_seg_out = hm[-1]['seg']

    hm[-1]['hm'] = hm[-1]['hm'].sigmoid()
    decodehm = decode_hm(hm[-1]['hm'],hm[-1]['reg'],0.3,"cpu") ## 归一化结果
    x_cords = np.array((decodehm[0][:,0]/128*512).cpu()).astype(int) # 坐标点为128*128的坐标点， 放大四倍还原为512*512的
    y_cords =  np.array((decodehm[0][:,1]/128*512).cpu()).astype(int)
    pred_kp = []

    ll_predict = ll_seg_out
    # ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')

    _, ll_seg_mask = torch.max(ll_predict, 1)
    # cv2.imwrite(f'tools/seg/chart2019/{i}_fused.png',ll_predict.squeeze().cpu().numpy()*255)
    ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
    # ll_seg_mask = cv2.resize(ll_seg_mask, (ori_w, ori_h))
    # timage = cv2.imread
    img_det = show_seg_result(img_t, ll_seg_mask, 0,0,opt.save_dir+'seg/', is_ll=True)
    kpimg = cv2.resize(ori_img, (512, 512))
    for x,y in zip(x_cords,y_cords):
        # if(checkifbackground((x,y),ori_img)):
        #     continue
        pred_kp.append((x,y))
        cv2.circle(img_det,(int(x),int(y)),4,(0,0,255),-1)
        cv2.circle(kpimg,(int(x),int(y)),4,(0,0,255),-1)
    # pred_kp = process_kps(pred_kp, ll_seg_mask)
    if len(det_out) >0:
        inf_out, _ = det_out
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)

        det=det_pred[0]

    # Lane line post-processing
    # ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
    # ll_seg_mask = connect_lane(ll_seg_mask)
    # img_ll = cv2.imread(path)
    
    det_result = []

    if len(det_out):
        det[:,:4] = scale_coords(img.shape[2:],det[:,:4],(512,512)).round()
        for *xyxy,conf,cls in reversed(det):
            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
            det_result.append([int(xyxy[0]/512*ori_w), int(xyxy[1]/512*ori_h),\
                                int(xyxy[2]/512*ori_w), int(xyxy[3]/512*ori_h), int(cls), conf])
    print('检测图像中的文本')


    # ocr_result = result_ocr(ori_path)
    # ocr_result =  ocr.ocr(ori_img, cls=False)[0]
    det_dict = []
    # 为每个检测框分配文本
    for det  in det_result:
        x1, y1, x2, y2, label, conf = det
        if conf<0.5:
            continue
        maxiou = 0
        maxiou_text = ''
        # for result in ocr_result:
        #     text = result[1][0]
        #     box = result[0]
        #     # print(box, result)
        #     if abs(box[1][0]-box[0][0]) <5:
        #         new_box = [box[1][0], box[1][1], box[3][0], box[3][1]]
        #     else:
        #         new_box = [box[0][0], box[0][1], box[2][0], box[2][1]]
        #     iou = cac_IOU([x1, y1, x2, y2], new_box)
        #     if iou>maxiou:
        #         maxiou = iou
        #         maxiou_text = text

        det_dict.append([[x1, y1, x2, y2], label_dict[label], float(conf.cpu())])


    # lineprocess = LineMatchData(det_dict, ori_img)
    # lines, legend_marker_text = lineprocess.kp_group(pred_kp, './t.png')
    lines = {}
    # kpimg = ori_img.copy()
    dataimg = ori_img.copy()
    for key, value in lines.items():
        for point in lines[key]:
            box, data = point
            cv2.circle(img_det,(int(box[0]),int(box[1])),3,(255,0,0),-1)
            cv2.circle(kpimg,(int(box[0]),int(box[1])),3,(255,0,0),-1)
            cv2.circle(dataimg,(int(box[0]),int(box[1])),3,(255,0,0),-1)
            cv2.putText(dataimg,str(round(data[1],2)), (int(box[0]),int(box[1])-10),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,255),1)
    
    cv2.imwrite(f'results/kp_seg_det/kp_{name}', img_det)
    cv2.imwrite(f'results/kp_{name}', kpimg)
    cv2.imwrite(f'results/kpdata_{name}', dataimg)
    a = [(255,182,193),(255,240,245), (	0,0,255), (148,0,211),
         (0,191,255), (64,224,208), (50,205,50), (124,252,0), (255,255,224), (255,69,0) ]
    for det in det_result:
        x1,y1,x2,y2,c,_ =det
        cv2.rectangle(kpimg, (int(x1),int(y1)), (int(x2),int(y2)), a[c], 2)
    cv2.imwrite('t.png', kpimg)
    # 绘
    return det_dict, lines

def detect(cfg, opt, chartocr=False):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    if os.path.exists(opt.save_dir+'points/'):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir+'points/')  # make new dir

    if os.path.exists(opt.save_dir+'seg/'):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir+'seg/')  # make new dir

    if os.path.exists(opt.save_dir+'deteciton/'):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir+'detection/')  # make new dir
    half = device.type == 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = get_net_kp_seg(cfg)
    model = get_net(cfg)
    # model = get_net_kpline(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    # if half:
    #     model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    # img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    pred_time = 0
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):

        img_t = img.copy()
        img = transform(img).to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        c,h,w =img.shape
        img_name = os.path.basename(path)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        ## chartocr 获取边界框
        if chartocr :
            legendbox,inplot, outplot = chartocr_legend_plotarea(path[i])
        ori_img = cv2.imread(path.replace('image', 'ori'))
        # ocr_result = result_ocr(path.replace('image', 'ori'))
        ocr_result = {}
        t1 = time_synchronized()
        out = model(img)
        pred_time += time_synchronized() - t1
        if len(out)==3:
            det_out, hm,ll_seg_out = out
        elif len(out) == 2:
            det_out = []
            hm,ll_seg_out= model(img)
        else:
            det_out = []
            hm = model(img)[0]
            ll_seg_out = hm[-1]['seg']
        decodehm = decode_hm(hm[-1]['hm'],hm[-1]['reg'],0.3,"cuda") ## 归一化结果
        # a= hm.cpu()

        # t = cv2.imread(path,0)
        # python(img_hm,heatmap[-1]['hm'][i],save_dir,False,epoch,i)
        # show_heatmap(t.copy(),hm[-1]['hm'][0],opt.save_dir+'points/',num=i)
        # 绘制对应点：
        # img_t = cv2.imread(path)
        img_h,img_w,_ = img_t.shape

        # 曲线分组
        # word_infos = result_ocr(path.replace('image','ori'))
        # legendcls = findlagend(word_infos, path.replace('image','ori'))
        # lines = Clusterpoint(img_t,decodehm[0],legendcls)
        # img_t = cv2.resize(img_t,(128,128))


        # 去除背景噪声点
        ll_predict = ll_seg_out
        # ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')

        _, ll_seg_mask = torch.max(ll_predict, 1)
        # cv2.imwrite(f'tools/seg/chart2019/{i}_fused.png',ll_predict.squeeze().cpu().numpy()*255)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        x_cords = np.array((decodehm[0][:,0]*4).cpu()).astype(int) # 坐标点为128*128的坐标点， 放大四倍还原为512*512的
        y_cords =  np.array((decodehm[0][:,1]*4).cpu()).astype(int)
        pred_kp = []
        # timage = cv2.imread
        points = [[x,y] for x,y in zip(x_cords,y_cords) ]
        # pred_kp = process_kps(points, ll_seg_mask)
        # for x,y in zip(x_cords,y_cords):
        #     if(checkifbackground((x,y),img_t)):
        #         continue
        #     pred_kp.append((x,y))
        for box in pred_kp:
             cv2.circle(img_t,(int(box[0]),int(box[1])),2,(0,0,255),-1)
        # cv2.imwrite(f'inference/output/chart2019/points/{i}.png', img_t)
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        # det = []
        if len(det_out) >0:
            inf_out, _ = det_out
            inf_time.update(t2-t1,img.size(0))

            # Apply NMS
            t3 = time_synchronized()
            det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
            t4 = time_synchronized()

            nms_time.update(t4-t3,img.size(0))
            det=det_pred[0]

            save_path = str(opt.save_dir +'/detection/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        # da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        # _, da_seg_mask = torch.max(da_seg_mask, 1)
        # da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
        # point
        # line_point = decode_line(ll_seg_out.cpu(),0.5)


        # Lane line post-processing
        # ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        # ll_seg_mask = connect_lane(ll_seg_mask)
        img_ll = cv2.imread(path)
        img_det = show_seg_result(img_ll, ll_seg_mask, i,0,opt.save_dir+'seg/', is_ll=True)
        det_result = []

        if len(det_out):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
                det_result.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls), conf])

                # if int(cls) ==6:
                #     legends.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])
                # elif int(cls) == 7:
                #     legends_text.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])
                # elif int(cls) == 5:
                #     y_label.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])
            if dataset.mode == 'images':
                cv2.imwrite(save_path,img_det)

            
        ## 曲线分组
        # lineprocess = LineMatchData(det_result, ori_img, ocr_result)
        # lines, legend_marker_text = lineprocess.kp_group(pred_kp, path)
        # # 绘制数据点
        # img = lineprocess.plot_line_point(lines)
        # cv2.imwrite(opt.save_dir+'points/' +img_name, img)
        # img = lineprocess.plot_line_point(lines, True, True)
        # cv2.imwrite(opt.save_dir+'points/instance_'+img_name, img)
        # print(lines)



    print('model predict(%.3fs)' % (pred_time))
    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))



parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='weights/epoch-19.pth', help='model.pth path(s)')
parser.add_argument('--source', type=str, default='material/image/val/', help='source')  # file/folder   ex:inference/images
parser.add_argument('--img-size', type=int, default=(512,512), help='inference size (pixels)')
parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--save-dir', type=str, default='results/', help='directory to save results')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
opt = parser.parse_args(args=[])

if __name__ == '__main__':
    import glob 
    path = 'material/image/val/'
    imgs = glob.glob('ICPR/ori/val/*.png')
    # imgs = ['1.png','6.png', '101.png', '100.png']
    imgs = ['pictures/20.png']
    with torch.no_grad():
        for img in imgs:
            # img = path+img
            # img = 'instance_segmention/Mask2Former/imgclass_test/微信图片_20231211144924.png'
            # print(img)
            # if '105.png' not in img:
            #     continue
            # img = cv2.imread(img)
            single_img_demo(img)
            # break
