import time
from LineEX.modules.KP_detection.utils import checkifbackground, metric
from lib.core.evaluate import ConfusionMatrix,SegmentationMetric
from lib.core.general import non_max_suppression,check_img_size,scale_coords,xyxy2xywh,xywh2xyxy,box_iou,coco80_to_coco91_class,plot_images,ap_per_class,output_to_target
# from lib.utils.utils import time_synchronized
from lib.utils import plot_img_and_mask,plot_one_box,show_seg_result
from lib.utils.decode_heatmap import show_heatmap
import torch
from threading import Thread
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json
import random
import cv2
import os
import math
from torch.cuda import amp

from lib.utils.decode_heatmap import decode_hm,show_heatmap,decode_line
from lib.utils.azure_ocr import result_ocr
import sys
# sys.path.append('/root/data1/YOLOP/LineEX')
from LineEX.modules.KP_detection.util import *
from tqdm import tqdm
def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
          writer_dict, logger, device, rank=-1):
    """
    train for one epoch

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return total_loss, head_losses
    - writer_dict:
    outputs(2,)
    output[0] len:3, [1,3,32,32,85], [1,3,16,16,85], [1,3,8,8,85]
    output[1] len:1, [2,256,256]
    output[2] len:1, [2,256,256]
    output[3] len:1, [2,256,256]
    target(2,)
    target[0] [1,n,5]
    target[1] [2,256,256]
    target[2] [2,256,256]
    Returns:
    None

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    for i, (input, target, paths, shapes) in enumerate(train_loader):
        if i>2:
            break
        input =  torch.tensor(input,dtype=torch.float32).to(device)
        optimizer.zero_grad()
        intermediate = time.time()
        #print('tims:{}'.format(intermediate-start))
        num_iter = i + num_batch * (epoch - 1)

        if num_iter < num_warmup:
            # warm up
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
            xi = [0, num_warmup]
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])

        data_time.update(time.time() - start)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
        # with amp.autocast(enabled=device != 'cpu'):
        outputs = model(input)
        total_loss, head_losses = criterion(outputs, target, shapes,model)
            # print(head_losses)

        # compute gradient and do update step
        
        # scaler.scale(total_loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        total_loss.backward()
        optimizer.step()

        if rank in [-1, 0]:
            # measure accuracy and record loss
            losses.update(total_loss.item(), input.size(0))

            # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
            #                                  target.detach().cpu().numpy())
            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - start)
            end = time.time()
            if i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                # writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

from lib.config import cfg


def get_pivort(pred, image_path):
   
    
    ground_kp = []
    ground_lines = []
    timage = cv2.imread(image_path)
    h,w,c = timage.shape
    gt_point = image_path.replace('image', 'points').replace('png', 'txt')
    with open(gt_point, 'r') as f:
        lines = f.readlines()
        for kp in lines:
            if float( kp.strip().split(',')[-1]) != 2:
                continue
            kp = list(map(int,list(map(float, kp.strip().split(',')[:-1]))))
            ground_kp.extend(kp)
    pred_kp = []
    g_kp = []
    g_x_cords = np.array(ground_kp[0::2]).astype(int)
    g_y_cords = np.array(ground_kp[1::2]).astype(int)

    if not isinstance(pred,list):
        pred = pred.cpu().numpy()
        x_cords = (pred[:,0]*w).astype(int)
        y_cords = (pred[:,1]*h).astype(int)
        for x,y in zip(x_cords,y_cords):
            if(checkifbackground((x,y),timage)):
                continue

            pred_kp.append((x,y))
    
    for x,y in zip(g_x_cords,g_y_cords):
        g_kp.append((x,y))

    # chartocr_kps = get_chartocr_kp(CHART_OCR_KP,image_name,dataset=dataset)
    
    return pred_kp,g_kp

from lib.utils.line_process import  process_kps

def validate(epoch,config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
    """
    validata

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return 
    - writer_dict: 

    Return:
    None
    """
    # setting
    max_stride = 32
    weights = None
    losses = AverageMeter()
    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir) and rank in [-1,0]:
        print(save_dir, rank)
        os.mkdir(save_dir)

    # print(save_dir)
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE] #imgsz is multiple of max_stride
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU 
    # test_batch_size = 40
    training = False
    is_coco = False #is coco dataset
    save_conf=False # save auto-label confidences
    verbose=False
    save_hybrid=False
    log_imgs,wandb = min(16,100), None

    nc = cfg.det_class
    iouv = torch.linspace(0.5,0.95,10).to(device)     #iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen =  0 
    confusion_matrix = ConfusionMatrix(nc=model.nc) #detector confusion matrix
    # da_metric = SegmentationMetric(config.num_seg_class) #segment confusion matrix    
    ll_metric = SegmentationMetric(2) #segment confusion matrix

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()
    
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    
    losses = AverageMeter()

    # 曲线精度评估
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    ll_prec_seg = AverageMeter()
    ll_rec_seg = AverageMeter()

    T_inf = AverageMeter()
    T_nms = AverageMeter()

    # 关键点
    pivo_pre_det = AverageMeter()
    pivo_rec_det = AverageMeter()
    pivo_f1_det = AverageMeter()
   

    # switch to train mode
    model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    oks_p1, oks_r1,  oks_f1, oks_p2, oks_r2,  oks_f2 = [], [], [], [], [], []
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        # if batch_i >5:
        #     break
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape    #batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0]

            # t = time_synchronized()
            out = model(img)
            # out = out[1:]
            if len(out)>=3:
                det_out, heatmap, ll_seg_out = out
                inf_out,train_out = det_out
                temp = [out[0][1]]
                temp.extend(out[1:])
            elif len(out)==2:
                det_out = []
                inf_out,train_out = [],[]
                heatmap, ll_seg_out = out
                temp = out
            else:
                det_out = []
                inf_out,train_out = [],[]
                heatmap = out[0]
                ll_seg_out = heatmap[-1]['seg']
                temp = out
            # t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf/img.size(0),img.size(0))
            heatmap[-1]['hm'] = heatmap[-1]['hm'].sigmoid()
            # 关键点评估：

            hm = heatmap
            pivots = decode_hm(hm[-1]['hm'],hm[-1]['reg'],0.3,False if device=='cpu' else True)
            #lane line segment evaluation
            _,ll_predict=torch.max(ll_seg_out, 1)
             
            _,ll_gt=torch.max(target[-1], 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            for i, (pivot, path) in enumerate(zip(pivots, paths)):
                pred_kp, gt_kp = get_pivort(pivot, path) 
                # pred_kp = process_kps(pred_kp, ll_predict[i])
                recall_oks,precision_oks,F1_oks = metric(pred_kp,gt_kp,[], path,relaxed = False)
                pivo_pre_det.update(precision_oks, 1)
                pivo_rec_det.update(recall_oks, 1)
                pivo_f1_det.update(F1_oks, 1)
            # oks_p1.append(precision_oks)
            # oks_r1.append(recall_oks)
            # oks_f1.append(F1_oks)
            

            loss_all = []
            
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()
            line_pre, line_rec, line_f = ll_metric.linePreRecall()
             
            ll_acc_seg.update(ll_acc,img.size(0))
            ll_IoU_seg.update(ll_IoU,img.size(0))
            ll_mIoU_seg.update(ll_mIoU,img.size(0))
            ll_prec_seg.update(line_pre, img.size(0))
            ll_rec_seg.update(line_rec,img.size(0) )
            # temp = [out[0][1]]
            # temp.extend(out[1:])
            
            total_loss, head_losses = criterion(temp, target, shapes,model)   #Compute loss
            #
            loss_all.append(total_loss.item()/img.size(0))
            losses.update(total_loss.item(), img.size(0))
            if batch_i % config.PRINT_FREQ == 0:
                writer = writer_dict['writer']
                global_steps = writer_dict['valid_global_steps']
                writer.add_scalar('val_loss', losses.val, global_steps)
                # writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['valid_global_steps'] = global_steps + 1
            #NMS
            if   len(inf_out) and config.datamode!='chartocr':
                # t = time_synchronized()
                target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            
                output = non_max_suppression(inf_out, conf_thres= config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
            else:
                output = []
            #output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)
            #output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRES, iou_thres=config.TEST.NMS_IOU_THRES)
            # t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0),img.size(0))

            if config.TEST.PLOTS and rank in [-1,0]:
                if batch_i == 0:
                    for i in range(test_batch_size):

                        img_ll = cv2.imread(paths[i])
                        ll_seg_mask = ll_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
                        
                        ll_gt_mask = target[-1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_gt_mask = torch.max(ll_gt_mask, 1)

                        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                        ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()
                        # seg_mask = seg_mask > 0.5
                        # plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)
                        img_ll1 = img_ll.copy()
                        _ = show_seg_result(img_ll, ll_seg_mask, i,epoch,save_dir, is_ll=True)
                        _ = show_seg_result(img_ll1, ll_gt_mask, i, epoch, save_dir, is_ll=True, is_gt=True)
                        # 绘制热图
                        img_hm = cv2.imread(paths[i],0)
                        show_heatmap(img_hm,heatmap[-1]['hm'][i],save_dir,False,epoch,i)
                        show_heatmap(img_hm,target[1][i],save_dir,True,epoch,i)
                        img_det = cv2.imread(paths[i])
                        name = os.path.basename(paths[i])
                        img_gt = img_det.copy()
                        if len(output):
                            det = output[i].clone()
                            if len(det):
                                det[:,:4] = scale_coords(img[i].shape[1:],det[:,:4],img_det.shape).round()
                            for *xyxy,conf,cls in reversed(det):
                                #print(cls)
                                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
                            cv2.imwrite(save_dir+"/batch_{}_{}_det_pred_{}".format(epoch,i,name),img_det)

                            labels = target[0][target[0][:, 0] == i, 1:]
                            # print(labels)
                            labels[:,1:5]=xywh2xyxy(labels[:,1:5])
                            if len(labels):
                                labels[:,1:5]=scale_coords(img[i].shape[1:],labels[:,1:5],img_gt.shape).round()
                            for cls,x1,y1,x2,y2 in labels:
                                #print(names)
                                #print(cls)
                                label_det_gt = f'{names[int(cls)]}'
                                xyxy = (x1,y1,x2,y2)
                                plot_one_box(xyxy, img_gt , label=label_det_gt, color=colors[int(cls)], line_thickness=2)
                            cv2.imwrite(save_dir+"/batch_{}_{}_det_gt_{}".format(epoch,i, name),img_gt)

        # Statistics per image
        # output([xyxy,conf,cls])
        # target[0] ([img_id,cls,xyxy])
        target[0][:, 1][target[0][:, 1]==7] = 5 # legendtext
        target[0][:, 1][target[0][:, 1]==6] = 4 # marker
        target[0][:, 1][target[0][:, 1]==4] = 3 # ticks
        target[0][:, 1][target[0][:, 1]==5] = 3 # ticks
            
        if  len(output):
            for si, pred in enumerate(output):
                pred[:, 5][ (pred[:, 5] == 4) | (pred[:, 5] == 5)] = 3
                pred[:, 5][ pred[:, 5] == 6] = 4
                pred[:, 5][ pred[:, 5] == 7] = 5
                # if config.datamode=='FS':
                #     pred = pred[( pred[:, 5] != 4) & (pred[:, 5] !=5) & (pred[:, 5]<8)]
                # elif config.datamode=='material' or config.datamode=='ICPR':
                #     pred = pred[(pred[:, 5]<8)]
                # elif config.datamode=='Chart2019':
                    # pred = pred[(pred[:, 5] !=6)]
                labels = target[0][target[0][:, 0] == si, 1:]     #all object in one image 

                nl = len(labels)    # num of object
                # labels = labels
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Append to text file
                if config.TEST.SAVE_TXT:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # W&B logging
                if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                "class_id": int(cls),
                                "box_caption": "%s %.3f" % (names[cls], conf),
                                "scores": {"class_score": conf},
                                "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

                # Append to pycocotools JSON dictionary
                if config.TEST.SAVE_JSON:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                    'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                    'bbox': [round(x, 3) for x in b],
                                    'score': round(p[4], 5)})


                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    if config.TEST.PLOTS:
                        confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                    # Per target class
                    for cls in torch.unique(tcls_tensor):                    
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            # n*m  n:pred  m:label
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # print(ious)
                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if config.TEST.PLOTS and batch_i < 3:
            f = save_dir +'/'+ f'test_batch{batch_i}_labels.jpg'  # labels
            #Thread(target=plot_images, args=(img, target[0], paths, f, names), daemon=True).start()
            f = save_dir +'/'+ f'test_batch{batch_i}_pred.jpg'  # predictions
            #Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

    # Compute statistics
    # stats : [[all_img_correct]...[all_img_tcls]]
 
    mp, mr,mf, map50,map70, map =0, 0, 0, 0, 0, 0
    nt = torch.zeros(1)
   
    if len(stats) and config.datamode != 'chartocr':
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

        map70 = None
        map75 = None
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
            ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, mf, map50, map70, map = p.mean(), r.mean(), f1.mean(), ap50.mean(), ap70.mean(),ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        if rank in [0,-1]:
            pf = '%20s' + '%12.3s' * 8  # print format
            print(pf  % ('all', 'seen', 'sum', 'mp', 'mr','mf', 'map50','map70', 'map'))
            pf = '%20s' + '%12.3g' * 8  # print format
            print(pf % ('all', seen, nt.sum(), mp, mr,mf, map50,map70, map))
            #print(map70)
            #print(map75)
            pf = '%20s' + '%12.3s' * 8 
            print(pf  % ('label', 'seen', 'sum', 'p', 'r','f1', 'map50','map70', 'map'))
            pf = '%20s' + '%12.3g' * 8 
        # Print results per class
        if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats) and rank in [0, -1]:
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], f1[i], ap50[i], ap70[i], ap[i]))
        # Print speeds
        t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
        if not training and rank in [0, -1]:
            print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

        # Plots
        if config.TEST.PLOTS:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            if wandb and wandb.run:
                wandb.log({"Images": wandb_images})
                wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

        # Save JSON
        if config.TEST.SAVE_JSON and len(jdict):
            w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
            anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
            pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
            print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
            with open(pred_json, 'w') as f:
                json.dump(jdict, f)

            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api
                eval = COCOeval(anno, pred, 'bbox')
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]  # image IDs to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                print(f'pycocotools unable to run: {e}')

        # Return results
        if not training and rank in [0,-1]:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
            print(f"Results saved to {save_dir}{s}")
        model.float()  # for training
        maps = np.zeros(nc) + map
        # print(maps)
        # print(len(maps))
        # print(ap_class)
        # print(len(ap_class))
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

    # da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_prec_seg.avg, ll_rec_seg.avg,ll_IoU_seg.avg, ll_mIoU_seg.avg)

    # print(da_segment_result)
    # print(ll_segment_result)
    if len(output) and config.datamode != 'chartocr':
        detect_result = np.asarray([mp, mr, map50, map])
    else:
        maps = []
        detect_result = np.asarray([0,0,0,0])
    # print('mp:{},mr:{},map50:{},map:{}'.format(mp, mr, map50, map))
    #print segmet_result
    t = [T_inf.avg, T_nms.avg]
    pivot_result = [pivo_pre_det.avg, pivo_rec_det.avg, pivo_f1_det.avg]
    return pivot_result, ll_segment_result, detect_result, losses.avg, maps, t, loss_all
        

# from LineEX.modules.CE_detection.run import *
# from LineEX.modules.KP_detection.run import *
# from LineEX.modules.KP_detection.run import keypoints
# from LineEX.modules.CE_detection.util import box_ops
# from DeepRule.test_pipe_type_cloud import chartocrpivot
from LineEX.modules.KP_detection.utils import get_keypoints
def normalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]    
    image /= 255.0
    image -= mean
    image /= std
    return image

def LineEX_Validate(epoch,config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
    """
    validata

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return 
    - writer_dict: 

    Return:
    None
    """
    # setting
    max_stride = 32
    weights = None
    losses = AverageMeter()
    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # print(save_dir)
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE] #imgsz is multiple of max_stride
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU 
    # test_batch_size = 8
    training = False
    is_coco = False #is coco dataset
    save_conf=False # save auto-label confidences
    verbose=False
    save_hybrid=False
    log_imgs,wandb = min(16,100), None

    nc = cfg.det_class
    iouv = torch.linspace(0.5,0.95,10).to(device)     #iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen =  0 
    confusion_matrix = ConfusionMatrix(nc) #detector confusion matrix
    # da_metric = SegmentationMetric(config.num_seg_class) #segment confusion matrix    
    ll_metric = SegmentationMetric(2) #segment confusion matrix
    names = [str(i) for i in range(nc)]

    names = {k: v for k, v in enumerate(names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()
    
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    
    losses = AverageMeter()

    # 曲线精度评估
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    ll_prec_seg = AverageMeter()
    ll_rec_seg = AverageMeter()

    T_inf = AverageMeter()
    T_nms = AverageMeter()

    # 关键点
    pivo_pre_det = AverageMeter()
    pivo_rec_det = AverageMeter()
    pivo_f1_det = AverageMeter()
   

    # switch to train mode
    # model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    oks_p1, oks_r1,  oks_f1, oks_p2, oks_r2,  oks_f2 = [], [], [], [], [], []
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape    #batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0]


            # t = time_synchronized()
            if model.lower() == 'lineex':

                outputs = det_model(img)
            else:
                outputs = []
            for i in range(len(paths)):
                if model.lower() == 'lineex':
                    pivots  = get_keypoints(LineEX_D, paths[i],(288, 384))
                elif model.lower() == 'chartocr':
                    det, pivots = chartocrpivot(paths[i])
                    det = det[(det[...,-1]!=4) & (det[...,-1]!=5)  & (det[...,-1]!=0)]
                    det[...,-1][det[...,-1] == 1]=0
                    det[...,-1][det[...,-1] == 2]=1
                    det[...,-1][det[...,-1] == 3]=2
                    outputs.append(det.to(device))
                pred_kp, gt_kp = get_pivort(pivots, paths[i]) 
                recall_oks,precision_oks,F1_oks = metric(pivots,gt_kp,[], paths[i],relaxed = False)
                pivo_pre_det.update(precision_oks, 1)
                pivo_rec_det.update(recall_oks, 1)
                pivo_f1_det.update(F1_oks, 1)
            # pivots = LineEX_D(img,return_attn =False)
            if model.lower() =='lineex':
                det_outs = []
                for i in range(len(paths)):
                    # image = cv2.imread(paths[i])
                    # image_ = image.copy()
                    # image = image.astype(np.float32)
                    # image = normalize(image)
                    # image = torch.from_numpy(image).to(CUDA_)
                    # image = image.permute(2, 0, 1)
                    # image = utils.nested_tensor_from_tensor_list([image])
                    # outputs = det_model(image.to(device))
                    pred_logits = outputs['pred_logits'][i][:, :nc]
                    pred_boxes = outputs['pred_boxes'][i]

                    max_output = pred_logits.softmax(-1).max(-1)
                    topk = max_output.values.topk(100)

                    pred_logits = pred_logits[topk.indices]
                    b,_  = pred_logits.shape
                    confs =  pred_logits.softmax(-1).max(-1).values.view(b,1)
                    
                    pred_boxes = pred_boxes[topk.indices]
                    pred_classes = pred_logits.argmax(axis=1).view(b,1)
                    
                    pred_boxes = (box_ops.box_cxcywh_to_xyxy(pred_boxes)*512)
                    temp = torch.concat((pred_boxes, confs, pred_classes), axis=1)
                    index = torchvision.ops.nms(temp[...,0:4], temp[..., 4], config.TEST.NMS_IOU_THRESHOLD)

                    reult = temp[index][(temp[index][...,-1]<9) & (temp[index][...,-1]!=0) \
                        & (temp[index][...,-1]!=4) & (temp[index][...,-1]!=5)]
                    # print(reult.shape)
                    reult[...,5][reult[...,5]==1] =0
                    reult[...,5][reult[...,5]==2] =1
                    reult[...,5][reult[...,5]==3] =2
                    reult[...,5][reult[...,5]==6] = 3
                    reult[...,5][reult[...,5]==7] = 4
                    reult[...,5][reult[...,5]==8] = 5
                    
                    reult = reult[reult[...,4,...]>config.TEST.NMS_CONF_THRESHOLD]
                    # print(reult)
                    det_outs.append(reult)
                # det_outs = torch.cat(det_outs, 0).view(len(det_outs),b,-1)
            elif model.lower() == 'chartocr':
                det_outs = outputs
            loss_all = []
            #lane line segment evaluation
            # temp.extend(out[1:])
            
            # total_loss, head_losses = criterion(temp, target, shapes,model)   #Compute loss
            # #
            # loss_all.append(total_loss.item()/img.size(0))
            # losses.update(total_loss.item(), img.size(0))
            if batch_i % config.PRINT_FREQ == 0:
                writer = writer_dict['writer']
                global_steps = writer_dict['valid_global_steps']
                writer.add_scalar('val_loss', losses.val, global_steps)
                # writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['valid_global_steps'] = global_steps + 1
            #output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)
            #output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRES, iou_thres=config.TEST.NMS_IOU_THRES)
            # t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0),img.size(0))
        if len(det_outs):
                # t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        
            # output = non_max_suppression(inf_out, conf_thres= config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
        else:
            output = []
        # if model == 'lineex':
        target[0][:, 1][target[0][:, 1]==4] = 3 # ticks
        target[0][:, 1][target[0][:, 1]==5] = 3 # ticks
        target[0][:, 1][target[0][:, 1]==7] = 5 # legendtext
        target[0][:, 1][target[0][:, 1]==6] = 4 # marker
        
        # target[0][:, 1][target[0][:, 1]==2] = 3 # xitle
        # target[0][:, 1][target[0][:, 1]==1] = 2 # charttitle
        # target[0][:, 1][target[0][:, 1]==0] = 1 # ytitle
        # else:
        #    target[0][:, 1][target[0][:, 1]==7] = 8 
       
        if config.TEST.PLOTS:
            if batch_i ==0:
                for i in range(test_batch_size):
                    img_det = cv2.imread(paths[i])
                    img_gt = img_det.copy()
                    if len(det_outs):
                        det = det_outs[i].clone()
                        if len(det):
                            det[:,:4] = scale_coords(img[i].shape[1:],det[:,:4],img_det.shape).round()
                        for *xyxy,conf,cls in reversed(det):
                            #print(cls)
                            if int(cls) > 8:
                                continue
                            label_det_pred = f'{names[int(cls)]}'
                            plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
                        cv2.imwrite(save_dir+"/batch_{}_{}_det_pred.png".format(epoch,i),img_det)

                        labels = target[0][target[0][:, 0] == i, 1:]
                        # print(labels)
                        labels[:,1:5]=xywh2xyxy(labels[:,1:5])
                        if len(labels):
                            labels[:,1:5]=scale_coords(img[i].shape[1:],labels[:,1:5],img_gt.shape).round()
                        for cls,x1,y1,x2,y2 in labels:
                            #print(names)
                            #print(cls)

                            label_det_gt = f'{names[int(cls)]}'
                            xyxy = (x1,y1,x2,y2)
                            plot_one_box(xyxy, img_gt , label=label_det_gt, color=colors[int(cls)], line_thickness=2)
                        cv2.imwrite(save_dir+"/batch_{}_{}_det_gt.png".format(epoch,i),img_gt)

        # Statistics per image
        # output([xyxy,conf,cls])
        # target[0] ([img_id,cls,xyxy])
        if  len(det_outs):
            for si, pred in enumerate(det_outs):

                labels = target[0][target[0][:, 0] == si, 1:]     #all object in one image 
                nl = len(labels)    # num of object
                # labels = labels
        
                tcls = labels[:, 0].tolist() if nl else []  # target class

                path = Path(paths[si])
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    if config.TEST.PLOTS:
                        confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                    # Per target class
                    for cls in torch.unique(tcls_tensor):                    
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            # n*m  n:pred  m:label
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if config.TEST.PLOTS and batch_i < 3:
            f = save_dir +'/'+ f'test_batch{batch_i}_labels.jpg'  # labels
            #Thread(target=plot_images, args=(img, target[0], paths, f, names), daemon=True).start()
            f = save_dir +'/'+ f'test_batch{batch_i}_pred.jpg'  # predictions
            #Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

    # Compute statistics
    # stats : [[all_img_correct]...[all_img_tcls]]
    if len(stats) and config.datamode!='chartocr':
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

        map70 = None
        map75 = None
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
            ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, mf, map50, map70, map = p.mean(), r.mean(), f1.mean(), ap50.mean(), ap70.mean(),ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        
        pf = '%20s' + '%12.3s' * 8  # print format
        print(pf  % ('all', 'seen', 'sum', 'mp', 'mr','mf', 'map50','map70', 'map'))
        pf = '%20s' + '%12.3g' * 8  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr,mf, map50,map70, map))
        #print(map70)
        #print(map75)
        pf = '%20s' + '%12.3s' * 8  
        print(pf  % ('label', 'seen', 'sum', 'p', 'r','f1', 'map50','map70', 'map'))
        pf = '%20s' + '%12.3g' * 8 
        # Print results per class
        if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], f1[i], ap50[i], ap70[i], ap[i]))
        # Print speeds
        t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
        if not training:
            print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

        # Plots
        if config.TEST.PLOTS:
            confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
            if wandb and wandb.run:
                wandb.log({"Images": wandb_images})
                wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

        # Save JSON
        if config.TEST.SAVE_JSON and len(jdict):
            w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
            anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
            pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
            print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
            with open(pred_json, 'w') as f:
                json.dump(jdict, f)

            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api
                eval = COCOeval(anno, pred, 'bbox')
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]  # image IDs to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                print(f'pycocotools unable to run: {e}')

        # Return results
        if not training:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
            print(f"Results saved to {save_dir}{s}")
        # model.float()  # for training
        maps = np.zeros(nc) + map
        # print(maps)
        # print(len(maps))
        # print(ap_class)
        # print(len(ap_class))
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

    # da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_prec_seg.avg, ll_rec_seg.avg,ll_IoU_seg.avg, ll_mIoU_seg.avg)

    # print(da_segment_result)
    # print(ll_segment_result)
    if len(det_outs) and config.datamode!='chartocr':
        detect_result = np.asarray([mp, mr, map50, map])
    else:
        maps = []
        detect_result = np.asarray([0,0,0,0])
    # print('mp:{},mr:{},map50:{},map:{}'.format(mp, mr, map50, map))
    #print segmet_result
    t = [T_inf.avg, T_nms.avg]
    pivot_result = [pivo_pre_det.avg, pivo_rec_det.avg, pivo_f1_det.avg]
    return pivot_result, ll_segment_result, detect_result, losses.avg, maps, t, loss_all


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0