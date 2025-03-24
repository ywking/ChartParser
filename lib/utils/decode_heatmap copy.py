import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
import cv2
import matplotlib.pyplot as plt


def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def detect_heatmap( image, heatmap_save_path,model):
    
    #---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #---------------------------------------------------------#
    # image       = cvtColor(image)
    #---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    #---------------------------------------------------------#
    # image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
    #-----------------------------------------------------------#
    #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
    #-----------------------------------------------------------#
    # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

    with torch.no_grad():
        # images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
        
        images = images.cuda()
        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        outputs = model(images)
        # if self.backbone == 'hourglass':
        #     outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
    
    plt.imshow(image, alpha=1)
    plt.axis('off')
    mask        = np.zeros((image.size[1], image.size[0]))
    score       = np.max(outputs[0][0].permute(1, 2, 0).cpu().numpy(), -1)
    score       = cv2.resize(score, (image.size[0], image.size[1]))
    normed_score    = (score * 255).astype('uint8')
    mask            = np.maximum(mask, normed_score)
    
    plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
    print("Save to the " + heatmap_save_path)
    plt.show()

def show_heatmap(image,outputs,heatmap_save_path,is_gt=False,batch=0,num=0):
    # pred (b,c,h,w)
    # target (b,h,w,c)
    # im = np.array(image[0].cpu().permute(1,2,0)).astype('uint8')
    # image = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    plt.figure(figsize=(2,2))
    if is_gt:
        image = cv2.resize(image, (outputs.shape[1], outputs.shape[0]))
    else:
        image = cv2.resize(image, (outputs.shape[2], outputs.shape[1]))
    plt.imshow(image, alpha=1)
    plt.axis('off')
    if is_gt:
        heatmap_save_path += f'/batch_{batch}_{num}_hm_gt.png'
        hm = outputs.cpu().numpy()
    else:
        heatmap_save_path += f'/batch_{batch}_{num}_hm_pred.png'
        hm = outputs.permute(1, 2, 0).cpu().numpy()
    mask        = np.zeros((image.shape[0], image.shape[1]))
    score       = np.max(hm, -1)
    # score       = cv2.resize(score, (image.shape[0], image.shape[1]))
    normed_score    = (score*255).astype('uint8')
    mask            = np.maximum(mask, normed_score)
    
    plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
    print("Save to the " + heatmap_save_path)
    plt.show()
    plt.close()


def decode_hm(pred_hms, pred_offsets, confidence, cuda):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128, 
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    #-------------------------------------------------------------------------#
    #   只传入一张图片，循环只进行一次
    #-------------------------------------------------------------------------#
    for batch in range(b):
        #-------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_wh         128*128, 2              特征点的预测宽高
        #                                           在预测过程的前处理以及后处理视频中讲的有点小问题，不是调整参数，就是宽高
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        #-------------------------------------------------------------------------#
        heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        # pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        yv, xv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        #-------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        #-------------------------------------------------------------------------#
        xv, yv      = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv      = xv.cuda()
            yv      = yv.cuda()

        #-------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        #-------------------------------------------------------------------------#
        class_conf, class_pred  = torch.max(heat_map, dim = -1)
        mask                    = class_conf > confidence

        #-----------------------------------------#
        #   取出得分筛选后对应的结果
        #-----------------------------------------#
        # pred_wh_mask        = pred_wh[mask]
        pred_offset_mask    = pred_offset[mask]
        # if len(pred_wh_mask) == 0:
        #     detects.append([])
        #     continue     

        #----------------------------------------#
        #   计算调整后预测框的中心
        #----------------------------------------#
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        #----------------------------------------#
        #   计算预测框的宽高
        #----------------------------------------#
        # half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        #----------------------------------------#
        #   获得预测框的左上角和右下角
        #----------------------------------------#
        # print(xv,yv)
        # print(xv_mask,yv_mask)
        bboxes = torch.concat([xv_mask, yv_mask],dim=1)
        # bboxes[:, [0, 2]] /= output_w
        # bboxes[:, [1, 3]] /= output_h
        detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect)

    return detects


def decode_line(pred_hms,confidence):
    pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    #-------------------------------------------------------------------------#
    #   只传入一张图片，循环只进行一次
    #-------------------------------------------------------------------------#
    for batch in range(b):
        #-------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_wh         128*128, 2              特征点的预测宽高
        #                                           在预测过程的前处理以及后处理视频中讲的有点小问题，不是调整参数，就是宽高
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        #-------------------------------------------------------------------------#
        heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        # pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
        # pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        yv, xv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        #-------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        #-------------------------------------------------------------------------#
        xv, yv      = xv.flatten().float(), yv.flatten().float()
        # if cuda:
        #     xv      = xv.cuda()
        #     yv      = yv.cuda()

        #-------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        #-------------------------------------------------------------------------#
        class_conf, class_pred  = torch.max(heat_map, dim = -1)
        mask                    = class_conf > confidence

        #-----------------------------------------#
        #   取出得分筛选后对应的结果
        #-----------------------------------------#
        # pred_wh_mask        = pred_wh[mask]
        # pred_offset_mask    = pred_offset[mask]
        # if len(pred_wh_mask) == 0:
        #     detects.append([])
        #     continue     

        #----------------------------------------#
        #   计算调整后预测框的中心
        #----------------------------------------#
        xv_mask = torch.unsqueeze(xv[mask], -1)
        yv_mask = torch.unsqueeze(yv[mask], -1)
        #----------------------------------------#
        #   计算预测框的宽高
        #----------------------------------------#
        # half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        #----------------------------------------#
        #   获得预测框的左上角和右下角
        #----------------------------------------#
        # print(xv,yv)
        # print(xv_mask,yv_mask)
        bboxes = torch.concat([xv_mask, yv_mask],dim=1)
        # bboxes[:, [0, 2]] /= output_w
        # bboxes[:, [1, 3]] /= output_h
        detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect)

    return detects