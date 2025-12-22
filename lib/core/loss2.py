import torch.nn as nn
import torch
from .general import bbox_iou
from .postprocess import build_targets
from lib.core.evaluate import SegmentationMetric
import math
from functools import partial

import torch
import torch.nn.functional as F

# log = torch.sigmoid()

class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)
        self.mse = nn.MSELoss()
        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg
        # self.embedingloss = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
        

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        # head_losses = [ll
        #                 for l, f, t in zip(self.losses, head_fields, head_targets)
        #                 for ll in l(f, t)]
        #
        # assert len(self.lambdas) == len(head_losses)
        # loss_values = [lam * l
        #                for lam, l in zip(self.lambdas, head_losses)
        #                if l is not None]
        # total_loss = sum(loss_values) if loss_values else None
        # print(model.nc)
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            # predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            # targets: gts [det_targets, segment_targets, lane_targets]
            model:
            predictions:  predicts of [[det_head1, det_head2, det_head3],hm,reg,seg]
            targets: gts [det_targets, hm,reg, mask, seg]
        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj, BCEseg = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
        c_loss = 0
        off_loss = 0
        ## 计算热图和偏移损失
        for hg in predictions[1]:

            c_loss          += BCEseg(hg['hm'].permute(0,2,3,1), targets[1])
            # wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        += reg_l1_loss(hg['reg'], targets[2], targets[3])
        
        # 曲线分割Loss
        drive_area_seg_predicts = predictions[-1]
        drive_area_seg_targets = targets[-1]
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)

        # lane_line_seg_predicts = predictions[2].view(-1)
        # lane_line_seg_targets = targets[2].view(-1)
        # lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)

        # metric = SegmentationMetric(2)
        # nb, _, height, width = targets[1].shape
        # pad_w, pad_h = shapes[0][1][1]
        # pad_w = int(pad_w)
        # pad_h = int(pad_h)
        # _,lane_line_pred=torch.max(predictions[2], 1)
        # _,lane_line_gt=torch.max(targets[2], 1)
        # lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
        # lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        # metric.reset()
        # metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
        # IoU = metric.IntersectionOverUnion()
        # liou_ll = 1 - IoU

        s = 3 / no  # output count scaling
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]

        lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        # lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        # liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]

        
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:
            lseg_da = 0 * lseg_da
            c_loss = 0 * c_loss
            off_loss = 0 * off_loss
            # lseg_ll = 0 * lseg_ll
            # liou_ll = 0 * liou_ll
            
        if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            # c_loss = 0 * c_loss
            # off_loss = 0 * off_loss

        if cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da

        if cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            # lseg_ll = 0 * lseg_ll
            # liou_ll = 0 * liou_ll
        # off_loss = 0
        # loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
        # var_loss, dist_loss, reg_loss = self.embedingloss(pix_embedding, instance_label)
        loss =  lbox + lobj + lcls + c_loss + lseg_da + off_loss
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), c_loss.item(), loss.item())


def focal_loss(pred, target):
    # pred = pred.sigmoid()
    # print(pred)
    pred = pred.permute(0, 2, 3, 1)

    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    # pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    # pred = torch.sigmoid(pred)
    pred = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

def get_loss(cfg, device):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT])).to(device)
    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0:
         BCEcls, BCEobj,BCEseg = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma),FocalLoss(BCEseg, gamma)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)

    # 计算聚类损失

    return loss



class MultiHeadLoss2(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)
        self.mse = nn.MSELoss()
        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas
        self.cfg = cfg
        # self.embedingloss = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
        

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        # head_losses = [ll
        #                 for l, f, t in zip(self.losses, head_fields, head_targets)
        #                 for ll in l(f, t)]
        #
        # assert len(self.lambdas) == len(head_losses)
        # loss_values = [lam * l
        #                for lam, l in zip(self.lambdas, head_losses)
        #                if l is not None]
        # total_loss = sum(loss_values) if loss_values else None
        # print(model.nc)
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            # predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            # targets: gts [det_targets, segment_targets, lane_targets]
            model:
            predictions:  predicts of [[det_head1, det_head2, det_head3],hm,reg,seg]
            targets: gts [det_targets, hm,reg, mask, seg]
        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        cfg = self.cfg
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = build_targets(cfg, predictions[0], targets[0], model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        BCEcls, BCEobj, BCEseg = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        c_loss = 0
        off_loss = 0
        ## 计算热图和偏移损失
        for hg in predictions[0]:
            c_loss          += BCEseg(hg['hm'].permute(0,2,3,1), targets[0])
            # wh_loss         = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss        += reg_l1_loss(hg['reg'], targets[1], targets[2])

        # 曲线分割Loss
        drive_area_seg_predicts = predictions[-1]
        drive_area_seg_targets = targets[3]
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)

        # lane_line_seg_predicts = predictions[2].view(-1)
        # lane_line_seg_targets = targets[2].view(-1)
        # lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)

        # metric = SegmentationMetric(2)
        # nb, _, height, width = targets[1].shape
        # pad_w, pad_h = shapes[0][1][1]
        # pad_w = int(pad_w)
        # pad_h = int(pad_h)
        # _,lane_line_pred=torch.max(predictions[2], 1)
        # _,lane_line_gt=torch.max(targets[2], 1)
        # lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
        # lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        # metric.reset()
        # metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
        # IoU = metric.IntersectionOverUnion()
        # liou_ll = 1 - IoU

        # s = 3 / no  # output count scaling
        # lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        # lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.) * self.lambdas[1]
        # lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]

        # lseg_da *= cfg.LOSS.DA_SEG_GAIN * self.lambdas[3]
        # lseg_ll *= cfg.LOSS.LL_SEG_GAIN * self.lambdas[4]
        # liou_ll *= cfg.LOSS.LL_IOU_GAIN * self.lambdas[5]

        
        if cfg.TRAIN.DET_ONLY or cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:
            lseg_da = 0 * lseg_da
            c_loss = 0 * c_loss
            off_loss = 0 * off_loss
            # lseg_ll = 0 * lseg_ll
            # liou_ll = 0 * liou_ll
            
        if cfg.TRAIN.SEG_ONLY or cfg.TRAIN.ENC_SEG_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            # c_loss = 0 * c_loss
            # off_loss = 0 * off_loss

        if cfg.TRAIN.LANE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            lseg_da = 0 * lseg_da

        if cfg.TRAIN.DRIVABLE_ONLY:
            lcls = 0 * lcls
            lobj = 0 * lobj
            lbox = 0 * lbox
            # lseg_ll = 0 * lseg_ll
            # liou_ll = 0 * liou_ll
        # off_loss = 0
        loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
        # var_loss, dist_loss, reg_loss = self.embedingloss(predictions[-1], targets[-1])
        # loss =  lbox + lobj + lcls + c_loss + lseg_da + off_loss
        # loss = c_loss + lseg_da + off_loss
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lseg_da.item(), c_loss.item(), loss.item())

def get_loss2(cfg, device):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)
    -device: cpu or gpu device

    Returns:
    -loss: (MultiHeadLoss)

    """
    # class loss criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
    # object loss criteria
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
    # segmentation loss criteria
    BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.LOSS.SEG_POS_WEIGHT]),reduction='mean').to(device)
    # Focal loss
    gamma = cfg.LOSS.FL_GAMMA  # focal loss gamma
    if gamma > 0:
        BCEcls, BCEobj,BCEseg = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma),FocalLoss(BCEseg, gamma)

    loss_list = [BCEcls, BCEobj, BCEseg]
    loss = MultiHeadLoss2(loss_list, cfg=cfg, lambdas=cfg.LOSS.MULTI_HEAD_LAMBDA)

    # 计算聚类损失

    return loss

# example
# class L1_Loss(nn.Module)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

from torch.nn.modules.loss import _Loss


class DiscriminativeLoss(_Loss):
    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=False, size_average=True):
        super(DiscriminativeLoss, self).__init__(reduction='mean')
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target):

        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, embedding, seg_gt):
        batch_size, embed_dim, H, W = embedding.shape
        embedding = embedding.reshape(batch_size, embed_dim, H*W)
        seg_gt = seg_gt.reshape(batch_size, H*W)

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H*W)
            seg_gt_b = seg_gt[b]  # (H*W)

            labels, indexs = torch.unique(seg_gt_b, return_inverse=True)
            num_lanes = len(labels)
            if num_lanes == 0:
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)

                if not seg_mask_i.any():
                    continue
                
                embedding_i = embedding_b * seg_mask_i
                mean_i = torch.sum(embedding_i, dim=1) / torch.sum(seg_mask_i)
                centroid_mean.append(mean_i)
                # ---------- var_loss -------------
                var_loss = var_loss + torch.sum(F.relu(
                    torch.norm(embedding_i[:,seg_mask_i] - mean_i.reshape(embed_dim, 1), dim=0) - self.delta_var) ** 2) / torch.sum(seg_mask_i) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)

                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)   # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_dist

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_dist) ** 2) / (
                        num_lanes * (num_lanes - 1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size

        return var_loss, dist_loss, reg_loss