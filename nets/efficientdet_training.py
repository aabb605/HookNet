import math
from functools import partial

import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calc_iou(a, b):
    max_length = torch.max(a)
    a = a / max_length
    b = b / max_length
    
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

def calc_iou_loss(preds, bbox, eps=1e-6):
    x1 = torch.max(preds[:, 0], bbox[:, 0])
    y1 = torch.max(preds[:, 1], bbox[:, 1])
    x2 = torch.min(preds[:, 2], bbox[:, 2])
    y2 = torch.min(preds[:, 3], bbox[:, 3])
    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)
    inters = w * h
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
                bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    ious = (inters / uni).clamp(min=eps)
    iou_loss = -ious.log()
    #iou_loss = torch.mean(iou_loss)
    return iou_loss

def alpha_giou_loss(preds, bbox, eps=1e-7, alpha=2):
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])
    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)
    inters = iw * ih
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
                bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps
    #ious = inters / uni
    #ious = (inters / uni).clamp(min=eps)
    ious = torch.pow(inters / uni + eps, alpha)
    ex1 = torch.min(preds[:, 0], bbox[:, 0])
    ey1 = torch.min(preds[:, 1], bbox[:, 1])
    ex2 = torch.max(preds[:, 2], bbox[:, 2])
    ey2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)
    enclose = ew * eh + eps
    #giou = ious - (enclose - uni) / enclose
    giou = ious - torch.pow((enclose - uni) / enclose + eps, alpha)
    giou = torch.clamp(giou, min=-1.0, max=1.0)
    giou_loss = 1 - giou
    #giou_loss = torch.mean(giou_loss)
    return giou_loss

def calc_giou_loss(preds, bbox, eps=1e-7):
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])
    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)
    inters = iw * ih
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
                bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps
    #ious = inters / uni
    ious = (inters / uni).clamp(min=eps)
    ex1 = torch.min(preds[:, 0], bbox[:, 0])
    ey1 = torch.min(preds[:, 1], bbox[:, 1])
    ex2 = torch.max(preds[:, 2], bbox[:, 2])
    ey2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)
    enclose = ew * eh + eps
    giou = ious - (enclose - uni) / enclose
    giou = torch.clamp(giou, min=-1.0, max=1.0)
    giou_loss = 1 - giou
    #giou_loss = torch.mean(giou_loss)
    return giou_loss

def calc_diou_loss(preds, bbox, eps=1e-7):
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])
    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)
    inters = iw * ih
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * ( bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    iou = inters / (uni + eps).clamp(min=eps)
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2
    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2
    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])
    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
    diou = iou - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)
    diou_loss = 1 - diou
    #diou_loss = torch.mean(diou_loss)
    return diou_loss

def alpha_diou_loss(preds, bbox, eps=1e-7, alpha=2):
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])
    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)
    inters = iw * ih
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * ( bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    #iou = inters / (uni + eps).clamp(min=eps)
    ious = torch.pow(inters / uni + eps, alpha)
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2
    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2
    inter_diag = ((cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2 )**alpha
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])
    outer_diag = ((ox1 - ox2) ** 2 + (oy1 - oy2) ** 2 )**alpha + eps
    diou = ious - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)
    diou_loss = 1 - diou
    #diou_loss = torch.mean(diou_loss)
    return diou_loss

def calc_ciou_loss(preds, bbox, eps=1e-7):
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])
    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)
    inters = iw * ih
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
                bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    iou = inters / (uni + eps).clamp(min=eps)
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2
    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2
    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])
    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
    diou = iou - inter_diag / outer_diag
    wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
    hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
    wpreds = preds[:, 2] - preds[:, 0] + 1.0
    hpreds = preds[:, 3] - preds[:, 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)
    ciou_loss = 1 - ciou
    #ciou_loss = torch.mean(ciou_loss)
    return ciou_loss

def get_target(anchor, bbox_annotation, classification, cuda):

    IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

    IoU_max, IoU_argmax = torch.max(IoU, dim=1)

    targets = torch.ones_like(classification) * -1
    targets = targets.type_as(classification)

    targets[torch.lt(IoU_max, 0.4), :] = 0

    positive_indices = torch.ge(IoU_max, 0.5)

    assigned_annotations = bbox_annotation[IoU_argmax, :]

    targets[positive_indices, :] = 0
    targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

    num_positive_anchors = positive_indices.sum()
    return targets, num_positive_anchors, positive_indices, assigned_annotations

def encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):

    assigned_annotations = assigned_annotations[positive_indices, :]

    anchor_widths_pi = anchor_widths[positive_indices]
    anchor_heights_pi = anchor_heights[positive_indices]
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

    gt_widths = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)

    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
    targets_dw = torch.log(gt_widths / anchor_widths_pi)
    targets_dh = torch.log(gt_heights / anchor_heights_pi)

    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
    targets = targets.t()
    return targets

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, alpha = 0.25, gamma = 2.0, cuda = True):

        batch_size = classifications.shape[0]

        dtype = regressions.dtype
        anchor = anchors[0, :, :].to(dtype)

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        regression_losses = []
        classification_losses = []
        for j in range(batch_size):

            bbox_annotation = annotations[j]
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            
            classification = torch.clamp(classification, 5e-4, 1.0 - 5e-4)
            
            if len(bbox_annotation) == 0:

                alpha_factor = torch.ones_like(classification) * alpha
                alpha_factor = alpha_factor.type_as(classification)

                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = - (torch.log(1.0 - classification))
                
                cls_loss = focal_weight * bce
                
                classification_losses.append(cls_loss.sum())

                regression_losses.append(torch.tensor(0).type_as(classification))
                    
                continue

            targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(anchor, 
                                                                                        bbox_annotation, classification, cuda)

            alpha_factor = torch.ones_like(targets) * alpha
            alpha_factor = alpha_factor.type_as(classification)

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = - (targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            zeros = zeros.type_as(cls_loss)
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                targets = encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y)
                '''
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
                '''
                Giou_loss = alpha_giou_loss(regression[positive_indices,:],targets)
                #Diou_loss = calc_diou_loss(regression[positive_indices,:],targets)
                regression_losses.append(Giou_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).type_as(classification))

        c_loss = torch.stack(classification_losses).mean()
        r_loss = torch.stack(regression_losses).mean()
        loss = c_loss + r_loss
        return loss, c_loss, r_loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    if epoch<20:
        lr = 0.001
    else:
        lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
