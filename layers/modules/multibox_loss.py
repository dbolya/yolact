# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, decode
from utils.functions import sanitize_coordinates

from data import cfg, mask_type, activation_func

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio
        
        self.mask_dim = cfg.mask_dim
        self.use_gt_bboxes = cfg.use_gt_bboxes
        self.train_masks = cfg.train_masks
        
        # Extra loss coefficients to get all the losses to be in a similar range
        self.mask_alpha = 0.2 / cfg.mask_dim
        self.bbox_alpha = 5 if cfg.use_yolo_regressors else 1

        if cfg.mask_proto_normalize_mask_loss:
            self.mask_alpha *= 30
        
        if cfg.mask_proto_least_squares_loss:
            self.ls_proto_alpha = 3 * 10
            self.ls_coeff_alpha = 3 * 1

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20*20/70/70
        self.l1_alpha = 0.1

    def forward(self, predictions, targets, masks):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]
            
            * Only if mask_type == lincomb
        """
        if cfg.mask_type == mask_type.lincomb:
            loc_data, conf_data, mask_data, priors, proto_data = predictions
        else:
            loc_data, conf_data, mask_data, priors = predictions
        
        num = loc_data.size(0)
        # This is necessary for training on multiple GPUs because
        # DataParallel will cat the priors from each GPU together
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # Match priors (default boxes) and ground truth boxes
        # These tensors will be created with the same device as loc_data
        loc_t = loc_data.new(num, num_priors, 4)
        conf_t = loc_data.new(num, num_priors).long()
        idx_t = loc_data.new(num, num_priors).long()

        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.pos_threshold, self.neg_threshold,
                  truths, defaults, labels,
                  loc_t, conf_t, idx_t, idx, loc_data[idx])

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') * self.bbox_alpha

        # Mask Loss
        loss_p = 0 # Proto loss
        if self.train_masks:
            if cfg.mask_type == mask_type.direct:
                if self.use_gt_bboxes:
                    pos_masks = []
                    for idx in range(num):
                        pos_masks.append(masks[idx][idx_t[idx, pos[idx]]])
                    masks_t = torch.cat(pos_masks, 0)
                    masks_p = mask_data[pos, :].view(-1, self.mask_dim)
                    loss_m = F.binary_cross_entropy(masks_p, masks_t, reduction='sum') * self.mask_alpha
                else:
                    loss_m = self.direct_mask_loss(pos_idx, idx_t, loc_data, mask_data, priors, masks)
            elif cfg.mask_type == mask_type.lincomb:
                loss_m = self.lincomb_mask_loss(pos, idx_t, loc_data, mask_data, priors, proto_data, masks)
                
                if cfg.mask_proto_loss is not None:
                    if cfg.mask_proto_loss == 'l1':
                        loss_p = torch.mean(torch.abs(proto_data)) / self.l1_expected_area * self.l1_alpha
                    elif cfg.mask_proto_loss == 'disj':
                        loss_p = -torch.mean(torch.max(F.log_softmax(proto_data, dim=-1), dim=-1)[0])
                        
        else:
            loss_m = 0

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]
        
        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos]        = 0 # filter out pos boxes
        loss_c[conf_t < 0] = 0 # filter out neutrals (conf_t = -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        
        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg[pos]        = 0
        neg[conf_t < 0] = 0 # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N
        loss_m /= N
        return loss_l, loss_c, loss_m + loss_p


    def direct_mask_loss(self, pos_idx, idx_t, loc_data, mask_data, priors, masks):
        """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
        loss_m = 0
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                cur_pos_idx = pos_idx[idx, :, :]
                cur_pos_idx_squeezed = cur_pos_idx[:, 1]

                # Shape: [num_priors, 4], decoded predicted bboxes
                pos_bboxes = decode(loc_data[idx, :, :], priors.data)
                pos_bboxes = pos_bboxes[cur_pos_idx].view(-1, 4).clamp(0, 1)
                pos_lookup = idx_t[idx, cur_pos_idx_squeezed]

                cur_masks = masks[idx]
                pos_masks = cur_masks[pos_lookup, :, :]
                
                # Convert bboxes to absolute coordinates
                num_pos, img_height, img_width = pos_masks.size()

                # Take care of all the bad behavior that can be caused by out of bounds coordinates
                x1, x2 = sanitize_coordinates(pos_bboxes[:, 0], pos_bboxes[:, 2], img_width)
                y1, y2 = sanitize_coordinates(pos_bboxes[:, 1], pos_bboxes[:, 3], img_height)

                # Crop each gt mask with the predicted bbox and rescale to the predicted mask size
                # Note that each bounding box crop is a different size so I don't think we can vectorize this
                scaled_masks = []
                for jdx in range(num_pos):
                    tmp_mask = pos_masks[jdx, y1[jdx]:y2[jdx], x1[jdx]:x2[jdx]]

                    # Restore any dimensions we've left out because our bbox was 1px wide
                    while tmp_mask.dim() < 2:
                        tmp_mask = tmp_mask.unsqueeze(0)

                    new_mask = F.adaptive_avg_pool2d(tmp_mask.unsqueeze(0), cfg.mask_size)
                    scaled_masks.append(new_mask.view(1, -1))

                mask_t = torch.cat(scaled_masks, 0).gt(0.5).float() # Threshold downsampled mask
            
            pos_mask_data = mask_data[idx, cur_pos_idx_squeezed, :]
            loss_m += F.binary_cross_entropy(pos_mask_data, mask_t, reduction='sum') * self.mask_alpha

        return loss_m
    

    def lincomb_mask_loss(self, pos, idx_t, loc_data, mask_data, priors, proto_data, masks):
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)

        loss_m = 0

        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                downsampled_masks = F.adaptive_avg_pool2d(masks[idx], (mask_h, mask_w))
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
                
                if cfg.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()

            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            
            pos_bboxes = decode(loc_data[idx, :, :], priors.data)
            pos_bboxes = pos_bboxes[cur_pos, :]

            proto_masks = proto_data[idx]
            proto_coef  = mask_data[idx, cur_pos, :]
            
            # If we have over the allowed number of masks, select a random sample
            if proto_coef.size(0) > cfg.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[:cfg.masks_to_train]

                proto_coef = proto_coef[select, :]
                pos_bboxes = pos_bboxes[select, :]
                pos_idx_t  = pos_idx_t[select]

            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]

            if cfg.mask_proto_least_squares_loss:
                # Instead of computing the loss as a downstream task, i.e. with |Proto * coeffs - gt|,
                # Compute it directly from the least squares solution x in A x = b.
                # I.e. compute the loss as |x - coeffs| + |Proto * x - gt|
                #
                # Remember to use no coeff / mask nonlinearity for this option.

                # Don't compute gradients because backprop through singular svd(A.T * A) is unstable.
                with torch.no_grad():
                    # Size: [70*70, num_dets]
                    b = downsampled_masks.view(-1, downsampled_masks.size(-1))
                    # Size: [70*70, mask_dim]
                    A = proto_masks.view(-1, proto_data.size(-1))

                    # Some fancy math stuffs to calculate the inverse of a singular A.T * A
                    U, s, V = torch.svd(A.t() @ A)
                    ATA_inv = U / s @ V.t()
                    
                    # x is the least squares solution to A x = b (Size: [256, num_dets])
                    x = (ATA_inv @ A.t()) @ b

                x_pos  = x[:, pos_idx_t]
                
                coeff_loss = F.smooth_l1_loss(proto_coef.t(), x_pos, reduction='sum')
                proto_loss = F.smooth_l1_loss(proto_masks @ x_pos, mask_t, reduction='sum')

                loss_m += self.ls_coeff_alpha * coeff_loss + self.ls_proto_alpha * proto_loss

                continue              

            # Size: [mask_h, mask_w, num_pos]
            pred_masks = torch.matmul(proto_masks, proto_coef.t())
            pred_masks = cfg.mask_proto_mask_activation(pred_masks)

            if cfg.mask_proto_crop:
                # Take care of all the bad behavior that can be caused by out of bounds coordinates
                x1, x2 = sanitize_coordinates(pos_bboxes[:, 0], pos_bboxes[:, 2], mask_w)
                y1, y2 = sanitize_coordinates(pos_bboxes[:, 1], pos_bboxes[:, 3], mask_h)

                # "Crop" predicted masks by zeroing out everything not in the predicted bbox
                # TODO: Write a cuda implementation of this to get rid of the loop
                crop_mask = torch.zeros(mask_h, mask_w, num_pos)
                for jdx in range(num_pos):
                    crop_mask[y1[jdx]:y2[jdx], x1[jdx]:x2[jdx], jdx] = 1
                pred_masks = pred_masks * crop_mask
            
            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(pred_masks, mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='none')

            if cfg.mask_proto_normalize_mask_loss:
                gt_area  = torch.sum(mask_t,   dim=(0, 1))
                pre_loss = torch.sum(pre_loss, dim=(0, 1))

                loss_m += torch.sum(pre_loss / (torch.sqrt(gt_area) + 1))
            else:
                loss_m += torch.sum(pre_loss)

        return loss_m * self.mask_alpha
