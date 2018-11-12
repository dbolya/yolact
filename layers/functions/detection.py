import torch
import torch.nn.functional as F
from ..box_utils import decode, nms, jaccard
from utils import timer

from data import cfg, mask_type


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        
        self.cross_class_nms = False
        self.fast_nms = False

    def __call__(self, predictions):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data   = predictions['loc']
        conf_data  = predictions['conf']
        mask_data  = predictions['mask']
        prior_data = predictions['priors']

        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data  = predictions['inst']  if 'inst'  in predictions else None

        with timer.env('Detect'):
            num = loc_data.size(0)  # batch size
            output = torch.zeros(num, self.top_k, 1 + 1 + 4 + mask_data.size(2))

            num_priors = prior_data.size(0)
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)

            # Decode predictions into bboxes.
            for i in range(num):
                decoded_boxes = decode(loc_data[i], prior_data)
                
                if self.cross_class_nms:
                    self.detect_cross_class(i, conf_preds, decoded_boxes, mask_data, inst_data, output)
                else:
                    self.detect_per_class(i, conf_preds, decoded_boxes, mask_data, inst_data, output)
        
        return {'output': output, 'proto_data': proto_data}


    def detect_cross_class(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data, output):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        conf_scores, class_labels = torch.max(conf_preds[batch_idx, 1:], dim=0)

        c_mask = conf_scores.gt(self.conf_thresh)
        scores = conf_scores[c_mask]
        classes = class_labels[c_mask]
    
        if scores.size(0) == 0:
            return
    
        l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
        boxes = decoded_boxes[l_mask].view(-1, 4)
        masks = mask_data[batch_idx, l_mask[:, 0], :]
        if inst_data is not None:
            inst = inst_data[batch_idx, l_mask[:, 0], :]
        
        # idx of highest scoring and non-overlapping boxes per classif cfg.use_coeff_nms:
        if cfg.use_coeff_nms:
            if inst_data is not None:
                ids, count = self.coefficient_nms(inst, scores, top_k=self.top_k)
            else:
                ids, count = self.coefficient_nms(masks, scores, top_k=self.top_k)
        else:
            # Use this function instead for not 100% correct nms but 4ms faster
            if self.fast_nms:
                ids, count = self.box_nms(boxes, scores, self.nms_thresh, self.top_k)
            else:
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k, force_cpu=cfg.force_cpu_nms)
                ids = ids[:count]
        
        output[batch_idx, :count] = \
            torch.cat((classes[ids].unsqueeze(1).float(), scores[ids].unsqueeze(1), boxes[ids], masks[ids]), 1)

    def detect_per_class(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data, output):
        """ Perform nms for each non-background class predicted. """
        conf_scores = conf_preds[batch_idx].clone()
        tmp_output = torch.zeros(self.num_classes-1, self.top_k, output.size(2))

        for cl in range(1, self.num_classes):
            c_mask = conf_scores[cl].gt(self.conf_thresh)
            scores = conf_scores[cl][c_mask]
            if scores.size(0) == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            masks = mask_data[batch_idx, l_mask[:, 0], :]
            if inst_data is not None:
                inst = inst_data[batch_idx, l_mask[:, 0], :]
            
            if cfg.use_coeff_nms:
                if inst_data is not None:
                    ids, count = self.coefficient_nms(inst, scores, top_k=self.top_k)
                else:
                    ids, count = self.coefficient_nms(masks, scores, top_k=self.top_k)
            else:
                # Use this function instead for not 100% correct nms but 4ms faster
                if self.fast_nms:
                    ids, count = self.box_nms(boxes, scores, self.nms_thresh, self.top_k)
                else:
                    ids, count = nms(boxes, scores, self.nms_thresh, self.top_k, force_cpu=cfg.force_cpu_nms)
                    ids = ids[:count]
            
            classes = torch.ones(count, 1).float()*(cl-1)
            tmp_output[cl-1, :count] = \
                torch.cat((classes, scores[ids].unsqueeze(1), boxes[ids], masks[ids]), 1)

        # Pool all the outputs together regardless of class and pick the top_k
        tmp_output = tmp_output.view(-1, output.size(2))
        _, idx = tmp_output[:, 1].sort(0, descending=True)
        output[batch_idx, :, :] = tmp_output[idx[:self.top_k], :]
    

    def coefficient_nms(self, coeffs, scores, cos_threshold=0.9, top_k=400):
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        coeffs_norm = F.normalize(coeffs[idx], dim=1)

        # Compute the pairwise cosine similarity between the coefficients
        cos_similarity = coeffs_norm @ coeffs_norm.t()
        
        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        cos_similarity.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the cos similarity matrix along the columns, each column will represent the
        # maximum cosine similarity between this element and every element with a higher
        # score than this element.
        cos_max, _ = torch.max(cos_similarity, dim=0)

        # Now just filter out the ones higher than the threshold
        idx_out = idx[cos_max <= cos_threshold]


        # new_mask_norm = F.normalize(masks[idx_out], dim=1)
        # print(new_mask_norm[:5] @ new_mask_norm[:5].t())
        
        return idx_out, idx_out.size(0)
    
    def box_nms(self, boxes, scores, iou_threshold=0.5, top_k=400):
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes = boxes[idx]

        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = torch.max(iou, dim=0)

        # Now just filter out the ones higher than the threshold
        idx_out = idx[iou_max <= iou_threshold]
        
        return idx_out, idx_out.size(0)

        
                
