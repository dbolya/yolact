import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from utils import timer

from data import get_cfg
cfg = get_cfg()


class Detect(Function):
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

    def forward(self, loc_data, conf_data, mask_data, prior_data):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_size**2]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_size**2)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        if cfg.force_cpu_detect:
            loc_data = loc_data.cpu()
            conf_data = conf_data.cpu()
            mask_data = mask_data.cpu()
            prior_data = prior_data.cpu()

        with timer.env('Detect'):
            num = loc_data.size(0)  # batch size
            num_priors = prior_data.size(0)
            output = torch.zeros(num, self.top_k, 1 + 1 + 4 + mask_data.size(2))
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)

            # Decode predictions into bboxes.
            for i in range(num):
                decoded_boxes = decode(loc_data[i], prior_data)
                
                if self.cross_class_nms:
                    self.detect_cross_class(i, conf_preds, decoded_boxes, mask_data, output)
                else:
                    self.detect_per_class(i, conf_preds, decoded_boxes, mask_data, output)
        
        return output

    def detect_cross_class(self, batch_idx, conf_preds, decoded_boxes, mask_data, output):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        conf_scores, class_labels = torch.max(conf_preds[batch_idx, 1:], 0)

        c_mask = conf_scores.gt(self.conf_thresh)
        scores = conf_scores[c_mask]
        classes = class_labels[c_mask]
        
        if scores.size(0) == 0:
            return
        
        l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
        boxes = decoded_boxes[l_mask].view(-1, 4)
        masks = mask_data[batch_idx, l_mask[:, 0], :]
        # idx of highest scoring and non-overlapping boxes per class
        ids, count = nms(boxes, scores, self.nms_thresh, self.top_k, force_cpu=cfg.force_cpu_nms)
        ids = ids[:count]
        output[batch_idx, :count] = \
            torch.cat((classes[ids].unsqueeze(1).float(), scores[ids].unsqueeze(1), boxes[ids], masks[ids]), 1)

    def detect_per_class(self, batch_idx, conf_preds, decoded_boxes, mask_data, output):
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
            # idx of highest scoring and non-overlapping boxes per class
            ids, count = nms(boxes, scores, self.nms_thresh, self.top_k, force_cpu=cfg.force_cpu_nms)
            ids = ids[:count]
            classes = torch.ones(count, 1).float()*(cl-1)
            tmp_output[cl-1, :count] = \
                torch.cat((classes, scores[ids].unsqueeze(1), boxes[ids], masks[ids]), 1)

        # Pool all the outputs together regardless of class and pick the top_k
        tmp_output = tmp_output.view(-1, output.size(2))
        _, idx = tmp_output[:, 1].sort(0, descending=True)
        output[batch_idx, :, :] = tmp_output[idx[:self.top_k], :]

        
                
