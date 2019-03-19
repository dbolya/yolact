# -*- coding: utf-8 -*-
import torch
from utils import timer

from data import cfg

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b, iscrowd=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    if iscrowd:
        return inter / area_a
    else:
        return inter / union  # [A,B]

def change(gt, priors):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf
    
    Input should be in point form (xmin, ymin, xmax, ymax).

    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for 
    """
    num_priors = priors.size(0)
    num_gt     = gt.size(0)

    gt_w = (gt[:, 2] - gt[:, 0])[:, None].expand(num_gt, num_priors)
    gt_h = (gt[:, 3] - gt[:, 1])[:, None].expand(num_gt, num_priors)

    gt_mat =     gt[:, None, :].expand(num_gt, num_priors, 4)
    pr_mat = priors[None, :, :].expand(num_gt, num_priors, 4)

    diff = gt_mat - pr_mat
    diff[:, :, 0] /= gt_w
    diff[:, :, 2] /= gt_w
    diff[:, :, 1] /= gt_h
    diff[:, :, 3] /= gt_h

    return -torch.sqrt( (diff ** 2).sum(dim=2) )




def match(pos_thresh, neg_thresh, truths, priors, labels, crowd_boxes, loc_t, conf_t, idx_t, idx, loc_data):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        idx: (int) current batch index.
        loc_data: (tensor) The predicted bbox regression coordinates for this batch.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    decoded_priors = decode(loc_data, priors) if cfg.use_prediction_matching else point_form(priors)
    
    # Size [num_objects, num_priors]
    overlaps = jaccard(truths, decoded_priors) if not cfg.use_change_matching else change(truths, decoded_priors)

    # Size [num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    # Size [num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    
    # For the best prior for each gt object, set its overlap to 2. This ensures
    # that it won't get thresholded out in the threshold step even if the IoU is
    # under the negative threshold. This is because we want at least one anchor
    # to match with each ground truth or else we'd be wasting training data.
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    # Set the index of the pair (prior, gt) we set the overlap for above.
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]            # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1           # Shape: [num_priors]

    conf[best_truth_overlap < pos_thresh] = -1  # label as neutral
    conf[best_truth_overlap < neg_thresh] =  0  # label as background

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1

    loc = encode(matches, priors)
    loc_t[idx]  = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf   # [num_priors] top class label for each prior
    idx_t[idx]  = best_truth_idx # [num_priors] indices for lookup


def encode(matched, priors):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.
    
    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4]
        - priors:  The tensor of all priors with shape [num_priors, 4]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """

    if cfg.use_yolo_regressors:
        # Exactly the reverse of what we did in decode
        # In fact encode(decode(x, p), p) should be x
        boxes = center_size(matched)

        loc = torch.cat((
            boxes[:, :2] - priors[:, :2],
            torch.log(boxes[:, 2:] / priors[:, 2:])
        ), 1)

        return loc
    else:
        variances = [0.1, 0.2]

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def decode(loc, priors):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    
    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]
    
    Returns: A tensor of decoded relative coordinates in point form 
             form with size [num_priors, 4]
    """

    if cfg.use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = torch.cat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * torch.exp(loc[:, 2:])
        ), 1)

        return point_form(boxes)
    else:
        variances = [0.1, 0.2]
        
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes



def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200, force_cpu=True):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    cuda_enabled = boxes.is_cuda

    if force_cpu:
        boxes = boxes.cpu()
        scores = scores.cpu()

    timer.start('NMS')
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
        
    if cuda_enabled:
        keep = keep.cuda()

    timer.stop()
    return keep, count


def sanitize_coordinates(_x1, _x2, img_size, cast=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 *= img_size
    _x2 *= img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1-1, min=0)
    x2 = torch.clamp(x2+1, max=img_size)

    return x1, x2

def crop(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    with torch.no_grad():
        h, w, n = masks.size()
        boxes = boxes.clone() # Some in-place stuff goes on here
        x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=True)
        y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=True)

        rows = torch.arange(w, device=masks.device)[None, :, None].expand(h, w, n)
        cols = torch.arange(h, device=masks.device)[:, None, None].expand(h, w, n)
        
        masks_left  = rows >= x1[None, None, :]
        masks_right = rows <  x2[None, None, :]
        masks_up    = cols >= y1[None, None, :]
        masks_down  = cols <  y2[None, None, :]
        
        crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask.float()

