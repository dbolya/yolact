"""
This script compiles all the bounding boxes in the training data and
clusters them for each convout resolution on which they're used.

Run this script from the Yolact root directory.
"""

import os.path as osp
import json, pickle
import sys
from math import sqrt
from itertools import product
import torch
import random

import numpy as np

dump_file = 'weights/bboxes.pkl'
aug_file  = 'weights/bboxes_aug.pkl'

use_augmented_boxes = True


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

# Also convert to point form
def to_relative(bboxes):
    return np.concatenate((bboxes[:, 2:4] / bboxes[:, :2], (bboxes[:, 2:4] + bboxes[:, 4:]) / bboxes[:, :2]), axis=1)


def make_priors(conv_size, scales, aspect_ratios):
    prior_data = []
    conv_h = conv_size[0]
    conv_w = conv_size[1]

    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        x = (i) / conv_w
        y = (j) / conv_h
        
        for scale, ars in zip(scales, aspect_ratios):
            for ar in ars:
                w = scale * ar / conv_w
                h = scale / ar / conv_h

                # Point form
                prior_data += [x, y, x + w, y + h]
    
    return np.array(prior_data).reshape(-1, 4)

# fixed_ssd_config
# scales = [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [2.1, 2.37], [1.8, 1.92]]
# aspect_ratios = [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3, 3] ]
# conv_sizes = [(35, 35), (18, 18), (9, 9), (5, 5), (3, 3), (2, 2)]

scales = [[1.68, 2.91],
          [2.95, 2.22, 0.84],
          [2.23, 2.17, 3.12],
          [0.76, 1.94, 2.72],
          [2.10, 2.65],
          [1.80, 1.92]]
aspect_ratios = [[[0.72, 0.96], [0.68, 1.17]],
                 [[1.28, 0.66], [0.63, 1.23], [0.89, 1.40]],
                 [[2.05, 1.24], [0.57, 0.83], [0.61, 1.15]],
                 [[1.00, 2.21], [0.47, 1.60], [1.44, 0.79]],
                 [[1.00, 1.41, 0.71, 1.73, 0.58], [1.08]],
                 [[1.00, 1.41, 0.71, 1.73, 0.58], [1.00]]]
conv_sizes = [(35, 35), (18, 18), (9, 9), (5, 5), (3, 3), (2, 2)]

# yrm33_config
# scales = [ [5.3] ] * 5
# aspect_ratios = [ [[1, 1/2, 2]] ]*5
# conv_sizes = [(136, 136), (67, 67), (33, 33), (16, 16), (8, 8)]


SMALL = 0
MEDIUM = 1
LARGE = 2

if __name__ == '__main__':
        
    with open(dump_file, 'rb') as f:
        bboxes = pickle.load(f)

    sizes = []
    smalls = []
    for i in range(len(bboxes)):
        area = bboxes[i][4] * bboxes[i][5]
        if area < 32 ** 2:
            sizes.append(SMALL)
            smalls.append(area)
        elif area < 96 ** 2:
            sizes.append(MEDIUM)
        else:
            sizes.append(LARGE)

    # Each box is in the form [im_w, im_h, pos_x, pos_y, size_x, size_y]
    
    if use_augmented_boxes:
        with open(aug_file, 'rb') as f:
            bboxes_rel = pickle.load(f)
    else:
        bboxes_rel = to_relative(np.array(bboxes))
        

    with torch.no_grad():
        sizes = torch.Tensor(sizes)

        anchors = [make_priors(cs, s, ar) for cs, s, ar in zip(conv_sizes, scales, aspect_ratios)]
        anchors = np.concatenate(anchors, axis=0)
        anchors = torch.Tensor(anchors).cuda()

        bboxes_rel = torch.Tensor(bboxes_rel).cuda()
        perGTAnchorMax = torch.zeros(bboxes_rel.shape[0]).cuda()

        chunk_size = 1000
        for i in range((bboxes_rel.size(0) // chunk_size) + 1):
            start = i * chunk_size
            end   = min((i + 1) * chunk_size, bboxes_rel.size(0))
            
            ious = jaccard(bboxes_rel[start:end, :], anchors)
            maxes, maxidx = torch.max(ious, dim=1)

            perGTAnchorMax[start:end] = maxes
    

        hits = (perGTAnchorMax > 0.5).float()

        print('Total recall: %.2f' % (torch.sum(hits) / hits.size(0) * 100))
        print()

        for i, metric in zip(range(3), ('small', 'medium', 'large')):
            _hits = hits[sizes == i]
            _size = (1 if _hits.size(0) == 0 else _hits.size(0))
            print(metric + ' recall: %.2f' % ((torch.sum(_hits) / _size) * 100))



