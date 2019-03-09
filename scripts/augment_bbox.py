
import os.path as osp
import json, pickle
import sys
from math import sqrt
from itertools import product
import torch
from numpy import random

import numpy as np


max_image_size = 550
augment_idx = 0
dump_file = 'weights/bboxes_aug.pkl'
box_file = 'weights/bboxes.pkl'

def augment_boxes(bboxes):
	bboxes_rel = []
	for box in bboxes:
		bboxes_rel.append(prep_box(box))
	bboxes_rel = np.concatenate(bboxes_rel, axis=0)

	with open(dump_file, 'wb') as f:
		pickle.dump(bboxes_rel, f)

def prep_box(box_list):
	global augment_idx
	boxes = np.array([box_list[2:]], dtype=np.float32)

	# Image width and height
	width, height = box_list[:2]

	# To point form
	boxes[:, 2:] += boxes[:, :2]


	# Expand
	ratio = random.uniform(1, 4)
	left = random.uniform(0, width*ratio - width)
	top = random.uniform(0, height*ratio - height)

	height *= ratio
	width  *= ratio

	boxes[:, :2] += (int(left), int(top))
	boxes[:, 2:] += (int(left), int(top))


	# RandomSampleCrop
	height, width, boxes = random_sample_crop(height, width, boxes)


	# RandomMirror
	if random.randint(0, 2):
		boxes[:, 0::2] = width - boxes[:, 2::-2]

	
	# Resize
	boxes[:, [0, 2]] *= (max_image_size / width)
	boxes[:, [1, 3]] *= (max_image_size / height)
	width = height = max_image_size


	# ToPercentCoords
	boxes[:, [0, 2]] /= width
	boxes[:, [1, 3]] /= height

	if augment_idx % 50000 == 0:
		print('Current idx: %d' % augment_idx)

	augment_idx += 1

	return boxes




sample_options = (
	# using entire original input image
	None,
	# sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
	(0.1, None),
	(0.3, None),
	(0.7, None),
	(0.9, None),
	# randomly sample a patch
	(None, None),
)

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def random_sample_crop(height, width, boxes=None):
	global sample_options
	
	while True:
		# randomly choose a mode
		mode = random.choice(sample_options)
		if mode is None:
			return height, width, boxes

		min_iou, max_iou = mode
		if min_iou is None:
			min_iou = float('-inf')
		if max_iou is None:
			max_iou = float('inf')

		for _ in range(50):
			w = random.uniform(0.3 * width, width)
			h = random.uniform(0.3 * height, height)

			if h / w < 0.5 or h / w > 2:
				continue

			left = random.uniform(0, width - w)
			top = random.uniform(0, height - h)

			rect = np.array([int(left), int(top), int(left+w), int(top+h)])
			overlap = jaccard_numpy(boxes, rect)
			if overlap.min() < min_iou and max_iou < overlap.max():
				continue

			centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

			m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
			m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
			mask = m1 * m2

			if not mask.any():
				continue

			current_boxes = boxes[mask, :].copy()
			current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
			current_boxes[:, :2] -= rect[:2]
			current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
			current_boxes[:, 2:] -= rect[:2]

			return h, w, current_boxes


if __name__ == '__main__':
	
	with open(box_file, 'rb') as f:
		bboxes = pickle.load(f)

	augment_boxes(bboxes)
