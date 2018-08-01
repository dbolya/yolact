"""
Instead of clustering bbox widths and heights, this script
directly optimizes average IoU across the training set given
the specified number of anchor boxes.

Run this script from the Yolact root directory.
"""

import pickle

import numpy as np
from scipy.optimize import minimize

dump_file = 'weights/bboxes.pkl'

# Load widths and heights from a dump file. Obtain this with
# python3 scripts/cluster_bbox_sizes.py save
with open(dump_file, 'rb') as f:
	bboxes = np.array(pickle.load(f))

def iou(box1, box2):
	"""
	Computes the iou between each element of box1 and each element of box2.
	Note that this assumes both boxes have the same center point.

	Args:
		- box1: shape [a, 2] an array of (w, h)
		- box2: shape [b, 2] an array of (w, h)
	
	Returns:
		- pairwise ious with shape [a, b]
	"""
	a = box1.shape[0]
	b = box2.shape[0]

	box1 = np.expand_dims(box1, axis=1).repeat(b, axis=1)
	box2 = np.expand_dims(box2, axis=0).repeat(a, axis=0)

	mins = np.fmin(box1, box2)

	area1 = box1[:, :, 0] * box1[:, :, 1]
	area2 = box2[:, :, 0] * box2[:, :, 1]
	intersection = mins[:, :, 0] * mins[:, :, 1]

	union = area1 + area2 - intersection

	return intersection / union


def avg_iou(ars:list):
	# Create boxes by first making all the scales and then applying the aspect ratio
	test_boxes = np.array([[[2**n]*2 for n in range(3,10)]]*len(ars), dtype=np.float32)
	for i in range(len(ars)):
		test_boxes[i, :, 0] *= ars[i]
		test_boxes[i, :, 1] /= ars[i]
	test_boxes = test_boxes.reshape(-1, 2)

	ious = iou(test_boxes, bboxes)

	return np.average(np.max(ious, axis=0))

# TODO: These scripts don't take into account the fact that we're resizing the boxes. Fix that.
if __name__ == '__main__':
	res = minimize(lambda x: -avg_iou(x), [1.19613237, 0.70644902, 0.1], method='Nelder-Mead')
	print(res.x)
	print('%.4f' % avg_iou(res.x))

# Optimization results:
# mIoU = 0.5635  [0.91855469]
# mIoU = 0.6416  [1.19613237 0.70644902]
# mIoU = 0.6703  [1.45072431 0.96152597 0.63814035]
# mIoU = 0.6857  [1.13719648 0.83178079 0.59529905 1.68513610]


