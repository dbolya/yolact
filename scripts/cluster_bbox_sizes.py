"""
This script compiles all the bounding boxes in the training data and
clusters them for each convout resolution on which they're used.

Run this script from the Yolact root directory.
"""

import os.path as osp
import json, pickle
import sys

import numpy as np
import sklearn.cluster as cluster

dump_file = 'weights/bboxes.pkl'
max_size = 550

num_scale_clusters = 5
num_aspect_ratio_clusters = 3

def to_relative(bboxes):
	return bboxes[:, 2:4] / bboxes[:, :2]

def process(bboxes):
	return to_relative(bboxes) * max_size

if __name__ == '__main__':
		
	with open(dump_file, 'rb') as f:
		bboxes = pickle.load(f)

	bboxes = np.array(bboxes)
	bboxes = process(bboxes)
	bboxes = bboxes[(bboxes[:, 0] > 1) * (bboxes[:, 1] > 1)]

	scale  = np.sqrt(bboxes[:, 0] * bboxes[:, 1]).reshape(-1, 1)

	clusterer = cluster.KMeans(num_scale_clusters, random_state=99, n_jobs=4)
	assignments = clusterer.fit_predict(scale)
	counts = np.bincount(assignments)

	cluster_centers = clusterer.cluster_centers_

	center_indices = list(range(num_scale_clusters))
	center_indices.sort(key=lambda x: cluster_centers[x, 0])

	for idx in center_indices:
		center = cluster_centers[idx, 0]
		boxes_for_center = bboxes[assignments == idx]
		aspect_ratios = (boxes_for_center[:,0] / boxes_for_center[:,1]).reshape(-1, 1)

		c = cluster.KMeans(num_aspect_ratio_clusters, random_state=idx, n_jobs=4)
		ca = c.fit_predict(aspect_ratios)
		cc = np.bincount(ca)

		c = list(c.cluster_centers_.reshape(-1))
		cidx = list(range(num_aspect_ratio_clusters))
		cidx.sort(key=lambda x: -cc[x])

		# import code
		# code.interact(local=locals())

		print('%.3f (%d) aspect ratios:' % (center, counts[idx]))
		for idx in cidx:
			print('\t%.2f (%d)' % (c[idx], cc[idx]))
		print()
		# exit()


