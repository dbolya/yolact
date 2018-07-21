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

COCO_ROOT = osp.join('.', 'data/coco/')

annotation_file = 'instances_train2014.json'
annotation_path = osp.join(COCO_ROOT, 'annotations/', annotation_file)

dump_file = 'weights/bboxes.pkl'

if __name__ == '__main__':
	if len(sys.argv) < 2 or sys.argv[1].lower() not in ('save', 'load'):
		print('Usage: python3 cluster_bbox_sizes.py [save|load]')
		print('Save first then load.')
		exit()

	if sys.argv[1].lower() == 'save':
		with open(annotation_path, 'r') as f:
			annotations = json.load(f)['annotations']

		bboxes = []

		for ann in annotations:
			if 'bbox' in ann:
				bboxes.append(ann['bbox'][2:])

		print(bboxes[:10])

		with open(dump_file, 'wb') as f:
			pickle.dump(bboxes, f)
	else:
		
		with open(dump_file, 'rb') as f:
			bboxes = pickle.load(f)

		bboxes = np.array(bboxes)
		scale  = np.sqrt(bboxes[:, 0] * bboxes[:, 1]).reshape(-1, 1)

		num_clusters = 6

		clusterer = cluster.KMeans(num_clusters, random_state=99, n_jobs=4)
		assignments = clusterer.fit_predict(np.log(scale + 0.0001))
		counts = np.bincount(assignments)

		cluster_centers = np.exp(clusterer.cluster_centers_)

		center_indices = list(range(num_clusters))
		center_indices.sort(key=lambda x: cluster_centers[x, 0])

		for idx in center_indices:
			center = cluster_centers[idx, 0]
			# boxes_for_center = bboxes
			boxes_for_center = bboxes[assignments == idx]
			aspect_ratios = np.log(((boxes_for_center[:,0]+0.001) / (boxes_for_center[:,1]+0.001)).reshape(-1, 1))

			c = cluster.KMeans(num_clusters, random_state=idx, n_jobs=4)
			ca = c.fit_predict(aspect_ratios)
			cc = np.bincount(ca)

			c = list(np.exp(c.cluster_centers_.reshape(-1)))
			cidx = list(range(num_clusters))
			cidx.sort(key=lambda x: -cc[x])

			print('%.2f (%d) aspect ratios:' % (center, counts[idx]))
			for idx in cidx:
				print('\t%.2f (%d)' % (c[idx], cc[idx]))
			print()
			# exit()


