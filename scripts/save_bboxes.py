""" This script transforms and saves bbox coordinates into a pickle object for easy loading. """


import os.path as osp
import json, pickle
import sys

import numpy as np

COCO_ROOT = osp.join('.', 'data/coco/')

annotation_file = 'instances_train2014.json'
annotation_path = osp.join(COCO_ROOT, 'annotations/', annotation_file)

dump_file = 'weights/bboxes.pkl'

with open(annotation_path, 'r') as f:
	annotations_json = json.load(f)

annotations = annotations_json['annotations']
images = annotations_json['images']
images = {image['id']: image for image in images}
bboxes = []

for ann in annotations:
	image = images[ann['image_id']]
	w,h = (image['width'], image['height'])
	
	if 'bbox' in ann:
		bboxes.append([w, h] + ann['bbox'][2:])

with open(dump_file, 'wb') as f:
	pickle.dump(bboxes, f)
