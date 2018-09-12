import json
import os
import sys
from collections import defaultdict

usage_text = """
This script creates a coco annotation file by mixing one or more existing annotation files.

Usage: python data/scripts/mix_sets.py output_name [set1 range1 [set2 range2 [...]]]

To use, specify the output annotation name and any number of set + range pairs, where the sets
are in the form instances_<set_name>.json and ranges are python-evalable ranges. The resulting
json will be spit out as instances_<output_name>_.json in the same folder as the input sets.

Note that the "_" is to tell our COCODetection data loader that the image file names contain folders.
This is so you don't have to copy all the images into the same directory or use symlinks.

For instance,
    python data/scripts/mix_sets.py trainval35k train2014 : val2014 :35000

This will create an instance_trainval35k_.json file with all images and corresponding annotations
from train2014 and the first 35000 images from val2014.

You can also specify only one set:
    python data/scripts/mix_sets.py minival5k val2014 -5000:

This will take the last 5k images from val2014 and put it in instances_minival5k_.json.
"""

annotations_path = 'data/coco/annotations/instances_%s.json'
fields_to_combine = ('images', 'annotations')
fields_to_steal   = ('info', 'categories', 'licenses')

if __name__ == '__main__':
	if len(sys.argv) < 4 or len(sys.argv) % 2 != 0:
		print(usage_text)
		exit()

	out_name = sys.argv[1]
	sets = sys.argv[2:]
	sets = [(sets[2*i], sets[2*i+1]) for i in range(len(sets)//2)]

	out = {x: [] for x in fields_to_combine}

	for idx, (set_name, range_str) in enumerate(sets):
		print('Loading set %s...' % set_name)
		with open(annotations_path % set_name, 'r') as f:
			set_json = json.load(f)

		# "Steal" some fields that don't need to be combined from the first set
		if idx == 0:
			for field in fields_to_steal:
				out[field] = set_json[field]
		
		print('Building image index...')
		image_idx = {x['id']: x for x in set_json['images']}

		# Add the parent path to the image file name, so you don't have to symlink them
		for val in image_idx.values():
			val['file_name'] = set_name + '/' + val['file_name']

		print('Collecting annotations...')
		anns_idx = defaultdict(lambda: [])

		for ann in set_json['annotations']:
			anns_idx[ann['image_id']].append(ann)

		export_ids = list(anns_idx.keys())
		export_ids.sort()
		export_ids = eval('export_ids[%s]' % range_str, {}, {'export_ids': export_ids})

		print('Adding %d images...' % len(export_ids))
		for _id in export_ids:
			out['images'].append(image_idx[_id])
			out['annotations'] += anns_idx[_id]

		print('Done.\n')

	print('Saving result...')
	with open(annotations_path % (out_name + '_'), 'w') as out_file:
		json.dump(out, out_file)
