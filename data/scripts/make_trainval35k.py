import json
import os

annotations_path = 'data/coco/annotations/annotations/instances_%s2014.json'
result_path      = 'data/coco/annotations/instances_trainval35k.json'
fields_to_combine = ('images', 'licenses', 'annotations', 'categories')

print('Loading training instances...')
with open(annotations_path % 'train', 'r') as train_file:
	train_json = json.load(train_file)

print('Loading validation instances...')
with open(annotations_path % 'val', 'r') as val_file:
	val_json = json.load(val_file)

print('Combining validation and test...')
for field in fields_to_combine:
	for entry in val_json[field]:
		train_json[field].append(entry)

print('Saving result...')
with open(result_path, 'w') as out_file:
	json.dump(train_json, out_file)
