"""
Runs the coco-supplied cocoeval script to evaluate detections
outputted by using the output_coco_json flag in eval.py.
"""


import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parser = argparse.ArgumentParser(description='COCO Detections Evaluator')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str)
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str)
parser.add_argument('--gt_ann_file',   default='data/coco/annotations/instances_val2017.json', type=str)
parser.add_argument('--eval_type',     default='both', choices=['bbox', 'mask', 'both'], type=str)
args = parser.parse_args()



if __name__ == '__main__':

	eval_bbox = (args.eval_type in ('bbox', 'both'))
	eval_mask = (args.eval_type in ('mask', 'both'))

	print('Loading annotations...')
	gt_annotations = COCO(args.gt_ann_file)
	if eval_bbox:
		bbox_dets = gt_annotations.loadRes(args.bbox_det_file)
	if eval_mask:
		mask_dets = gt_annotations.loadRes(args.mask_det_file)

	if eval_bbox:
		print('\nEvaluating BBoxes:')
		bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()
	
	if eval_mask:
		print('\nEvaluating Masks:')
		bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()



