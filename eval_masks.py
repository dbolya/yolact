from data import coco as cfg
from data import COCO_ROOT, COCODetection, MEANS, COLORS, COCO_CLASSES
from ssd import build_ssd
from utils.augmentations import BaseTransform
from utils.functions import MovingAverage
from layers.box_utils import jaccard
from utils import timer

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile

import matplotlib.pyplot as plt
import cv2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='YOLACT COCO Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to evaulate model')
parser.add_argument('--coco_root', default=COCO_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cross_class_nms', default=True, type=str2bool,
                    help='Whether to use cross-class nms (faster) or do nms per class')
parser.add_argument('--display_masks', default=True, type=str2bool,
                    help='Whether or not to display masks over bounding boxes')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                    help='Whether or not to display bboxes around masks')
parser.add_argument('--display_gt_bboxes', default=False, type=str2bool,
                    help='Whether or not to display thin lines representing gt bboxes in addition to the predicted ones.')

args = parser.parse_args()

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """
    Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.

    Source: https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

def evaluate(net, dataset):
    frame_times = MovingAverage()
    first_time = True

    try:
        dataset_indices = list(range(len(dataset)))
        # random.shuffle(dataset_indices)
        for i, it in zip(dataset_indices, list(range(len(dataset)))):
            timer.reset()

            # timer.start('Loading Data')
            img, gt, gt_masks, h, w = dataset.pull_item(i)
            # timer.stop('Loading Data')
            
            gt_bboxes = torch.FloatTensor(gt[:, :4]).cpu()
            gt_bboxes[:, 0] *= w
            gt_bboxes[:, 2] *= w
            gt_bboxes[:, 1] *= h
            gt_bboxes[:, 3] *= h

            img_numpy = cv2.resize((img.permute(1, 2, 0).cpu().numpy() / 255.0 + np.array(MEANS) / 255.0).astype(np.float32), (w, h))
            img_numpy = np.clip(img_numpy, 0, 1)

            batch = Variable(img.unsqueeze(0))
            if args.cuda:
                batch = batch.cuda()

            preds = net(batch).data

            timer.start('Postprocessing')
            all_boxes = []
            dets = preds[0, :]

            classes = dets[:, 0]
            boxes = dets[:, 2:6]
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            scores = list(dets[:, 1].cpu().numpy())
            masks = dets[:, 6:].cpu().numpy()

            score_idx = list(range(len(scores)))
            score_idx.sort(key=lambda x: -scores[x])
        
            for k in score_idx[:args.top_k]:
                x1, y1, x2, y2 = boxes[k].cpu().numpy().astype(np.int32)
                
                box_test = boxes[k].cpu().view([-1, 4])
                match_idx = torch.max(jaccard(box_test, gt_bboxes), 1)[1].item()
                    
                all_boxes.append({
                    'score': float(scores[k]),
                    'p1': (x1, y1),
                    'p2': (x2, y2),
                    'class': COCO_CLASSES[classes[k].long().item()],
                    'mask': masks[k, :].reshape(cfg['mask_size'], cfg['mask_size']),
                    'b1': (gt_bboxes[match_idx, 0], gt_bboxes[match_idx, 1]),
                    'b2': (gt_bboxes[match_idx, 2], gt_bboxes[match_idx, 3])
                })

            timer.stop('Postprocessing')

            # timer.start('Drawing Image')
            for j in reversed(range(args.top_k)):
                box_obj = all_boxes[j]
                p1, p2 = (box_obj['p1'], box_obj['p2'])
                mask_w, mask_h = (p2[0] - p1[0], p2[1] - p1[1])
                text_pt = (p1[0], p2[1] - 5)
                color = COLORS[j % len(COLORS)]

                mask = cv2.resize(box_obj['mask'], (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
                mask_alpha = (mask > np.average(mask)).astype(np.float32) * 0.0015
                color_np = np.array(color[:3]).reshape(1, 1, 3)
                mask_overlay = np.tile(color_np, (mask.shape[0], mask.shape[1], 1))
                
                if args.display_bboxes:
                    cv2.rectangle(img_numpy, p1, p2, color, 2)
                
                if args.display_gt_bboxes:
                    cv2.rectangle(img_numpy, box_obj['b1'], box_obj['b2'], color, 1)

                if args.display_masks:
                    overlay_image_alpha(img_numpy, mask_overlay, p1, mask_alpha)
                
                cv2.putText(img_numpy, box_obj['class'], text_pt, cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv2.LINE_AA)
            # timer.stop('Drawing Image')
            
            timer.print_stats()
            
            if it > 1:
                frame_times.add(timer.total_time())
                print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
            
            plt.imshow(np.clip(img_numpy, 0, 1))
            plt.show()
    except KeyboardInterrupt:
        print('Stopping...')



if __name__ == '__main__':
    
    print('Loading model...')
    net = build_ssd('test', 300, cfg['num_classes'])
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()

    dataset = COCODetection(args.coco_root, 'val2014', 
                            BaseTransform(net.size, MEANS))

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    net.detect.cross_class_nms = args.cross_class_nms

    evaluate(net, dataset)


