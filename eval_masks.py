from data import coco as cfg
from data import COCO_ROOT, COCODetection, MEANS, COLORS, COCO_CLASSES
from ssd import build_ssd
from utils.augmentations import BaseTransform

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time

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
    try:
        for i in range(len(dataset)):
            img, gt, gt_masks, h, w = dataset.pull_item(i)
            img_numpy = cv2.resize((img.permute(1, 2, 0).cpu().numpy() / 255.0 + np.array(MEANS) / 255.0).astype(np.float32), (w, h))

            batch = Variable(img.unsqueeze(0))
            if args.cuda:
                batch = batch.cuda()

            time_start = time.time()
            preds = net(batch).data

            all_boxes = []

            for j in range(1, preds.size(1)):
                dets = preds[0, j, :]
                mask = dets[:, 0].gt(0.0).expand(preds.size(3), dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, preds.size(3))
                if dets.size(0) == 0:
                    continue

                boxes = dets[:, 1:5]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                masks = dets[:, 5:].cpu().numpy()
            
                for k in range(boxes.size(0)):
                    x1, y1, x2, y2 = boxes[k].cpu().numpy().astype(np.int32)
                    all_boxes.append({
                        'score': float(scores[k]),
                        'p1': (x1, y1),
                        'p2': (x2, y2),
                        'class': COCO_CLASSES[j-1],
                        'mask': masks[k, :].reshape(cfg['mask_size'], cfg['mask_size'])
                    })

            all_boxes.sort(key=lambda x: -x['score'])
            
            for j in reversed(range(args.top_k)):
                box_obj = all_boxes[j]
                p1, p2 = (box_obj['p1'], box_obj['p2'])
                mask_w, mask_h = (p2[0] - p1[0], p2[1] - p1[1])
                text_pt = (p1[0], p2[1] - 5)
                color = COLORS[j % len(COLORS)]

                mask = cv2.resize(box_obj['mask'], (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
                mask_alpha = (mask > np.average(mask)).astype(np.float32) * 0.005
                color_np = np.array(color[:3]).reshape(1, 1, 3)
                mask_overlay = np.tile(color_np, (mask.shape[0], mask.shape[1], 1))
                
                cv2.rectangle(img_numpy, p1, p2, color, 2)
                overlay_image_alpha(img_numpy, mask_overlay, p1, mask_alpha)
                cv2.putText(img_numpy, box_obj['class'], text_pt, cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv2.LINE_AA)
            
            plt.imshow(img_numpy)
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

    evaluate(net, dataset)


