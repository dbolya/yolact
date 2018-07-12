from data import coco as cfg
from data import COCO_ROOT, COCODetection, MEANS, COLORS, COCO_CLASSES
from ssd import build_ssd
from utils.augmentations import BaseTransform
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard
from utils import timer
from utils.functions import sanitize_coordinates
from pycocotools import cocoeval

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle

import matplotlib.pyplot as plt
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
                    help='Whether or not to display thin lines representing gt bboxes in addition to the predicted ones')
parser.add_argument('--display_scores', default=False, type=str2bool,
                    help='Whether or not to display scores in addition to classes')
parser.add_argument('--display', default=False, type=str2bool,
                    help='Whether or not to display the qualitative results')
parser.add_argument('--shuffle', default=False, type=str2bool,
                    help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
parser.add_argument('--ap_data_file', default='ap_data.pkl', type=str,
                    help='In quantitative mode, the file to save detections before calculating mAP.')
parser.add_argument('--resume', dest='resume', action='store_true')
parser.set_defaults(resume=False)
parser.add_argument('--max_images', default=-1, type=int,
                    help='The maximum number of images from the dataset to consider. Use -1 for all.')

args = parser.parse_args()

iou_thresholds = [x / 100 for x in range(50, 100, 5)]

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

def prep_display(dets, img, gt, gt_masks, h, w):
    gt_bboxes = torch.FloatTensor(gt[:, :4]).cpu()
    gt_bboxes[:, [0, 2]] *= w
    gt_bboxes[:, [1, 3]] *= h
    
    img_numpy = cv2.resize((img.permute(1, 2, 0).cpu().numpy() / 255.0 + np.array(MEANS) / 255.0).astype(np.float32), (w, h))
    img_numpy = np.clip(img_numpy, 0, 1)
    
    timer.start('Postprocessing')
    
    classes = dets[:, 0]
    boxes = dets[:, 2:6]
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    scores = list(dets[:, 1].cpu().numpy())
    masks = dets[:, 6:].cpu().numpy()

    score_idx = list(range(len(scores)))
    score_idx.sort(key=lambda x: -scores[x])

    all_boxes = []
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

        # if box_obj['score'] < 0.1:
        #     continue

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
        
        text_str = '%s (%.2f)' % (box_obj['class'],box_obj['score']) if args.display_scores else box_obj['class']
        cv2.putText(img_numpy, text_str, text_pt, cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2, cv2.LINE_AA)
    # timer.stop('Drawing Image')
    
    timer.print_stats()

    return img_numpy

def mask_iou(mask1, mask2):
    """ Inputs inputs are matricies of size _ x N. Output is size _1 x _2."""
    timer.start('Mask IoU')

    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    ret = intersection / union
    timer.stop('Mask IoU')
    return ret.cpu()

def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w):
    """ Returns a list of APs for this image, wich each element being for a class  """
    timer.start('Prepare gt')
    gt_boxes = torch.Tensor(gt[:, :4])
    gt_boxes[:, [0, 2]] *= w
    gt_boxes[:, [1, 3]] *= h
    gt_classes = list(gt[:, 4].astype(int))
    gt_masks = torch.Tensor(gt_masks).view(-1, h*w)
    timer.stop('Prepare gt')

    timer.start('Sort')
    _, sort_idx = dets[:, 1].sort(0, descending=True)
    dets = dets[sort_idx, :]
    timer.stop('Sort')

    timer.start('Prepare pred')
    classes = list(dets[:, 0].cpu().numpy().astype(int))
    scores = list(dets[:, 1].cpu().numpy().astype(float))
    boxes = dets[:, 2:6]
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = torch.stack((x1, y1, x2, y2), dim=1)
    masks = dets[:, 6:].cpu().numpy()
    full_masks = torch.zeros(masks.shape[0], h, w)
    timer.stop('Prepare pred')

    # Scale up the predicted masks to be comparable to the gt masks
    timer.start('Scale masks')
    for i in range(masks.shape[0]):
        x1, y1, x2, y2 = boxes[i, :].cpu().numpy().astype(np.int32)

        mask_w = x2 - x1
        mask_h = y2 - y1

        # I don't know how this can happen (since they're sanitized to begin with), but it can
        if mask_w * mask_h <= 0 or mask_w < 0:
            continue

        pred_mask = masks[i, :].reshape(cfg['mask_size'], cfg['mask_size'], 1)
        local_mask = cv2.resize(pred_mask, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
        local_mask = (local_mask > np.average(local_mask)).astype(np.float32)
        full_masks[i, y1:y2, x1:x2] = torch.Tensor(local_mask)

    masks = full_masks.view(-1, h*w)
    timer.stop('Scale masks')
    
    num_pred = len(classes)
    num_gt   = len(gt_classes)

    mask_iou_cache = mask_iou(masks, gt_masks)
    bbox_iou_cache = jaccard(boxes.float(), gt_boxes.float())

    iou_types = [
        ('box',  lambda i,j: bbox_iou_cache[i, j].item()),
        ('mask', lambda i,j: mask_iou_cache[i, j].item())
    ]

    timer.start('Main loop')
    for _class in set(classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func in iou_types:
                gt_used = [False] * len(gt_classes)
                
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in range(num_pred):
                    if classes[i] != _class:
                        continue
                    
                    is_true = False
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        timer.stop('Main loop')
                        iou = iou_func(i, j)
                        timer.start('Main loop')

                        if iou > iou_threshold:
                            gt_used[j] = True
                            is_true = True
                            break
                    
                    ap_obj.push(scores[i], is_true)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """
        last_recall = 0
        num_true  = 0
        num_false = 0
        ap = 0

        if self.num_gt_positives == 0:
            return 0

        self.data_points.sort(key=lambda x: -x[0])

        for datum in self.data_points:
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            ap += precision * (recall - last_recall)
            last_recall = recall
        
        return ap




def evaluate(net, dataset):
    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else args.max_images

    try:
        if not args.display:
            # For each class and iou, stores tuples (score, isPositive)
            # Index ap_data[type][iouIdx][classIdx]
            ap_data = {
                'box' : [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds],
                'mask': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds]
            }
            progress_bar = ProgressBar(30, dataset_size)
            print()

        dataset_indices = list(range(dataset_size))
        if args.shuffle:
            random.shuffle(dataset_indices)
        
        for i, it in zip(dataset_indices, range(dataset_size)):
            timer.reset()

            # timer.start('Load Data')
            img, gt, gt_masks, h, w = dataset.pull_item(i)
            # timer.stop('Load Data')

            batch = Variable(img.unsqueeze(0))
            if args.cuda:
                batch = batch.cuda()

            preds = net(batch).data
            
            timer.start('Select preds')
            dets = preds[0, :, :]
            non_zero_score_mask = dets[:, 1].gt(0.0).expand(dets.size(1), dets.size(0)).t()
            dets = torch.masked_select(dets, non_zero_score_mask).view(-1, dets.size(1))
            timer.stop('Select preds')
            if dets.size(0) == 0:
                print('Warning: No detection for img idx %d.' % i)
                continue

            # Perform the meat of the operation here
            if args.display:
                img_numpy = prep_display(dets, img, gt, gt_masks, h, w)
            else:
                prep_metrics(ap_data, dets, img, gt, gt_masks, h, w)
            
            if it > 1:
                frame_times.add(timer.total_time())
            
            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(np.clip(img_numpy, 0, 1))
                plt.show()
            else:
                if it > 1: fps = 1 / frame_times.get_avg()
                else: fps = 0
                progress = (it+1) / dataset_size * 100
                progress_bar.set_val(it+1)
                print('Processing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                    % (repr(progress_bar), it+1, dataset_size, progress, fps), end='\r')
                # timer.print_stats()

        if not args.display:
            print()
            print('Saving data...')
            with open(args.ap_data_file, 'wb') as f:
                pickle.dump(ap_data, f)

            calc_map(ap_data)

    except KeyboardInterrupt:
        print('Stopping...')

def calc_map(ap_data):
    print('Calculating mAP...')
    aps = {'box': [], 'mask': []}

    for _class in range(len(COCO_CLASSES)):
        for iouIdx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iouIdx][_class]

                if not ap_obj.is_empty():
                    aps[iou_type].append(ap_obj.get_ap())

    print('BBox mAP: %.4f' % (sum(aps['box'])  / len(aps['box'])))
    print('Mask mAP: %.4f' % (sum(aps['mask']) / len(aps['mask'])))


if __name__ == '__main__':
    
    if args.resume and not args.display:   
        if args.resume:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

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


