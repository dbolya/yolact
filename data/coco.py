from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from pycocotools import mask as maskUtils

COCO_ROOT = osp.join('.', 'data/coco/')
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map

def segm_to_RLE(segm, h, w):
    """ Copied from coco.annToRLE, this is needed for parsing raw ann['segmentation'] objects. """
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, target, width, height, include_mask=False):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                if include_mask:
                    final_box.append(maskUtils.decode(segm_to_RLE(obj['segmentation'], height, width)))
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
                # If include_mask [xmin, ymin, xmax, ymax, label_idx, np.array(h, w) mask]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, root, image_set='trainval35k', transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO', prep_crowds=False):
        sys.path.append(osp.join(root, COCO_API))
        from pycocotools.coco import COCO
        self.root = osp.join(root, IMAGES, image_set)
        self.coco = COCO(osp.join(root, ANNOTATIONS,
                                  INSTANCES_SET.format(image_set)))
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.prep_crowds=prep_crowds

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, crowd = self.pull_item(index)
        return im, (gt, masks)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
        target = self.coco.loadAnns(ann_ids)

        # Filter out annotations with 'iscrowd'=True because they're not for this task
        # But the way COCOEval is implemented, we need crowd annotations for evaluation
        crowd = None
        if self.prep_crowds:
            crowd = [x for x in target if  ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]

        path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(path)
        height, width, _ = img.shape
        
        # Pool all the masks for this image into one [num_objects,height,width] matrix
        masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
        masks = np.vstack(masks)
        masks = masks.reshape(-1, height, width)

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
            
            if self.prep_crowds:
                if len(crowd) > 0:
                    crowd = self.target_transform(crowd, width, height, include_mask=True)
                else: crowd = None

        if self.transform is not None:
            target = np.array(target)
            img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                                                       target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, crowd

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
