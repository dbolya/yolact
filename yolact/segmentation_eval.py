import os

import cv2
import imutils
import numpy as np
import torch

from yolact.data import cfg, set_cfg
from yolact.utils.augmentations import FastBaseTransform
from yolact.utils.functions import SavePath
from yolact.yolact_model import Yolact
from yolact.pipeline_utils import prep_display

package = "imutils"

try:
    __import__(package)
except:
    os.system("pip3 install " + package)


class Segment:
    cfg.mask_proto_debug = False

    def __init__(self, weights: str):
        """
        :weights : weights file path
        """
        self.weights = weights
        self.model_path = SavePath.from_str(weights)
        config = self.model_path.model_name + '_config'
        set_cfg(config)

    @staticmethod
    def find_center(mask: np.ndarray):
        """
        :mask : binarized mask of prediction
        """
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        M = cv2.moments(cnts[0])
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
        except ZeroDivisionError:
            pass

    @staticmethod
    def adjust_centers(center: tuple, box_coords: np.ndarray):
        """
        Returns the true center coordinates with respect to the entire mask
        """
        return box_coords[0] + center[0], box_coords[1] + center[1]

    def predict(self, image_array: np.ndarray):
        """
        :image_path : image numpy array
        Format of returned boxes is [x1,y1,x2,y2], individual centers are tuples
        :return entire mask, individual masks, boxes, centers
        """
        with torch.no_grad():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            frame = torch.from_numpy(image_array).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            net = Yolact()
            net.detect.use_fast_nms = True
            net.detect.use_cross_class_nms = True
            net.load_weights(self.weights)
            net.eval()
            preds = net(batch)
            mask_entire, boxes = prep_display(preds, frame, None, None, undo_transform=False)
            if len(boxes) < 1:
                return mask_entire, None, None, None
            mask_dict = {}
            centers_dict = {}
            boxes_dict = {}
            for index in range(len(boxes)):
                current_box = boxes[index]
                mask_dict[index] = mask_entire[current_box[1]:current_box[3], current_box[0]:current_box[2]]
                center = Segment.find_center(mask_dict[index])
                if not center:
                    adjusted_center = None
                else:
                    adjusted_center = Segment.adjust_centers(center, current_box)
                centers_dict[index] = adjusted_center
                boxes_dict[index] = current_box

            return mask_entire, mask_dict, centers_dict, boxes_dict
