import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import cfg, mask_type
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from .box_utils import match, crop
import cv2
from datetime import datetime
import os

from utils.functions import make_net

class FastMaskIoUNet(nn.Module):

    def __init__(self):
        super(FastMaskIoUNet, self).__init__()
        input_channels = 1
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net, include_last_relu=True)

    def forward(self, x, target=None):
        cudnn.benchmark = False
        x = self.maskiou_net(x)
        cudnn.benchmark = True
        # global pooling
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

        if self.training:
            maskiou_t = target[0]
            label_t = target[1]
            label_t = label_t[:, None]
            maskiou_p = torch.gather(maskiou_p, dim=1, index=label_t).squeeze()
            loss_i = F.smooth_l1_loss(maskiou_p, maskiou_t, reduction='mean')
            return loss_i * cfg.maskiou_alpha
        else:
            return maskiou_p