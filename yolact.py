import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet
import numpy as np

from data.config import get_cfg
from layers import Detect

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage

cfg = get_cfg()

class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of priorbox aspect ratios to consider.
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - num_classes:   The number of classes to consider for classification.
        - mask_size:     The side length of the downsampled predicted mask.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[1], scales=[1],
                       num_classes=cfg.num_classes, mask_size=cfg.mask_size):
        super().__init__()

        self.num_classes = num_classes
        self.mask_size   = mask_size
        self.num_priors  = len(aspect_ratios) * len(scales)

        self.block = Bottleneck(in_channels, out_channels // 4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * (mask_size**2), kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_size**2]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        # The two branches of PM design (c)
        a = self.block(x)
        
        b = self.conv(x)
        b = self.bn(b)
        b = F.relu(b)
        
        # TODO: Possibly switch this out for a product
        x = a + b

        bbox = self.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        mask = self.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_size**2)
        
        # See box_utils.decode for an explaination of this
        bbox[:, :, :2] = F.sigmoid(bbox[:, :, :2]) - 0.5
        bbox[:, :, 0] /= conv_w
        bbox[:, :, 1] /= conv_h
        
        mask = F.sigmoid(mask)
        
        priors = self.make_priors(conv_h, conv_w)

        return (bbox, conf, mask, priors)
    
    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        with timer.env('makepriors'):
            if self.last_conv_size != (conv_w, conv_h):
                # Fancy fast way of doing a cartesian product
                priors = np.array(np.meshgrid(list(range(conv_w)),
                                              list(range(conv_h)),
                                              self.aspect_ratios,
                                              self.scales)).T.reshape(-1, 4)
                
                # The predictions will be in the order conv_h, conv_w, num_priors, but I don't
                # know if meshgrid ordering is deterministic, so let's sort it here to make sure
                # the elements are in this order. aspect_ratios and scales are interchangable
                # because the network can just learn which is which, but conv_h and conv_w orders
                # have to match or the network will get priors for a cell that is in the complete
                # wrong place in the image. Note: the sort order is from last to first (so 1, 0, etc.)
                ind = np.lexsort((priors[:,3],priors[:,2],priors[:,0],priors[:,1]), axis=0)
                priors = priors[ind]
                
                # Priors are in center-size form
                priors[:, [0, 1]] += 0.5

                # Compute the correct width and height of each bounding box
                aspect_ratios = priors[:, 2].copy() # In the form w / h
                scales        = priors[:, 3].copy()
                priors[:, 2] = scales * aspect_ratios # p_w = (scale / h) * w
                priors[:, 3] = scales / aspect_ratios # p_h = (scale / w) * h

                # Make those coordinate relative
                priors[:, [0, 2]] /= conv_w
                priors[:, [1, 3]] /= conv_h
                
                # Cache priors because copying them to the gpu takes time
                self.priors = torch.Tensor(priors)
                self.last_conv_size = (conv_w, conv_h)
        
        return self.priors



class Yolact(ResNet):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    Args:
        - selected_layers: The indices of the conv layers to use for prediction.
        - conv_channels:   The number of output channels for the added blocks.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
        - block:           The resnet block style to use. You probably shouldn't change this.
        - backbone_path:   If not None, the path of the backbone network to load in for training.
    """

    def __init__(self, selected_layers=range(2,7), conv_channels=1024,
                       pred_scales=[[.5, 1]]+[[1]]*3+[[1, 2]],
                       pred_aspect_ratios=[[1.05, 0.62],
                                           [1.29, 0.79, 0.47, 2.33, 0.27],
                                           [1.19, 0.72, 0.43, 2.13, 0.25],
                                           [1.34, 0.84, 0.52, 2.38, 0.30],
                                           [1.40, 0.95, 0.64, 2.16]],
                       block=Bottleneck, backbone_path=None):
        
        self.channels = [] # Populated by self._make_layer
        super().__init__(block, [3, 4, 23, 3])

        # We don't need these where we're going
        del self.fc
        del self.avgpool

        if backbone_path is not None:
            self.load_state_dict(torch.load(backbone_path))

        self.layers = nn.ModuleList([
            self.layer1, # conv2_x
            self.layer2, # conv3_x
            self.layer3, # conv4_x
            self.layer4, # conv5_x
            self._make_layer(block, conv_channels // 4, 1, stride=2),
            self._make_layer(block, conv_channels // 4, 1, stride=2),
            self._make_layer(block, conv_channels // 4, 1, stride=2),
        ])

        self.selected_layers = selected_layers
        self.prediction_layers = nn.ModuleList()

        for idx, layer_idx in enumerate(self.selected_layers):
            self.prediction_layers.append(PredictionModule(self.channels[layer_idx], self.channels[layer_idx],
                                                           aspect_ratios=pred_aspect_ratios[idx], scales=pred_scales[idx]))

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=200, conf_thresh=0.01, nms_thresh=0.45)

        # Now that we've stored them in self.layers, we don't need them under us, so remove these
        del self.layer1
        del self.layer2
        del self.layer3
        del self.layer4
    
    def _make_layer(self, block, planes, blocks, stride=1):
        self.channels.append(planes * block.expansion)
        return super()._make_layer(block, planes, blocks, stride)

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        with timer.env('pass1'):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            outs = []
            for layer in self.layers:
                x = layer(x)
                outs.append(x)

        with timer.env('pass2'):
            pred_outs = ([], [], [], [])
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                p = pred_layer(outs[idx])
                for out, pred in zip(pred_outs, p):
                    out.append(pred)

        with timer.env('cat'):
            pred_outs = [torch.cat(x, -2) for x in pred_outs]

        if self.training:
            return pred_outs
        else:
            pred_outs[1] = F.softmax(pred_outs[1], -1) # Softmax the conf output
            return self.detect(*pred_outs)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        self.load_state_dict(torch.load(path))



if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    net = Yolact(backbone_path='weights/resnet101_reducedfc.pth')
    net.train()

    # GPU
    # net = net.cuda()
    # cudnn.benchmark = True
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    x = torch.zeros((1, 3, 600, 600))

    y = net(x)

    for a in y:
        print(a.size())
    exit()
    
    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J')
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
