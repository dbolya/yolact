# config.py
import os.path
import copy

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)


class Config(object):
    """
    Holds the config for various networks.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)


# Datasets
coco_dataset = Config({
    'name': 'COCO',
    'split': 'train2014'
})

# Backbones
from backbone import ResNetBackbone, VGGBackbone
from torchvision.models.vgg import cfg as vggcfg
from math import sqrt

resnet101_backbone = Config({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
    'pred_aspect_ratios': [ [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]] ] * 6,
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
})

vgg16_arch = [[64, 64],
              [ 'M', 128, 128],
              [ 'M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              [ 'M', 512, 512, 512],
              [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]]

vgg16_backbone = Config({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vgg16_arch, [(256, 2), (128, 2), (128, 1), (128, 1)], [3]),

    'selected_layers': [3] + list(range(5, 10)),
    'pred_scales': [[5, 4]]*6,
    'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
})

# Configs
coco_base_config = Config({
    'dataset': coco_dataset,
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'mask_size': 20,
    
    # Input image size. If preserve_aspect_ratio is False, min_size is ignored.
    'min_size': 200,
    'max_size': 300,
    
    # Whether or not to do post processing on the cpu at test time (usually faster)
    'force_cpu_detect': True,
    'force_cpu_nms': True,

    # Whether or not to tie the mask loss to 0
    'train_masks': False,
    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, uses the faster r-cnn resizing scheme.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # Whether or not to use the prediction module (c) from DSSD
    'use_prediction_module': False,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,
    
    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,

    'backbone': None,
    'name': 'base_config',
})

yolact_resnet101_config = coco_base_config.copy({
    'name': 'yolact_resnet101',
    'backbone': resnet101_backbone,

    'max_size': 550,

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': True,
    'use_yolo_regressors': True,
})

# Pretty close to the original ssd300 just using resnet101 instead of vgg16
ssd600_resnet101_config = coco_base_config.copy({
    'name': 'ssd600_resnet101',
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(2, 8)),
        'pred_scales': [[5, 4]]*6,
        'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
    }),

    'max_size': 550,

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,
})

# Close to vanilla ssd300
ssd300_config = coco_base_config.copy({
    'name': 'ssd300',
    'backbone': vgg16_backbone.copy({
        'selected_layers': [3] + list(range(5, 10)),
        'pred_scales': [[5, 4]]*6,
        'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
    }),

    'max_size': 300,

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,
})

cfg = ssd600_resnet101_config.copy()

def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    
