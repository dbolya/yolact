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


# Datasets
coco_dataset = Config({
    'name': 'COCO',
    'split': 'train2014'
})

# Backbones
from backbone import ResNetBackbone, VGGBackbone
from torchvision.models.vgg import cfg as vggcfg

resnet101_backbone = Config({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),

    'selected_layers': list(range(2, 7)),
    'pred_scales': [[.5, 1], [1], [1], [1], [1, 2]],
    'pred_aspect_ratios': [[1.05, 0.62],
                           [1.29, 0.79, 0.47, 2.33, 0.27],
                           [1.19, 0.72, 0.43, 2.13, 0.25],
                           [1.34, 0.84, 0.52, 2.38, 0.30],
                           [1.40, 0.95, 0.64, 2.16]],
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
})

vgg16_backbone = Config({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vggcfg['D'],),

    'selected_layers': list(range(3, 9)),
    'pred_scales': [[4]]*6,
    'pred_aspect_ratios': [[1, 1.414, 0.707, 0.577, 1.73][:n] for n in [4, 6, 6, 6, 4, 4]],
})

# Configs
coco_base_config = Config({
    'dataset': coco_dataset,
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'mask_size': 16,
    'min_size': 200,
    'max_size': 300,
    'use_gt_bboxes': False,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    
    'backbone': None,
    'name': 'base_config',
})

yolact_resnet101_config = coco_base_config.copy({
    'name': 'yolact_vgg16',
    'backbone': vgg16_backbone,
})

def get_cfg():
    """ Returns the currently loaded config object. """
    return yolact_resnet101_config
