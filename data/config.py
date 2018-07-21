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
    To get the currently active config, call get_cfg()
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
resnet101_backbone = Config({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth'
})

vgg16_backbone = Config({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth'
})

# Configs
coco_base_config = Config({
    'dataset': coco_dataset,
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'mask_size': 16,
    'use_gt_bboxes': False,
    
    'backbone': None,
    'name': 'base_config',
    'min_size': 0,
    'max_size': 0,
})

yolact_resnet101_config = coco_base_config.copy({
    'name': 'yolact_resnet101',
    'min_size': 400,
    'max_size': 600,
    'backbone': resnet101_backbone,
})

def get_cfg():
    """ Returns the currently loaded config object. """
    return yolact_resnet101_config
