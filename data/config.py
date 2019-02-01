# config.py
import os.path
import copy

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

# These are in BRG and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

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
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


# Datasets
coco2014_dataset = Config({
    'name': 'COCO 2014',
    'train': 'train2014',
    'valid': 'val2014'
})

coco2017_dataset = Config({
    'name': 'COCO 2017',
    'train': 'train2017', # Trainval35k
    'valid': 'val2017'    # Minibatch5k
})

# Backbones
from backbone import ResNetBackbone, VGGBackbone, ResNetBackboneGN, DarkNetBackbone
from torchvision.models.vgg import cfg as vggcfg
from math import sqrt
import torch


resnet_transform = Config({
    'channel_order': 'RGB',
    'normalize': True,
    'subtract_means': False,
    'to_float': False,
})

vgg_transform = Config({
    # Note that though vgg is traditionally BRG,
    # the channel order of vgg_reducedfc.pth is RGB.
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': True,
    'to_float': False,
})

darknet_transform = Config({
    'channel_order': 'RGB',
    'normalize': False,
    'subtract_means': False,
    'to_float': True,
})

resnet101_backbone = Config({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
    'pred_aspect_ratios': [ [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]] ] * 6,
    'use_pixel_scales': False,
})

resnet101_gn_backbone = Config({
    'name': 'ResNet101_GN',
    'path': 'R-101-GN.pkl',
    'type': ResNetBackboneGN,
    'args': ([3, 4, 23, 3],),
    'transform': resnet_transform,

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
    'pred_aspect_ratios': [ [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]] ] * 6,
    'use_pixel_scales': False,
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
    'transform': resnet_transform,
})

darknet53_backbone = Config({
    'name': 'DarkNet53',
    'path': 'darknet53.pth',
    'type': DarkNetBackbone,
    'args': ([1, 2, 8, 8, 4],),
    'transform': darknet_transform,

    'selected_layers': list(range(3, 9)),
    'pred_scales': [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [1.8, 1.92]],
    'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
    'use_pixel_scales': False,
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
    'transform': vgg_transform,

    'selected_layers': [3] + list(range(5, 10)),
    'pred_scales': [[5, 4]]*6,
    'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
    'use_pixel_scales': False,
})

mask_type = Config({
    # Direct produces masks directly as the output of each pred module.
    # Parameters: mask_size, use_gt_bboxes
    'direct': 0,

    # Lincomb produces coefficients as the output of each pred module then uses those coefficients
    # to linearly combine features from an earlier convout to create image-sized masks.
    # Parameters:
    #   - masks_to_train (int): Since we're producing (near) full image masks, it'd take too much
    #                           vram to backprop on every single mask. Thus we select only a subset.
    #   - mask_proto_src (int): The input layer to the mask prototype generation network. This is an
    #                           index in backbone.layers. Use to use the image itself instead.
    #   - mask_proto_net (list<tuple>): A list of layers in the mask proto network with the last one
    #                                   being where the masks are taken from. Each conv layer is in
    #                                   the form (num_features, kernel_size, **kwdargs). An empty
    #                                   list means to use the source for prototype masks. If the
    #                                   kernel_size is negative, this creates a deconv layer instead.
    #                                   If the kernel_size is negative and the num_features is None,
    #                                   this creates a simple bilinear interpolation layer instead.
    #   - mask_proto_bias (bool): Whether to include an extra coefficient that corresponds to a proto
    #                             mask of all ones.
    #   - mask_proto_prototype_activation (func): The activation to apply to each prototype mask.
    #   - mask_proto_mask_activation (func): After summing the prototype masks with the predicted
    #                                        coeffs, what activation to apply to the final mask.
    #   - mask_proto_coeff_activation (func): The activation to apply to the mask coefficients.
    #   - mask_proto_crop (bool): If True, crop the mask with the predicted bbox during training.
    #   - mask_proto_crop_expand (float): If cropping, the percent to expand the cropping bbox by
    #                                     in each direction. This is to make the model less reliant
    #                                     on perfect bbox predictions.
    #   - mask_proto_loss (str [l1|disj]): If not None, apply an l1 or disjunctive regularization
    #                                      loss directly to the prototype masks.
    #   - mask_proto_binarize_downsampled_gt (bool): Binarize GT after dowsnampling during training?
    #   - mask_proto_normalize_mask_loss_by_sqrt_area (bool): Whether to normalize mask loss by sqrt(sum(gt))
    #   - mask_proto_reweight_mask_loss (bool): Reweight mask loss such that background is divided by
    #                                           #background and foreground is divided by #foreground.
    #   - mask_proto_grid_file (str): The path to the grid file to use with the next option.
    #                                 This should be a numpy.dump file with shape [numgrids, h, w]
    #                                 where h and w are w.r.t. the mask_proto_src convout.
    #   - mask_proto_use_grid (bool): Whether to add extra grid features to the proto_net input.
    #   - mask_proto_coeff_gate (bool): Add an extra set of sigmoided coefficients that is multiplied
    #                                   into the predicted coefficients in order to "gate" them.
    #   - mask_proto_prototypes_as_features (bool): For each prediction module, downsample the prototypes
    #                                 to the convout size of that module and supply the prototypes as input
    #                                 in addition to the already supplied backbone features.
    #   - mask_proto_prototypes_as_features_no_grad (bool): If the above is set, don't backprop gradients to
    #                                 to the prototypes from the network head.
    #   - mask_proto_remove_empty_masks (bool): Remove masks that are downsampled to 0 during loss calculations.
    #   - mask_proto_reweight_coeff (float): The coefficient to multiple the forground pixels with if reweighting.
    'lincomb': 1,
})

# Self explanitory. For use with mask_proto_*_activation
activation_func = Config({
    'tanh':    torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu':    lambda x: torch.nn.functional.relu(x, inplace=True),
    'none':    lambda x: x,
})

# Configs
coco_base_config = Config({
    'dataset': coco2014_dataset,
    'num_classes': 81,

    'max_iter': 400000,

    # The maximum number of detections for evaluation
    'max_num_detections': 100,

    # dw' = momentum * dw - lr * (grad + decay * w)
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,

    # For each lr step, what to multiply the lr with
    'gamma': 0.1,
    'lr_steps': (280000, 360000, 400000),

    # See mask_type for details.
    'mask_type': mask_type.direct,
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],
    'mask_proto_bias': False,
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
    'mask_proto_coeff_activation': activation_func.tanh,
    'mask_proto_crop': True,
    'mask_proto_crop_expand': 0,
    'mask_proto_loss': None,
    'mask_proto_binarize_downsampled_gt': True,
    'mask_proto_normalize_mask_loss_by_sqrt_area': False,
    'mask_proto_reweight_mask_loss': False,
    'mask_proto_grid_file': 'data/grid.npy',
    'mask_proto_use_grid':  False,
    'mask_proto_coeff_gate': False,
    'mask_proto_prototypes_as_features': False,
    'mask_proto_prototypes_as_features_no_grad': False,
    'mask_proto_remove_empty_masks': False,
    'mask_proto_reweight_coeff': 1,
    'mask_proto_coeff_diversity_loss': False,
    'mask_proto_normalize_emulate_roi_pooling': False,

    # If using batchnorm anywhere inthe backbone, freeze the batchnorm layer during training.
    # Note: any additional batch norm layers after the backbone will not be froze.
    'freeze_bn': False,

    # Set this to a config object if you want an FPN. The parameters for that object are in yolact.py under the class FPN.
    'fpn': None,

    # Use the same weights for each network head
    'share_prediction_module': False,

    # For hard negative mining, instead of using the negatives that are leastl confidently background,
    # use negatives that are most confidently not background.
    'ohem_use_most_confident': False,

    # Uses the same network format as mask_proto_net, except this time it's for adding extra head layers before the final
    # prediction in prediction modules. If this is none, no extra layers will be added.
    'extra_head_net': None,

    # What params should the final head layers have (the ones that predict box, confidence, and mask coeffs)
    'head_layer_params': {'kernel_size': 3, 'padding': 1},

    # Add extra layers between the backbone and the network heads
    # The order is (bbox, conf, mask)
    'extra_layers': (0, 0, 0),

    # During training, to match detections with gt, first compute the maximum gt IoU for each prior.
    # Then, any of those priors whose maximum overlap is over the positive threshold, mark as positive.
    # For any priors whose maximum is less than the negative iou threshold, mark them as negative.
    # The rest are neutral and not used in calculating the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.5,

    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 1,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Input image size. If preserve_aspect_ratio is False, min_size is ignored.
    'min_size': 200,
    'max_size': 300,
    
    # Whether or not to do post processing on the cpu at test time
    'force_cpu_nms': True,

    # Whether to use mask coefficient cosine similarity nms instead of bbox iou nms
    'use_coeff_nms': False,

    # Whether or not to have a separate branch whose sole purpose is to act as the coefficients for coeff_diversity_loss
    # Remember to turn on coeff_diversity_loss, or these extra coefficients won't do anything!
    # To see their effect, also remember to turn on use_coeff_nms.
    'use_instance_coeff': False,
    'num_instance_coeffs': 64,

    # Whether or not to tie the mask loss / box loss to 0
    'train_masks': True,
    'train_boxes': True,
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

    # A list of settings to apply after the specified iteration. Each element of the list should look like
    # (iteration, config_dict) where config_dict is a dictionary you'd pass into a config object's init.
    'delayed_settings': [],

    'backbone': None,
    'name': 'base_config',
})

# Pretty close to the original ssd300 just using resnet101 instead of vgg16
ssd550_resnet101_config = coco_base_config.copy({
    'name': 'ssd550_resnet101',
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(2, 8)),
        'pred_scales': [[5, 4]]*6,
        'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
    }),

    'max_size': 550,
    'mask_size': 20, # Turned out not to help much

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,
})

ssd550_resnet101_yolo_matching_config = ssd550_resnet101_config.copy({
    'name': 'ssd550_resnet101_yolo_matching',

    'mask_size': 16,

    'use_yolo_regressors': True,
    'use_prediction_matching': True,

    # Because of prediction matching, the number of positives goes up to high and thus
    # we run out of memory when training masks. The amount of memory for training masks
    # is proportional to the number of positives after all.
    'train_masks': False,
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
    'mask_size': 20, # Turned out not to help much

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,
})

# Close to vanilla ssd300 but bigger!
ssd550_config = ssd300_config.copy({
    'name': 'ssd550',
    'backbone': ssd300_config.backbone.copy({
        'args': (vgg16_arch, [(256, 2), (256, 2), (128, 2), (128, 1), (128, 1)], [4]),
        'selected_layers': [4] + list(range(6, 11)),
    }),

    'max_size': 550,
    'mask_size': 16,
})

yolact_resnet101_config = ssd550_resnet101_config.copy({
    'name': 'yolact_resnet101',

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': True,
    'use_prediction_matching': False,

    'mask_type': mask_type.lincomb,
    'masks_to_train': 100,
    'mask_proto_src': 0,
    'mask_proto_net': [],
})

yolact_resnet101_dedicated_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_dedicated',
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {'stride': 2})],
})

yolact_resnet101_deep_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_deep',
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'stride': 2}), (256, 3, {'stride': 2})] + [(256, 3, {})] * 3,
})

yolact_resnet101_shallow_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_shallow',
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'stride': 2}), (256, 3, {'stride': 2})],
})

yolact_resnet101_conv4_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_conv4',
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 5,
})

yolact_resnet101_deconv4_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_deconv4',
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(256, -2, {'stride': 2})] * 2 + [(256, 3, {'padding': 1})],
})

yolact_resnet101_maskrcnn_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_maskrcnn',
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(256, -2, {'stride': 2}), (256, 1, {})],
})

# Start of Ablations
yolact_resnet101_maskrcnn_1_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_1',
    'use_yolo_regressors': False,
})
yolact_resnet101_maskrcnn_2_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_2',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
})
yolact_resnet101_maskrcnn_3_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_3',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'use_prediction_module': True,
})
yolact_resnet101_maskrcnn_4_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_4',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'use_prediction_module': True,
    'mask_proto_bias': True,
})
yolact_resnet101_maskrcnn_5_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_5',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'use_prediction_module': True,
    'mask_proto_bias': True,
    'mask_proto_mask_activation': activation_func.none,
})
yolact_resnet101_maskrcnn_6_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_6',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'use_prediction_module': True,
    'mask_proto_bias': True,
    'mask_proto_mask_activation': activation_func.relu,
})

# Same config just with a different name so we can test bug fixes
yrm1_config = yolact_resnet101_maskrcnn_1_config.copy({
    'name': 'yrm1',
    'max_iter': 600000,
})

# Ablations 2: Electric Boogaloo
yrm7_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm7',
    'use_yolo_regressors': False,
    'mask_proto_coeff_activation': activation_func.sigmoid,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'mask_proto_mask_activation': activation_func.none,
})
yrm8_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm8',
    'use_yolo_regressors': False,
    'mask_proto_coeff_activation': activation_func.softmax,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'mask_proto_mask_activation': activation_func.none,
})
yrm9_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm9',
    'use_yolo_regressors': False,
    'mask_proto_coeff_activation': activation_func.sigmoid,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'mask_proto_mask_activation': activation_func.none,
    'mask_proto_crop': False,
})
yrm10_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm10',
    'use_yolo_regressors': False,
    'mask_proto_loss': 'l1',
})
yrm11_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm11',
    'use_yolo_regressors': False,
    'mask_proto_loss': 'disj',
})
yrm12_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm12',
    'use_yolo_regressors': False,
    'mask_proto_coeff_activation': activation_func.none,
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
})
yrm13_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm13',
    'use_yolo_regressors': False,
    'mask_proto_crop': False,
})
yrm13_35k_config = yrm13_config.copy({
    'name': 'yrm13_35k',
    'dataset': coco2017_dataset,
})

# This config is to emulate the DSSD SSD513 training parameters for an exact comparison.
yrm13_dssd_35k_config = yrm13_config.copy({
    'name': 'yrm13_dssd_35k',
    'dataset': coco2017_dataset,

    # Make sure the batch size is 20 for this
    'lr_steps': (160000, 220000, 240000),
    'max_iter': 240000,
})

yrm14_config = yolact_resnet101_maskrcnn_1_config.copy({
    'name': 'yrm14',
    'mask_proto_src': 3,
})

yrm15_config = yolact_resnet101_maskrcnn_1_config.copy({
    'name': 'yrm15',
    'negative_iou_threshold': 0.3,
})
yrm16_config = yolact_resnet101_maskrcnn_1_config.copy({
    'name': 'yrm16',
    'mask_proto_normalize_mask_loss_by_sqrt_area': True,
})
yrm16_2_config = yolact_resnet101_maskrcnn_1_config.copy({
    'name': 'yrm16_2',
    'mask_proto_normalize_mask_loss_by_sqrt_area': True,
})
yrm17_config = yrm13_config.copy({
    'name': 'yrm17',
    'mask_proto_use_grid': True,
})


fixed_ssd_config = yrm13_config.copy({
    'name': 'fixed_ssd',

    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(2, 8)),
        
        # Numbers derived from SSD300
        #
        # Usually, you'd encode these scales as pixel width and height. 
        # However, if you then increase your input image size, your anchors will be way too small.
        # To get around that, I encode relative size as scale / convout_size.
        #
        # Wait, hold on a second. That doesn't fix that problem whatsoever.
        # TODO: Encode scales as relative to image size, not convout size.
        #
        # Okay, maybe the reasoning could be relative receptive field size.
        # For instance, a scale of 1 is what one convout pixel directly sees as input from the image.
        # Of course, there are a lot of 3x3 kernels in here so hopefully the receptive field is larger
        # than just 1. But you really don't observe that to be the case, do you? ¯\_(ツ)_/¯
        'pred_scales': [
            [3.5, 4.95], # 30 / 300 * 35, sqrt((30 / 300) * (60 / 300)) * 35
            [3.6, 4.90], #
            [3.3, 4.02], # In general,
            [2.7, 3.10], #   min / 300 * conv_out_size,
            [2.1, 2.37], #   sqrt((min / 300) * (max / 300)) * conv_out_size
            [1.8, 1.92], #
        ],
        'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
    }),

})

fixed_ssd_gn_config = fixed_ssd_config.copy({
    'name': 'fixed_ssd_gn',
    
    'backbone': resnet101_gn_backbone.copy({
        'selected_layers': list(range(2, 8)),
        'pred_scales': [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [1.8, 1.92]],
        'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
    })
})

yrm18_config = yrm13_config.copy({
    'name': 'yrm18',
    'mask_proto_coeff_activation': activation_func.none,
    'backbone': fixed_ssd_config.backbone,
})

yrm19_config = yrm18_config.copy({
    'name': 'yrm19',
    'mask_proto_coeff_gate': True,
})

yrm20_config = fixed_ssd_config.copy({
    'name': 'yrm20',
    'use_prediction_module': True,
})

# This config will not work anymore (it was a bug)
# Any configs based off of it will also not work
yrm21_config = fixed_ssd_config.copy({
    'name': 'yrm21',
    # This option doesn't exist anymore
    'mask_proto_replace_deconv_with_upsample': True,
})

yrm22_config = fixed_ssd_config.copy({
    'name': 'yrm22',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(256, 1, {})],
})

yrm22_gn_config = fixed_ssd_gn_config.copy({
    'name': 'yrm22_gn',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(256, 1, {})],
    'crowd_iou_threshold': 0.7,
    'lr_steps': (280000, 410000, 458000),
    'max_iter': 458000,
})

yrm22_gn_highlr_config = yrm22_gn_config.copy({
    'name': 'yrm22_gn_highlr',
    'lr': 2e-3
})

yrm22_2_config = yrm22_config.copy({
    'name': 'yrm22_2',
    'crowd_iou_threshold': 1,
})

yrm22_crowd_config = yrm22_config.copy({
    'name': 'yrm22_crowd',
    'crowd_iou_threshold': 0.7,
})

# Continue training with crowds to see if anything improves
yrm22_long_config = yrm22_config.copy({
    'name': 'yrm22_long',
    'crowd_iou_threshold': 0.7,
    'lr_steps': (0, 280000, 360000, 400000),
})

yrm22_nopad_config = yrm22_crowd_config.copy({
    'name': 'yrm22_nopad',
    'mask_proto_net': [(256, 3, {'padding': 0})] * 4 + [(None, -2, {}), (256, 3, {'padding': 0})] * 2 + [(256, 1, {})],
})

yrm22_nodecay_config = yrm22_crowd_config.copy({
    'name': 'yrm22_nodecay',
    'decay': 0,
})

yrm22_freezebn_config = yrm22_crowd_config.copy({
    'name': 'yrm22_freezebn',
    'freeze_bn': True,
})

yrm22_fewerproto_config = yrm22_crowd_config.copy({
    'name': 'yrm22_fewerproto',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
})

yrm22_muchfewerproto_config = yrm22_crowd_config.copy({
    'name': 'yrm22_muchfewerproto',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(64, 1, {})],
})

yrm22_newreg_config = yrm22_crowd_config.copy({
    'name': 'yrm22_newreg',
    'gamma': 0.3, # approx sqrt(0.1)
    'lr_steps': (140000, 260000, 310000, 360000, 380000, 400000),
})

yrm22_optimanchor_config = yrm22_newreg_config.copy({
    'name': 'yrm22_optimanchor',
    'backbone': yrm22_newreg_config.backbone.copy({
        'pred_scales': [[1.73, 2.96], [3.12, 2.44, 1.01], [2.09, 2.25, 3.32], [0.90, 2.17, 3.00], [1.03, 2.16], [0.75]],
        'pred_aspect_ratios': [[[0.59, 0.95], [0.62, 1.18]], [[0.49, 0.75], [0.68, 1.26], [0.64, 1.57]], [[1.94, 1.28], [0.56, 0.84], [0.62, 1.13]], [[1.66, 2.63], [0.51, 1.82], [1.28, 0.76]], [[0.45, 2.43], [0.97]], [[1.88]]]
    }),
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})]
})

yrm22_coeffdiv_config = yrm22_newreg_config.copy({
    'name': 'yrm22_coeffdiv',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
    'mask_proto_coeff_diversity_loss': True,

    'use_coeff_nms': True,
})

yrm22_darknet_config = yrm22_newreg_config.copy({
    'name': 'yrm22_darknet',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],

    'backbone': darknet53_backbone,
    'mask_proto_src': 3,
})

yrm16_3_config = yrm21_config.copy({
    'name': 'yrm16_3',
    'mask_proto_normalize_mask_loss_by_sqrt_area': True,
})

yrm23_config = yrm21_config.copy({
    'name': 'yrm23',
    'extra_layers': (0, 0, 1),
})

yrm24_config = yrm21_config.copy({
    'name': 'yrm24',
    'train_boxes': False,
})

yrm32_config = yrm22_newreg_config.copy({
    'name': 'yrm32',
    'freeze_bn': False,
    'decay': 1e-4,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
    'extra_head_net': [(512, 3, {'padding': 1})] + [(256, 3, {'padding': 1})] * 2 + [(512, 3, {'padding': 1}), (1024, 3, {'padding': 1})],
    'head_layer_params': {'kernel_size': 1, 'padding': 0},
})

yrm32_protofeat_config = yrm32_config.copy({
    'name': 'yrm32_protoin',
    'mask_proto_prototypes_as_features': True,
    'mask_proto_prototypes_as_features_no_grad': True,
})

yrm32_massivelad_config = yrm32_config.copy({
    'name': 'yrm32_massivelad',
    'extra_head_net': None,
    'head_layer_params': {'kernel_size': 3, 'padding': 1},
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + 
        [(None, -2, {}), (256, 3, {'padding': 1})] +
        [(None, -2, {}), (128, 3, {'padding': 1})] +
        [(None, -2, {}), ( 64, 3, {'padding': 1})] +
        [(64, 1, {})],
})

yrm32_othermassivelad_config = yrm32_config.copy({
    'name': 'yrm32_othermassivelad',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + 
        [(None, -2, {}), (256, 3, {'padding': 1})] +
        [(None, -2, {}), (128, 3, {'padding': 1})] +
        [(None, -2, {}), ( 64, 3, {'padding': 1})] +
        [(64, 1, {})],
})

yrm32_bnmassivelad_config = yrm32_othermassivelad_config.copy({
    'name': 'yrm32_bnmassivelad',
    'freeze_bn': False,
})

yrm32_absoluteunit_config = yrm32_massivelad_config.copy({
    'name': 'yrm32_absoluteunit',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 2 + 
        [(None, -2, {}), (256, 3, {'padding': 1})] +
        [(None, -2, {}), (128, 3, {'padding': 1})] +
        [(None, -2, {}), ( 64, 3, {'padding': 1})] +
        [(None, -2, {}), ( 64, 3, {'padding': 1})] +
        [(64, 1, {})],
})

# Atrous!
yrm34_config = yrm32_config.copy({
    'name': 'yrm34',
    'backbone': yrm32_config.backbone.copy({
        'args': (yrm32_config.backbone.args[0], [2]),

        'selected_layers': list(range(2, 8)),
        'pred_scales': [[3.76], [3.72], [3.58], [3.14], [2.75], [2.12]],
        'pred_aspect_ratios': [[[0.86, 1.51, 0.55]], [[0.84, 1.45, 0.49]], [[0.88, 1.43, 0.52]], [[0.96, 1.61, 0.60]], [[0.91, 1.32, 0.66]], [[0.74, 1.22, 0.90]]]}),
    'extra_head_net': [(512, 3, {'padding': 1})] + [(512, 3, {'padding': 1}), (1024, 3, {'padding': 1})],
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
    'freeze_bn': True,
})

yrm34b_config = yrm34_config.copy({
    'name': 'yrm34b',
    'backbone': yrm34_config.backbone.copy({
        'scales': [[3.91, 2.31], [3.39, 1.86], [3.20, 2.93], [2.69, 2.62], [2.63, 2.05], [2.13]],
        'aspect_ratios': [[[0.66], [0.82]], [[0.61, 1.20], [1.30]], [[0.62, 1.02], [0.48, 1.60]], [[0.92, 1.66,
0.63], [0.43]], [[1.68, 0.98, 0.63], [0.59, 1.89, 1.36]], [[1.20, 0.86]]]
    })
})

yrm34c_config = yrm22_config.copy({
    'name': 'yrm34c',

    'backbone': yrm22_config.backbone.copy({
        'selected_layers': list(range(1, 7)),

        'pred_scales': [[3.91, 2.31], [3.39, 1.86], [3.20, 2.93], [2.69, 2.62], [2.63, 2.05], [2.13]],
        'pred_aspect_ratios': [[[0.66], [0.82]], [[0.61, 1.20], [1.30]], [[0.62, 1.02], [0.48, 1.60]], [[0.92, 1.66,
0.63], [0.43]], [[1.68, 0.98, 0.63], [0.59, 1.89, 1.36]], [[1.20, 0.86]]]
    }),
    
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
})

yrm22_test_onegpu_config = yrm22_freezebn_config.copy({
    'name': 'yrm22_test_onegpu'
})
yrm22_test_twogpu_config = yrm22_freezebn_config.copy({
    'name': 'yrm22_test_twogpu'
})

yrm35_config = yrm22_config.copy({
    'name': 'yrm35',
    'mask_proto_normalize_emulate_roi_pooling': True,
    'crowd_iou_threshold': 0.7,
})

yrm35_crop_config = yrm35_config.copy({
    'name': 'yrm35_crop',
    'mask_proto_crop': True,
    'lr_steps': (0, 280000, 360000, 500000, 650000),
    'max_iter': 800000,
})

yrm35_expand_config = yrm35_crop_config.copy({
    'name': 'yrm35_expand',
    'mask_proto_crop_expand': 0.1,
    'lr_steps': (140000, 260000, 310000, 360000, 380000, 400000),
})

yrm35_fpn_config = yrm22_config.copy({
    'name': 'yrm35_fpn',
    
    'backbone': fixed_ssd_config.backbone.copy({
        # 0 is conv2
        'selected_layers': list(range(0, 4)),
        
        # These scales and aspect ratios are derived from the FPN paper
        # https://arxiv.org/pdf/1612.03144.pdf
        'pred_scales': [ [4] ] * 5, # 32 / 800 * 136 ...
        'pred_aspect_ratios': [ [[1, 1/sqrt(2), sqrt(2)]] ]*5,
    }),

    # Finally, FPN
    # This replaces each selected layer with the corresponding FPN version
    'fpn': Config({
        'num_features': 256,
        'interpolation_mode': 'bilinear',

        'num_downsample': 1
    }),

    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(128, 1, {})],
    
    'extra_head_net': [(256, 3, {'padding': 1})],
    # 'head_layer_params': {'kernel_size': 1, 'padding': 0},
    
    'share_prediction_module': True,
    'crowd_iou_threshold': 0.7,

    # By their forces combined, they are... RoI Pooling!
    'mask_proto_normalize_emulate_roi_pooling': True,
    'mask_proto_crop': True,
})

yrm25_config = yrm22_config.copy({
    'name': 'yrm25',
    'mask_proto_reweight_mask_loss': True,
})

# Continue training config 25 with or without the reweighting
yrm25_a_config = yrm22_config.copy({
    'name': 'yrm25_a',
    'mask_proto_reweight_mask_loss': True,
    # Start at lr = 1e-4 instead of 1e-3
    'lr_steps': (0, 280000, 360000, 400000),
})

yrm25_b_config = yrm22_config.copy({
    'name': 'yrm25_b',
    'mask_proto_reweight_mask_loss': False,
    # Start at lr = 1e-4 instead of 1e-3
    'lr_steps': (0, 280000, 360000, 500000, 650000),
    'max_iter': 800000,
    # 'use_coeff_nms': True
})

yrm25_half_config = yrm25_config.copy({
    'name': 'yrm25_half',
    'mask_proto_reweight_coeff': 0.5,
})

yrm25_smol_config = yrm25_config.copy({
    'name': 'yrm25_smol',
    'mask_proto_reweight_coeff': 1/32,
})

yrm25_double_config = yrm25_config.copy({
    'name': 'yrm25_double',
    'mask_proto_reweight_coeff': 2,
})

# This is a big boi, tread with caution
yrm26_config = yrm22_config.copy({
    'name': 'yrm26',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -1.8, {}), (256, 3, {'padding': 1})] * 3 + [(256, 1, {})],

    # Because this is such a big boi, we use batch size 6. lr_steps / 6 * 8
    'lr_steps': (373333, 480000, 533333),
    'max_iter': 533333,
})

yrm27_config = yrm22_config.copy({
    'name': 'yrm27',
    'extra_layers': (1, 1, 1)
})

yrm28_config = yrm22_config.copy({
    'name': 'yrm28',
    'mask_proto_prototypes_as_features': True,
})

yrm29_config = yrm22_config.copy({
    'name': 'yrm29',
    'mask_proto_remove_empty_masks': True,
})

yrm28_2_config = yrm28_config.copy({
    'name': 'yrm28_2',
    'mask_proto_prototypes_as_features_no_grad': True, 
})

yrm28_reset_config = yrm28_2_config.copy({
    'name': 'yrm28_reset',
    'crowd_iou_threshold': 0.7,
})

yrm30_config = yrm22_config.copy({
    'name': 'yrm30',
    
    'backbone': fixed_ssd_config.backbone.copy({
        # 0 is conv2
        'selected_layers': list(range(0, 4)),
        
        # These scales and aspect ratios are derived from the FPN paper
        # https://arxiv.org/pdf/1612.03144.pdf
        'pred_scales': [ [5.3] ] * 5, # 32 / 800 * 136 ...
        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
    }),

    # Finally, FPN
    # This replaces each selected layer with the corresponding FPN version
    'fpn': Config({
        'num_features': 256,
        'interpolation_mode': 'bilinear',

        'num_downsample': 1
    }),

    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 6 + [(256, 1, {})],

    'share_prediction_module': True,
    'crowd_iou_threshold': 0.7,
})

yrm30_gn_config = yrm30_config.copy({
    'name': 'yrm30_gn',
    'backbone': fixed_ssd_gn_config.backbone.copy({
        # 0 is conv2
        'selected_layers': list(range(0, 4)),
        
        # These scales and aspect ratios are derived from the FPN paper
        # https://arxiv.org/pdf/1612.03144.pdf
        'pred_scales': [ [5.3] ] * 5, # 32 / 800 * 136 ...
        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
    }),
})

yrm30_lowlr_config = yrm30_config.copy({
    'name': 'yrm30_lowlr',
    'lr_steps': (0, 280000, 360000, 400000),
})

yrm30_halflr_config = yrm30_config.copy({
    'name': 'yrm30_halflr',
    'lr': 5e-4
})

yrm30_bighead_config = yrm30_gn_config.copy({
    'name': 'yrm30_bighead',
    'num_head_features': 512,
})

yrm30_oldsrc_config = yrm30_halflr_config.copy({
    'name': 'yrm30_oldsrc',
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(256, 1, {})],
})

yrm33_config = yrm30_config.copy({
    'name': 'yrm33',
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(128, 1, {})],
    'extra_head_net': [(256, 3, {'padding': 1}), (512, 3, {'padding': 1}), (1024, 3, {'padding': 1})],
    'head_layer_params': {'kernel_size': 1, 'padding': 0},
    'freeze_bn': True,
    'gamma': 0.3, # approx sqrt(0.1)
    'lr_steps': (140000, 260000, 310000, 360000, 380000, 400000),
    'lr': 1e-3,
})

yrm31_config = yrm22_config.copy({
    'name': 'yrm31',
    'ohem_use_most_confident': True
})


yolact_vgg16_config = ssd550_config.copy({
    'name': 'yolact_vgg16',

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': True,
    'use_prediction_matching': False,

    'mask_type': mask_type.lincomb,
    'masks_to_train': 100,
    'mask_proto_layer': 0,
})


yrm300vgg_config = coco_base_config.copy({
    'name': 'yrm300vgg',

    'backbone': vgg16_backbone.copy({
        'selected_layers': [3] + list(range(5, 10)),
        'pred_scales': [[3.5, 4.95], [3.6, 4.90], [3.3, 4.02], [2.7, 3.10], [2.1, 2.37], [1.8, 1.92]],
        'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
    }),

    'max_size': 300,

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,

    'mask_type': mask_type.lincomb,
    'masks_to_train': 100,
    'mask_proto_src': 3,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
    'mask_proto_crop': False,
    
    'crowd_iou_threshold': 0.7,
    
    'gamma': 0.3, # approx sqrt(0.1)
    'lr_steps': (140000, 260000, 310000, 360000, 380000, 400000),
})


cvpr_resnet101_config = coco_base_config.copy({
    'name': 'cvpr_resnet101',
    
    'dataset': coco2017_dataset,

    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(2, 8)),

        'pred_scales': [[1.68, 2.91],
                        [2.95, 2.22, 0.84],
                        [2.23, 2.17, 3.12],
                        [0.76, 1.94, 2.72],
                        [2.10, 2.65],
                        [1.80, 1.92]],

        'pred_aspect_ratios': [[[0.72, 0.96], [0.68, 1.17]],
                               [[1.28, 0.66], [0.63, 1.23], [0.89, 1.40]],
                               [[2.05, 1.24], [0.57, 0.83], [0.61, 1.15]],
                               [[1.00, 2.21], [0.47, 1.60], [1.44, 0.79]],
                               [[1.00, 1.41, 0.71, 1.73, 0.58], [1.08]],
                               [[1.00, 1.41, 0.71, 1.73, 0.58], [1.00]]]
    }),

    'max_size': 550,

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,

    'mask_type': mask_type.lincomb,
    'masks_to_train': 100,
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
    'mask_proto_crop': False,
    'mask_proto_coeff_diversity_loss': True,

    'use_coeff_nms': False,

    'use_instance_coeff': True,
    'num_instance_coeffs': 64,

    'crowd_iou_threshold': 0.7,
    
    'decay': 1e-4,
    'gamma': 0.3, # approx sqrt(0.1)
    'max_iter': 800000,
    'lr_steps': (100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000),
})

cvpr300_resnet101_config = cvpr_resnet101_config.copy({
    'name': 'cvpr300_resnet101',

    'backbone': cvpr_resnet101_config.backbone.copy({ 'selected_layers': list(range(1, 7)) }),

    'max_size': 300,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 2 + [(None, -2, {}), (256, 3, {'padding': 1})] * 3 + [(128, 1, {})],
})


cvpr_resnet50_config = cvpr_resnet101_config.copy({
    'name': 'cvpr_resnet50',

    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(2, 8)),

        'pred_scales': [[1.68, 2.91],
                        [2.95, 2.22, 0.84],
                        [2.23, 2.17, 3.12],
                        [0.76, 1.94, 2.72],
                        [2.10, 2.65],
                        [1.80, 1.92]],

        'pred_aspect_ratios': [[[0.72, 0.96], [0.68, 1.17]],
                               [[1.28, 0.66], [0.63, 1.23], [0.89, 1.40]],
                               [[2.05, 1.24], [0.57, 0.83], [0.61, 1.15]],
                               [[1.00, 2.21], [0.47, 1.60], [1.44, 0.79]],
                               [[1.00, 1.41, 0.71, 1.73, 0.58], [1.08]],
                               [[1.00, 1.41, 0.71, 1.73, 0.58], [1.00]]]
    }),
})


cvpr_darknet53_config = coco_base_config.copy({
    'name': 'cvpr_darknet53',
    
    'dataset': coco2017_dataset,

    'backbone': darknet53_backbone.copy({
        'selected_layers': list(range(3, 9)),

        'pred_scales': [[1.68, 2.91],
                        [2.95, 2.22, 0.84],
                        [2.23, 2.17, 3.12],
                        [0.76, 1.94, 2.72],
                        [2.10, 2.65],
                        [1.80, 1.92]],

        'pred_aspect_ratios': [[[0.72, 0.96], [0.68, 1.17]],
                               [[1.28, 0.66], [0.63, 1.23], [0.89, 1.40]],
                               [[2.05, 1.24], [0.57, 0.83], [0.61, 1.15]],
                               [[1.00, 2.21], [0.47, 1.60], [1.44, 0.79]],
                               [[1.00, 1.41, 0.71, 1.73, 0.58], [1.08]],
                               [[1.00, 1.41, 0.71, 1.73, 0.58], [1.00]]]
    }),

    'max_size': 550,

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,

    'mask_type': mask_type.lincomb,
    'masks_to_train': 100,
    'mask_proto_src': 3,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
    'mask_proto_crop': False,
    'mask_proto_coeff_diversity_loss': True,

    'use_coeff_nms': False,

    'use_instance_coeff': True,
    'num_instance_coeffs': 64,

    'crowd_iou_threshold': 0.7,
    
    'decay': 1e-4,
    'gamma': 0.3, # approx sqrt(0.1)
    'max_iter': 800000,
    'lr_steps': (100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000),
})


cvpr300_darknet53_config = cvpr_darknet53_config.copy({
    'name': 'cvpr300_darknet53',

    'backbone': cvpr_darknet53_config.backbone.copy({ 'selected_layers': list(range(2, 8)) }),

    'max_size': 300,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 2 + [(None, -2, {}), (256, 3, {'padding': 1})] * 3 + [(128, 1, {})],
})




cvpr2_resnet101_config = coco_base_config.copy({
    'name': 'cvpr2_resnet101',
    
    'dataset': coco2017_dataset,

    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(2, 8)),
        'pred_scales': [[2.62, 3.71], [2.70, 3.68], [2.47, 3.01], [2.03, 2.33], [1.58, 1.78], [1.8, 1.92]],
        'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
    }),

    'max_size': 600,

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,

    'mask_type': mask_type.lincomb,
    'masks_to_train': 100,
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],
    'mask_proto_crop': False,
    'mask_proto_coeff_diversity_loss': True,

    'use_coeff_nms': False,

    'use_instance_coeff': True,
    'num_instance_coeffs': 64,

    'crowd_iou_threshold': 0.7,
    
    'decay': 1e-4,
    'gamma': 0.3, # approx sqrt(0.1)
    'max_iter': 800000,
    'lr_steps': (120000, 240000, 360000, 480000, 600000, 700000, 800000),
})

cvpr2_300_resnet101_config = cvpr2_resnet101_config.copy({
    'name': 'cvpr2_300_resnet101',
    'backbone': cvpr2_resnet101_config.backbone.copy({ 'selected_layers': list(range(1, 7)), }),
    'max_size': 300,
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] * 3 + [(128, 1, {})],
})

cvpr2_300_resnet50_config = cvpr2_300_resnet101_config.copy({
    'name': 'cvpr2_300_resnet50',
    'backbone': resnet50_backbone.copy({
        'selected_layers': list(range(1, 7)),
        'pred_scales': [[2.62, 3.71], [2.70, 3.68], [2.47, 3.01], [2.03, 2.33], [1.58, 1.78], [1.8, 1.92]],
        'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
    }),
})

cvpr2_darknet53_config = cvpr2_resnet101_config.copy({
    'name': 'cvpr2_darknet53',
    'backbone': darknet53_backbone.copy({
        'selected_layers': list(range(3, 9)),
        'pred_scales': [[2.62, 3.71], [2.70, 3.68], [2.47, 3.01], [2.03, 2.33], [1.58, 1.78], [1.8, 1.92]],
        'pred_aspect_ratios': [ [[1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n], [1]] for n in [3, 5, 5, 5, 3, 3] ],
    }),
    'mask_proto_src': 3,
})

cvpr2_300_darknet53_config = cvpr2_darknet53_config.copy({
    'name': 'cvpr2_300_darknet53',
    'backbone': cvpr2_darknet53_config.backbone.copy({ 'selected_layers': list(range(2, 8)) }),
    'max_size': 300,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] * 3 + [(128, 1, {})],
})


yrm22_fcis_config = yrm22_config.copy({
    'name': 'yrm22_fcis',

    'backbone': yrm22_config.backbone.copy({
        # Second argument here is a trous on conv 5
        'args': (yrm22_config.backbone.args[0], [3]),

        'selected_layers': [3],
        'pred_scales': [[4, 8, 16, 32]],
        'pred_aspect_ratios': [[[sqrt(0.5), 1, sqrt(2)]] * 4],
    }),
    
    'crowd_iou_threshold': 0.7,
    'max_size': 600,

    'mask_proto_src': 3,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(None, -2, {}), (256, 3, {'padding': 1})] * 2 + [(128, 1, {})],

    'extra_head_net': [(1024, 1, {})],
})


# Default config
cfg = yrm22_fcis_config.copy()

def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    
