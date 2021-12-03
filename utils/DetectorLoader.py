from yolact import Yolact
import torch
from data import cfg, set_cfg
import torch.backends.cudnn as cudnn

__author__ = 'Ajay Bhargava'
__version__ = '0.1'

class YOLACT(object):
    '''
    Class for handling YOLACT model.
    def __init__():
        
        Arguments
        --------------------------------
        weights_file (str): location of the weights file
        config (str): name of the base-network upon which the model is built
        device (str): device to use the model, default is cuda, can be specified as CPU

        Returns
        --------------------------------
        Loaded model on device. 
    '''
    def __init__(self, 
                weights_file = '../weights/yolact_resnet50_54_800000.pth',
                config = 'yolact_resnet50_config',
                device = 'cuda', 
                fast_nms = True,
                detect = False,
                cross_class_nms = False, 
                mask_proto_debug = False):

        # Declare Configs
        set_cfg(config)

        # Set Detection
        if detect:
            cfg.eval_mask_branch = False
        
        # Declare CUDA 
        if device == 'cuda':
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Initialize Model
        self.model = Yolact()

        # Load model on CUDA 
        if device == 'cuda':
            self.model = self.model.cuda()

        self.model.load_weights(weights_file)
        self.model.eval()
        
        # Other configurations specific for YOLACT 
        if fast_nms:
            self.model.detect.use_fast_nms = fast_nms

        if cross_class_nms:
            self.model.detect.use_cross_class_nms = cross_class_nms

        if mask_proto_debug:
            self.model.detect.mask_proto_debug = mask_proto_debug
    
    @classmethod
    def detect(args):
        '''
        Class method for running a model detection
        def detect():

            Arguments
            --------------------------------
            
        '''
        