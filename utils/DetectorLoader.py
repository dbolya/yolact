from yolact import Yolact
import torch
import torch.backends.cudnn as cudnn
from data import cfg, set_cfg
from utils.augmentations import FastBaseTransform
from layers import output_utils
from utils import VideoReaders

class YOLACT(object):
    '''
    SuperClass for YOLACT model.
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
                weights_file = './weights/yolact_resnet50_54_800000.pth',
                config = 'yolact_resnet50_config',
                device = 'cuda',
                threshold = 0.1,
                fast_nms = True,
                detect = False,
                cross_class_nms = False):
        super(YOLACT, self).__init__()
        # Declare Configs
        set_cfg(config)

        # Set Threshold - move to subclass 
        self.threshold = threshold

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

    @staticmethod
    def prep_prediction(dets_out, img, h, w, thresh):
        '''
        Docstring for prep_display()
        Prepares a predicted image for use elsewhere. 
        '''
        with torch.no_grad():
            h, w, _ = img.shape
            tensor = output_utils.postprocess(dets_out, w, h, visualize_lincomb = False, crop_masks = True, score_threshold = thresh)
            idx = tensor[1].argsort(0, descending=True)
            classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in tensor]
            return classes, scores, boxes, masks
    
    def predict(self, image):
        '''
        Class method for running a model detection
        def detect():

            Arguments
            --------------------------------
            image (np.ndarray): Image to which a model inference is conducted.

            Returns
            --------------------------------
            classes (np.ndarray): Class identities of the predicted instances.
            scores (np.ndarray): Class scores of the predicted instances.
            bboxes (np.ndarray): Bounding boxes of the predicted instances.
            masks (np.ndarray): Pixelwise masks for the predicted instances. 

        '''
        cfg.mask_proto_debug = False
        # Convert Image to 'Frame' Object 
        self.frame = torch.from_numpy(image).cuda().float()
        self.batch = FastBaseTransform()(self.frame.unsqueeze(0))
        self.preds = self.model(self.batch)
        self.classes, self.scores, self.bboxes, self.masks = self.prep_prediction(self.preds, self.frame, None, None, self.threshold)
        #TODO (06/12/21)
        #Add a method to merge np.ndarrays of similar class, relabeling the pixel value to a new integer
        #Update to return merged pixelwise map.   
        return self.classes, self.scores, self.bboxes, self.masks