import matplotlib as mpl
import matplotlib.cm as cm
from torchvision import transforms, datasets

from yolact import Yolact
from yolact.layers.output_utils import postprocess
from yolact.utils.augmentations import FastBaseTransform
from yolact.utils.model import download_model_if_doesnt_exist
from yolact.data import cfg, set_cfg, set_dataset


import torch
import os
import PIL.Image as pil
import numpy as np


fast_nms = True
cross_class_nms = False
config = None

def merge_masks(masks):
    n_masks = np.concatenate([masks, masks, masks], -1)
    merged = np.zeros((n_masks.shape[1:]))
    colors = np.random.randint(0, 255, (n_masks.shape[0], 3))
    for m, c in zip(n_masks, colors):
        merged += c * m
        
    return merged.astype(int)

def infer_segmentation(model_name, img, top_k=15, score_threshold=0.15, crop=True):
    """
    Infer segmentation on an image
    Args:
        model_name: One of:
            "yolact_resnet50_54_800000.pth",
        img: cv2 image
    Returns: segmentation_array, disparity_image
    """
    config_names_paths = {
        "yolact_resnet50_54_800000.pth": "yolact_resnet50_config",
        "yolact_plus_resnet50_54_800000.pth": "yolact_plus_resnet50_config"
    }
    if model_name not in config_names_paths:
        raise Exception("Invalid Model Name")

    set_cfg(config_names_paths[model_name])
    os.makedirs('models', exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)

    with torch.no_grad():
        # LOADING PRETRAINED MODEL
        print("   Loading pretrained model")
        net = Yolact()
        net.load_weights(model_path, device)
        net.eval()

        net.to(device)

        net.detect.use_fast_nms = fast_nms
        net.detect.use_cross_class_nms = cross_class_nms
        cfg.mask_proto_debug = False

        # Load image and preprocess
        frame = torch.from_numpy(img).to(device).float()
        h, w, _ = frame.shape

        batch = FastBaseTransform(device)(frame.unsqueeze(0))

        # PREDICTION
        preds = net(batch)

        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(preds, w, h, visualize_lincomb = False,
                                        crop_masks        = crop,
                                        score_threshold   = score_threshold)
        cfg.rescore_bbox = save

        idx = t[1].argsort(0, descending=True)[:top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        
        # get only people
        person_idx = np.where(classes == 0)
        classes = classes[person_idx]
        scores = scores[person_idx]
        boxes = boxes[person_idx]
        masks = masks[person_idx]
        
        num_dets_to_consider = min(top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < score_threshold:
                num_dets_to_consider = j
                break

        masks = masks[:num_dets_to_consider, :, :, None]

    return masks.detach().cpu().numpy(), merge_masks(masks.detach().cpu().numpy())
