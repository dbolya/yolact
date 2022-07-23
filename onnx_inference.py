from data import cfg, MEANS, STD, set_cfg, mask_type
from layers.box_utils import mask_iou
import torch.nn.functional as F
import torch
import cv2
import os
from layers.output_utils import postprocess

import colorsys
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import onnxruntime as ort
from layers import Detect
from yolact import FastMaskIoUNet
from mmcv.ops import get_onnxruntime_op_path

############################################################
#  Visualization
############################################################


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    # cv2.imwrite('maskapplied.png',image)
    return image


def display_instances(
    image,
    boxes,
    masks,
    class_ids,
    class_names,
    scores=None,
    title="",
    figsize=(8, 8),
    show_mask=True,
    show_bbox=True,
    colors=None,
    captions=None,
    plot_path=None,
):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    # print(image.shape, masks.shape)
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    fig, ax = plt.subplots(
        1,
        figsize=figsize,
    )

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    # print(N, '-------')
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):

            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        # print(show_mask,'---------------')
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color, alpha=0.2)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if plot_path is not None:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        fig.savefig(plot_path)
    plt.close(fig)


#preprocess
class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()

        self.mean = torch.Tensor(MEANS).float()
        self.std  = torch.Tensor( STD ).float()

        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std  = self.std.cuda()
        self.mean = self.mean[None, :, None, None]
        self.std = self.std[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std  = self.std.to(img.device)
        
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        # if cfg.preserve_aspect_ratio:
        #     raise NotImplementedError

        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, (cfg.max_size, cfg.max_size), mode='bilinear', align_corners=False)

        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255
        
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        
        img = img[:, (2, 1, 0), :, :].contiguous()

        # Return value is in channel order [n, c, h, w] and RGB
        return img

def load_image(path: str):
    img =cv2.imread(path)
    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,_ = img.shape
    frame = torch.from_numpy(img).float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    return batch.numpy(), (h,w)

def PostProcess(preds, orig_shapes, maskiou_sess, score_threshold: float):
    h, w = orig_shapes
    classes, scores, boxes, masks = postprocess(preds, w, h, maskiou_net=maskiou_sess, crop_masks=True, score_threshold=score_threshold)
    # if classes.size(0) == 0:
    #     return

    classes = list(classes.cpu().numpy().astype(int))
    if isinstance(scores, list):
        box_scores = list(scores[0].cpu().detach().numpy().astype(float))
        mask_scores = list(scores[1].cpu().detach().numpy().astype(float))
    else:
        scores = list(scores.detach().cpu().detach().numpy().astype(float))
        box_scores = scores
        mask_scores = scores
    masks = masks.view(-1, h*w)
    boxes = boxes.cpu().numpy()
    masks = masks.view(-1, h, w).detach().cpu().numpy()

    structure_bbox_list =[]
    mask_list=[]
    for i in range(masks.shape[0]):
        # Make sure that the bounding box actually makes sense and a mask was produced
        if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:

            if mask_scores[i]>0.4:
                bbox = boxes[i,:]
                bbox = [round(float(x)*10)/10 for x in bbox]
                structure_bbox_list.append(bbox)

                mask_list.append(masks[i,:,:].astype(np.uint8))

    return structure_bbox_list, mask_list, classes, mask_scores

def main(args):
    #load model
    config = args.config
    set_cfg(config)
    cfg.mask_proto_debug = False

    detect_layer = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
                conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)
    if cfg.use_maskiou:
        maskiou_net_sess = ort.InferenceSession(args.onnx_paths[1], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        maskiou_net_sess_input_names = [inp.name for inp in maskiou_net_sess.get_inputs()]
        print('Input Names:', maskiou_net_sess_input_names)
        maskiou_net_sess_output_names = [out.name for out in maskiou_net_sess.get_outputs()]
        print(maskiou_net_sess_output_names)

    img_path = args.img_path
    img, orig_shapes =load_image(img_path)

    ## exported ONNX model with custom operators
    ort_custom_op_path = get_onnxruntime_op_path()
    assert os.path.exists(ort_custom_op_path)
    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(ort_custom_op_path)

    session = ort.InferenceSession(args.onnx_paths[0], session_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    print('Input Names:', input_names)
    output_names = [out.name for out in session.get_outputs()]
    print(output_names)
    onnx_results = session.run(None, {input_names[0] : img})
    detect_input = {"loc": torch.from_numpy(onnx_results[0]), 
                    "conf": torch.from_numpy(onnx_results[1]),
                    "mask": torch.from_numpy(onnx_results[2]),
                    "priors": torch.from_numpy(onnx_results[3]),
                    "proto": torch.from_numpy(onnx_results[4]),
                    }

    detect_layer.use_fast_nms = True
    detect_layer.use_cross_class_nms = False

    final_out = detect_layer(detect_input)

    structure_bbox_list, mask_list, classes, scores = PostProcess(final_out, orig_shapes, maskiou_net_sess, args.score_threshold)

    display_instances(
                cv2.imread(img_path),
                np.array([[y1, x1, y2, x2] for x1, y1, x2, y2 in structure_bbox_list]),
                np.stack(mask_list, axis=2),
                np.arange(len(structure_bbox_list)),
                [str(i) for i in range(len(structure_bbox_list))],
                show_mask=True,
                show_bbox=True,
                plot_path=img_path.split(".jpg")[0]+"_ort_out.png",
                figsize=(16, 16),
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Yolact onnx inference')
    parser.add_argument('--config', type=str, default='yolact_plus_base_config',
                    help='The config object to use.')
    parser.add_argument('--img_path', type=str, default="",
                    help='Give the path to image for inference')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                    help='Give the path to exported ONNX weights')
    parser.add_argument('--onnx_paths', type=str, default="",  nargs='+',
                    help='Give the path to exported ONNX weights')
    
    args = parser.parse_args()
    main(args)
    