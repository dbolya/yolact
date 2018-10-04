import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F

COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

def mask_iou(mask1, mask2):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    return intersection / union

def paint_mask(img_numpy, mask, color):
	h, w, _ = img_numpy.shape
	img_numpy = img_numpy.copy()

	mask = np.tile(mask.reshape(h, w, 1), (1, 1, 3))
	color_np = np.array(color[:3]).reshape(1, 1, 3)
	color_np = np.tile(color_np, (h, w, 1))
	mask_color = mask * color_np

	mask_alpha = 0.3

	# Blend image and mask
	image_crop = img_numpy * mask
	img_numpy *= (1-mask)
	img_numpy += image_crop * (1-mask_alpha) + mask_color * mask_alpha

	return img_numpy

# Inverse sigmoid
def logit(x):
	return np.log(x / (1-x + 0.0001) + 0.0001)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

img_fmt = '../data/coco/images/%012d.jpg'
with open('info.txt', 'r') as f:
	img_id = int(f.read())

img = plt.imread(img_fmt % img_id).astype(np.float32)
h, w, _ = img.shape

gt_masks    = np.load('gt.npy').astype(np.float32).transpose(1, 2, 0)
proto_masks = np.load('proto.npy').astype(np.float32)

proto_masks = torch.Tensor(proto_masks).permute(2, 0, 1).contiguous().unsqueeze(0)
proto_masks = F.interpolate(proto_masks, (h, w), mode='bilinear', align_corners=False).squeeze(0)
proto_masks = proto_masks.permute(1, 2, 0).numpy()

# # A x = b
ls_A = proto_masks.reshape(-1, proto_masks.shape[-1])
ls_b = gt_masks.reshape(-1, gt_masks.shape[-1])

# x is size [256, num_gt]
x = np.linalg.lstsq(ls_A, ls_b, rcond=None)[0]

approximated_masks = (np.matmul(proto_masks, x) > 0.5).astype(np.float32)

num_gt = approximated_masks.shape[2]
ious = mask_iou(torch.Tensor(approximated_masks.reshape(-1, num_gt).T),
				torch.Tensor(gt_masks.reshape(-1, num_gt).T))

ious = [int(ious[i, i].item() * 100) for i in range(num_gt)]
ious.sort(key=lambda x: -x)

print(ious)

gt_img = img.copy()

for i in range(num_gt):
	gt_img = paint_mask(gt_img, gt_masks[:, :, i], COLORS[i % len(COLORS)])
	
plt.imshow(gt_img / 255)
plt.title('GT')
plt.show()

for i in range(num_gt):
	img = paint_mask(img, approximated_masks[:, :, i], COLORS[i % len(COLORS)])

plt.imshow(img / 255)
plt.title('Approximated')
plt.show()
