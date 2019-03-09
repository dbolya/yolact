from backbone import DarkNetBackbone
import h5py
import torch

f = h5py.File('darknet53.h5', 'r')
m = f['model_weights']

yolo_keys = list(m.keys())
yolo_keys = [x for x in yolo_keys if len(m[x].keys()) > 0]
yolo_keys.sort()

sd = DarkNetBackbone().state_dict()

sd_keys = list(sd.keys())
sd_keys.sort()

# Note this won't work if there are 10 elements in some list but whatever that doesn't happen
layer_keys = list(set(['.'.join(x.split('.')[:-2]) for x in sd_keys]))
layer_keys.sort()

# print([x for x in sd_keys if x.startswith(layer_keys[0])])

mapping = {
	'.0.weight'      : ('conv2d_%d', 'kernel:0'),
	'.1.bias'        : ('batch_normalization_%d', 'beta:0'),
	'.1.weight'      : ('batch_normalization_%d', 'gamma:0'),
	'.1.running_var' : ('batch_normalization_%d', 'moving_variance:0'),
	'.1.running_mean': ('batch_normalization_%d', 'moving_mean:0'),
	'.1.num_batches_tracked': None,
}

for i, layer_key in zip(range(1, len(layer_keys) + 1), layer_keys):
	# This is pretty inefficient but I don't care
	for weight_key in [x for x in sd_keys if x.startswith(layer_key)]:
		diff = weight_key[len(layer_key):]
		
		if mapping[diff] is not None:
			yolo_key = mapping[diff][0] % i
			sub_key  = mapping[diff][1]

			yolo_weight = torch.Tensor(m[yolo_key][yolo_key][sub_key].value)
			if (len(yolo_weight.size()) == 4):
				yolo_weight = yolo_weight.permute(3, 2, 0, 1).contiguous()
			
			sd[weight_key] = yolo_weight

torch.save(sd, 'weights/darknet53.pth')

