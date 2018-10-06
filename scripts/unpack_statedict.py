import torch
import sys, os

# Usage python scripts/unpack_statedict.py path_to_pth out_folder/
# Make sure to include that slash after your out folder, since I can't
# be arsed to do path concatenation so I'd rather type out this comment

print('Loading state dict...')
state = torch.load(sys.argv[1])

if not os.path.exists(sys.argv[2]):
	os.mkdir(sys.argv[2])

print('Saving stuff...')
for key, val in state.items():
	torch.save(val, sys.argv[2] + key)
