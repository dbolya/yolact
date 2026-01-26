import torch.nn.functional as F
from torch import nn


class InterpolateModule(nn.Module):
    """Module version of `torch.nn.functional.interpolate`."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)
