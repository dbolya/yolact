import torch
from torch import nn
import torch.nn.functional as F

#locals
from data.config import Config
from utils.functions import make_net
from utils.script_module_wrapper import ScriptModuleWrapper, script_method_wrapper

class FastMaskIoUNet(ScriptModuleWrapper):

    def __init__(self, config:Config):
        super().__init__()

        cfg = config
        input_channels = 1
        last_layer = [(cfg.num_classes-1, 1, {})]
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)
        return maskiou_p

