from torchvision.models.resnet import ResNet, Bottleneck
import torch
import torch.nn as nn
from l2norm import L2Norm

from collections import OrderedDict

class ResNetBackbone(ResNet):

    def __init__(self, layers):
        # These will be populated by _make_layer
        self.channels = [] 
        self.layers = []

        super().__init__(Bottleneck, layers)

        # Since we can't assign modules before nn.Module.__init__
        self.layers = nn.ModuleList(self.layers)

        # We won't need these where we're going		
        del self.fc
        del self.avgpool

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        layer = super()._make_layer(block, planes, blocks, stride)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def forward(self, x):
        """ Returns a list of convouts for each layer. """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return outs

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(torch.load(path), strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)


class VGGBackbone(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        
        self.channels = []
        self.layers = nn.ModuleList()
        self.in_channels = 3

        # Keeps track of what the corresponding key will be in the state dict of the
        # pretrained model. For instance, layers.0.2 for us is 2 for the pretrained
        # model but layers.1.1 is 5.
        self.total_layer_count = 0
        self.state_dict_lookup = {}

        # Trust me on this one
        cfg_str = ' '.join([str(l) for l in cfg])
        # This is the most pythonic way, I swear
        cfg_layers_str = cfg_str.split('M')
        # I'm so sorry
        cfg_layers = [[int(x) for x in y.strip().split(' ')] for y in cfg_layers_str if y != '']
        # The end result is splitting e.g. [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
        # into [[64, 64], ['M', 128, 128], ['M', 256, 256]]
        cfg_layers = [([] if i == 0 else ['M']) + cfg_layers[i] for i in range(len(cfg_layers))]

        for layer_cfg in cfg_layers:
            self._make_layer(layer_cfg)

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, cfg):
        """
        Each layer is a sequence of conv layers usually preceded by a max pooling.
        Addapted from torchvision.models.vgg.make_layers.
        """
        layers = []

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # See the comment in __init__ for an explanation of this
                cur_layer_idx = self.total_layer_count + len(layers)
                self.state_dict_lookup[cur_layer_idx] = '%d.%d' % (len(self.layers), len(layers))

                # Add the layers
                layers.append(nn.Conv2d(self.in_channels, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                self.in_channels = v

        if len(self.layers) == 3:
            # An extra layer that doesn't exist in VGG originally,
            # so we have to reduce the layer count.
            layers.append(L2Norm(self.in_channels, scale=20))
            self.total_layer_count -= 1
        
        self.total_layer_count += len(layers)
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []

        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        
        return outs

    def transform_key(self, k):
        """ Transform e.g. features.24.bias to layers.4.1.bias """
        vals = k.split('.')
        if int(vals[0]) in (31, 33):
            return k
        layerIdx = self.state_dict_lookup[int(vals[0])]
        return 'layers.%s.%s' % (layerIdx, vals[1])

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        state_dict = OrderedDict([(self.transform_key(k), v) for k,v in state_dict.items()])

        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=128, downsample=2):
        """ Add a downsample layer to the backbone as per what SSD does. """
        layer = nn.Sequential(
            nn.Conv2d(self.in_channels, conv_channels, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels*2, kernel_size=3, stride=downsample),
            nn.ReLU()
        )

        self.in_channels = conv_channels*2
        self.channels.append(self.in_channels)
        self.layers.append(layer)
        
                


def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """
    backbone = cfg.type(*cfg.args)

    # Add downsampling layers until we reach the number we need
    num_layers = max(cfg.selected_layers) + 1

    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone
