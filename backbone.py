import torch
import torch.nn as nn
import pickle

from collections import OrderedDict

class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, atrous_layers=[], block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          dilation=self.dilation),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.norm_layer, self.dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)

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
        state_dict = torch.load(path)

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx-1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)




class ResNetBackboneGN(ResNetBackbone):

    def __init__(self, layers, num_groups=32):
        super().__init__(layers, norm_layer=lambda x: nn.GroupNorm(num_groups, x))

    def init_backbone(self, path):
        """ The path here comes from detectron. So we load it differently. """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1') # From the detectron source
            state_dict = state_dict['blobs']
        
        our_state_dict_keys = list(self.state_dict().keys())
        new_state_dict = {}
    
        gn_trans     = lambda x: ('gn_s' if x == 'weight' else 'gn_b')
        layeridx2res = lambda x: 'res' + str(int(x)+2)
        block2branch = lambda x: 'branch2' + ('a', 'b', 'c')[int(x[-1:])-1]

        # Transcribe each Detectron weights name to a Yolact weights name
        for key in our_state_dict_keys:
            parts = key.split('.')
            transcribed_key = ''

            if (parts[0] == 'conv1'):
                transcribed_key = 'conv1_w'
            elif (parts[0] == 'bn1'):
                transcribed_key = 'conv1_' + gn_trans(parts[1])
            elif (parts[0] == 'layers'):
                if int(parts[1]) >= self.num_base_layers: continue

                transcribed_key = layeridx2res(parts[1])
                transcribed_key += '_' + parts[2] + '_'

                if parts[3] == 'downsample':
                    transcribed_key += 'branch1_'
                    
                    if parts[4] == '0':
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[5])
                else:
                    transcribed_key += block2branch(parts[3]) + '_'

                    if 'conv' in parts[3]:
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[4])

            new_state_dict[key] = torch.Tensor(state_dict[transcribed_key])
        
        # strict=False because we may have extra unitialized layers at this point
        self.load_state_dict(new_state_dict, strict=False)







def darknetconvlayer(in_channels, out_channels, *args, **kwdargs):
    """
    Implements a conv, activation, then batch norm.
    Arguments are passed into the conv layer.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwdargs, bias=False),
        nn.BatchNorm2d(out_channels),
        # Darknet uses 0.1 here.
        # See https://github.com/pjreddie/darknet/blob/680d3bde1924c8ee2d1c1dea54d3e56a05ca9a26/src/activations.h#L39
        nn.LeakyReLU(0.1, inplace=True)
    )

class DarkNetBlock(nn.Module):
    """ Note: channels is the lesser of the two. The output will be expansion * channels. """

    expansion = 2

    def __init__(self, in_channels, channels):
        super().__init__()

        self.conv1 = darknetconvlayer(in_channels, channels,                  kernel_size=1)
        self.conv2 = darknetconvlayer(channels,    channels * self.expansion, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x




class DarkNetBackbone(nn.Module):
    """
    An implementation of YOLOv3's Darnet53 in
    https://pjreddie.com/media/files/papers/YOLOv3.pdf

    This is based off of the implementation of Resnet above.
    """

    def __init__(self, layers=[1, 2, 8, 8, 4], block=DarkNetBlock):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        
        self._preconv = darknetconvlayer(3, 32, kernel_size=3, padding=1)
        self.in_channels = 32
        
        self._make_layer(block, 32,  layers[0])
        self._make_layer(block, 64,  layers[1])
        self._make_layer(block, 128, layers[2])
        self._make_layer(block, 256, layers[3])
        self._make_layer(block, 512, layers[4])

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
    
    def _make_layer(self, block, channels, num_blocks, stride=2):
        """ Here one layer means a string of n blocks. """
        layer_list = []

        # The downsample layer
        layer_list.append(
            darknetconvlayer(self.in_channels, channels * block.expansion,
                             kernel_size=3, padding=1, stride=stride))

        # Each block inputs channels and outputs channels * expansion
        self.in_channels = channels * block.expansion
        layer_list += [block(self.in_channels, channels) for _ in range(num_blocks)]

        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layer_list))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """

        x = self._preconv(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return outs

    def add_layer(self, conv_channels=1024, stride=2, depth=1, block=DarkNetBlock):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, num_blocks=depth, stride=stride)
    
    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(torch.load(path), strict=False)





class VGGBackbone(nn.Module):
    """
    Args:
        - cfg: A list of layers given as lists. Layers can be either 'M' signifying
                a max pooling layer, a number signifying that many feature maps in
                a conv layer, or a tuple of 'M' or a number and a kwdargs dict to pass
                into the function that creates the layer (e.g. nn.MaxPool2d for 'M').
        - extra_args: A list of lists of arguments to pass into add_layer.
        - norm_layers: Layers indices that need to pass through an l2norm layer.
    """

    def __init__(self, cfg, extra_args=[], norm_layers=[]):
        super().__init__()
        
        self.channels = []
        self.layers = nn.ModuleList()
        self.in_channels = 3
        self.extra_args = list(reversed(extra_args)) # So I can use it as a stack

        # Keeps track of what the corresponding key will be in the state dict of the
        # pretrained model. For instance, layers.0.2 for us is 2 for the pretrained
        # model but layers.1.1 is 5.
        self.total_layer_count = 0
        self.state_dict_lookup = {}

        for idx, layer_cfg in enumerate(cfg):
            self._make_layer(layer_cfg)

        self.norms = nn.ModuleList([nn.BatchNorm2d(self.channels[l]) for l in norm_layers])
        self.norm_lookup = {l: idx for idx, l in enumerate(norm_layers)}

        # These modules will be initialized by init_backbone,
        # so don't overwrite their initialization later.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, cfg):
        """
        Each layer is a sequence of conv layers usually preceded by a max pooling.
        Adapted from torchvision.models.vgg.make_layers.
        """

        layers = []

        for v in cfg:
            # VGG in SSD requires some special layers, so allow layers to be tuples of
            # (<M or num_features>, kwdargs dict)
            args = None
            if isinstance(v, tuple):
                args = v[1]
                v = v[0]

            # v should be either M or a number
            if v == 'M':
                # Set default arguments
                if args is None:
                    args = {'kernel_size': 2, 'stride': 2}

                layers.append(nn.MaxPool2d(**args))
            else:
                # See the comment in __init__ for an explanation of this
                cur_layer_idx = self.total_layer_count + len(layers)
                self.state_dict_lookup[cur_layer_idx] = '%d.%d' % (len(self.layers), len(layers))

                # Set default arguments
                if args is None:
                    args = {'kernel_size': 3, 'padding': 1}

                # Add the layers
                layers.append(nn.Conv2d(self.in_channels, v, **args))
                layers.append(nn.ReLU(inplace=True))
                self.in_channels = v
        
        self.total_layer_count += len(layers)
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply an l2norm module to the selected layers
            # Note that this differs from the original implemenetation
            if idx in self.norm_lookup:
                x = self.norms[self.norm_lookup[idx]](x)
            outs.append(x)
        
        return outs

    def transform_key(self, k):
        """ Transform e.g. features.24.bias to layers.4.1.bias """
        vals = k.split('.')
        layerIdx = self.state_dict_lookup[int(vals[0])]
        return 'layers.%s.%s' % (layerIdx, vals[1])

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        state_dict = OrderedDict([(self.transform_key(k), v) for k,v in state_dict.items()])

        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=128, downsample=2):
        """ Add a downsample layer to the backbone as per what SSD does. """
        if len(self.extra_args) > 0:
            conv_channels, downsample = self.extra_args.pop()
        
        padding = 1 if downsample > 1 else 0
        
        layer = nn.Sequential(
            nn.Conv2d(self.in_channels, conv_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels*2, kernel_size=3, stride=downsample, padding=padding),
            nn.ReLU(inplace=True)
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
