import torch
import torch.nn as nn
import os
import math
from collections import deque
from pathlib import Path
from layers.interpolate import InterpolateModule

class MovingAverage():
    """ Keeps an average window of the specified number of items. """

    def __init__(self, max_window_size=1000):
        self.max_window_size = max_window_size
        self.reset()

    def add(self, elem):
        """ Adds an element to the window, removing the earliest element if necessary. """
        if not math.isfinite(elem):
            print('Warning: Moving average ignored a value of %f' % elem)
            return
        
        self.window.append(elem)
        self.sum += elem

        if len(self.window) > self.max_window_size:
            self.sum -= self.window.popleft()
    
    def append(self, elem):
        """ Same as add just more pythonic. """
        self.add(elem)

    def reset(self):
        """ Resets the MovingAverage to its initial state. """
        self.window = deque()
        self.sum = 0

    def get_avg(self):
        """ Returns the average of the elements in the window. """
        return self.sum / max(len(self.window), 1)

    def __str__(self):
        return str(self.get_avg())
    
    def __repr__(self):
        return repr(self.get_avg())
    
    def __len__(self):
        return len(self.window)


class ProgressBar():
    """ A simple progress bar that just outputs a string. """

    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0
        
        self.cur_num_bars = -1
        self._update_str()

    def set_val(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        if self.cur_val < 0:
            self.cur_val = 0

        self._update_str()
    
    def is_finished(self):
        return self.cur_val == self.max_val

    def _update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)
    
    def __repr__(self):
        return self.string
    
    def __str__(self):
        return self.string


def init_console():
    """
    Initialize the console to be able to use ANSI escape characters on Windows.
    """
    if os.name == 'nt':
        from colorama import init
        init()


class SavePath:
    """
    Why is this a class?
    Why do I have a class for creating and parsing save paths?
    What am I doing with my life?
    """

    def __init__(self, model_name:str, epoch:int, iteration:int):
        self.model_name = model_name
        self.epoch = epoch
        self.iteration = iteration

    def get_path(self, root:str=''):
        file_name = self.model_name + '_' + str(self.epoch) + '_' + str(self.iteration) + '.pth'
        return os.path.join(root, file_name)

    @staticmethod
    def from_str(path:str):
        file_name = os.path.basename(path)
        
        if file_name.endswith('.pth'):
            file_name = file_name[:-4]
        
        params = file_name.split('_')

        if file_name.endswith('interrupt'):
            params = params[:-1]
        
        model_name = '_'.join(params[:-2])
        epoch = params[-2]
        iteration = params[-1]
        
        return SavePath(model_name, int(epoch), int(iteration))

    @staticmethod
    def remove_interrupt(save_folder):
        for p in Path(save_folder).glob('*_interrupt.pth'):
            p.unlink()
    
    @staticmethod
    def get_interrupt(save_folder):
        for p in Path(save_folder).glob('*_interrupt.pth'): 
            return str(p)
        return None
    
    @staticmethod
    def get_latest(save_folder, config):
        """ Note: config should be config.name. """
        max_iter = -1
        max_name = None

        for p in Path(save_folder).glob(config + '_*'):
            path_name = str(p)

            try:
                save = SavePath.from_str(path_name)
            except:
                continue 
            
            if save.model_name == config and save.iteration > max_iter:
                max_iter = save.iteration
                max_name = path_name

        return max_name

def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """
    def make_layer(layer_cfg):
        nonlocal in_channels
        
        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels