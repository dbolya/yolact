import torch

# for making bounding boxes pretty
COLORS = ((244, 67, 54),
          (233, 30, 99),
          (156, 39, 176),
          (103, 58, 183),
          (63, 81, 181),
          (33, 150, 243),
          (3, 169, 244),
          (0, 188, 212),
          (0, 150, 136),
          (76, 175, 80),
          (139, 195, 74),
          (205, 220, 57),
          (255, 235, 59),
          (255, 193, 7),
          (255, 152, 0),
          (255, 87, 34),
          (121, 85, 72),
          (158, 158, 158),
          (96, 125, 139))

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)


class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    -> No u.
    """

    def __init__(self, config_dict: dict = None):
        if config_dict:
            for key, val in config_dict.items():
                self.__setattr__(key, val)

    def dict_to_Config(self, config_dict: dict):
        for key, val in config_dict.items():
            if isinstance(val, dict) and key not in ["head_layer_params"]:
                dict_val = val
                val = Config()
                val.dict_to_Config(dict_val)
            self.__setattr__(key, val)


activation_func = Config({
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu': lambda x: torch.nn.functional.relu(x, inplace=True),
    'none': lambda x: x
})

mask_type = Config({
    'direct': 0,
    'lincomb': 1
})

# Default config
cfg = Config()


def set_cfg(config_dict: dict):
    cfg.dict_to_Config(config_dict)


def set_dataset(dataset_name: str):
    """ Sets the dataset of the current config. """
    cfg.dataset = eval(dataset_name)
