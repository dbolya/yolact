from abc import ABC, abstractmethod
import contextlib

from layers import Detect
from yolact import FastMaskIoUNet

with contextlib.suppress(ModuleNotFoundError):
    from deepsparse import compile_model

with contextlib.suppress(ModuleNotFoundError):
    import onnxruntime as ort

import torch


def _convert_tensors_to_numpy(inputs):
    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()
    return inputs


def _make_list(inputs):
    if not isinstance(inputs, list):
        batch = [inputs]
    return batch


class YOLACTWrapper(ABC):
    """
    Base abstract class for wrapping YOLACT inference
    """
    def __init__(self, cfg):
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
                              conf_thresh=cfg.nms_conf_thresh,
                              nms_thresh=cfg.nms_thresh)
        self._maskiou_net = FastMaskIoUNet()
        self._keys = ['loc', 'conf', 'mask', 'priors', 'proto']

    def __call__(self, _inputs):
        """
        Run inputs through the graph and return returns detections

        :param _inputs: tensor type inputs
        """
        _inputs = _convert_tensors_to_numpy(_inputs)
        _outputs = self.forward(_inputs)
        return self._postprocess(_outputs)

    def _postprocess(self, outputs):
        if isinstance(outputs, dict):
            _outputs_iter = outputs.values()
        else:
            _outputs_iter = outputs
        outs = dict(zip(self._keys, map(torch.from_numpy, _outputs_iter)))
        return self.detect(outs, self)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward inputs through the model graph returns outputs

        :pre-condition: Expects numpy type inputs
        :post-condition: Returns numpy type outputs
        """
        pass


class DeepSparseWrapper(YOLACTWrapper):
    """
    Inference Wrapper for YOLACT that uses DeepSparse Engine
    """
    def __init__(self, filepath, cfg, num_cores=None,
                 batch_size=1):
        super().__init__(cfg)
        self.engine = compile_model(filepath, batch_size=batch_size,
                                    num_cores=num_cores)

    def forward(self, inputs):
        batch = _make_list(inputs)
        return self.engine.mapped_run(batch)


class ORTWrapper(YOLACTWrapper):
    """
    Inference Wrapper for YOLACT that uses ONNX Runtime Engine
    """
    def __init__(self, filepath, cfg):
        super().__init__(cfg)
        self.engine = ort.InferenceSession(filepath)

    def forward(self, inputs):
        return self.engine.run(None, {'input': inputs})
