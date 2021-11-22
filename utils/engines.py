"""
Utility classes and methods for supported engines
"""
from enum import Enum

class _ExtendedEnum(Enum):
    @classmethod
    def supported(cls):
        return list(map(lambda c: c.value, cls))


class Engine(_ExtendedEnum):
    """
    Class representing supported Inference Engines
    """
    DEEPSPARSE = 'deepsparse'
    ORT = 'ort'
    TORCH = 'torch'

    @classmethod
    def from_filepath(cls, filepath: str, default_onnx=None):
        """
        Try to infer engine from filename;
        Note this is experimental and might fail; if no filepath given defaults
        to TORCH, Default engine for ONNX is DeepSparse, for ORT engine pass a
        default parameter with 'ort'

        :param filepath:
        :return:
        """
        if (
                not filepath or filepath.endswith('.pt')
                or filepath.endswith('.pth')
        ):
            return Engine.TORCH

        if not default_onnx or default_onnx != 'ort':
            default_onnx = Engine.DEEPSPARSE
        else:
            default_onnx = Engine.ORT

        if filepath.endswith('.onnx') or filepath.startswith('zoo:'):
            return default_onnx

        raise ValueError(f"Cannot Infer Engine from {filepath}")