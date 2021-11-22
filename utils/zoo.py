from pathlib import Path
from sparsezoo import Zoo
from functools import wraps


def is_valid_stub(stub: str) -> bool:
    return stub and stub.startswith('zoo:')


def is_onnx(onnx_file: str) -> bool:
    return onnx_file and Path(onnx_file).suffix == '.onnx'


def check_stub_before_invoke(func):
    @wraps(func)
    def wrapper(stub: str, *args, **kwargs):
        if is_valid_stub(stub):
            return func(stub, *args, **kwargs)
        raise ValueError("Invalid Stub, Check for typo!!!")

    return wrapper


def _get_model_framework_file(model, path: str):
    transfer_request = 'recipe_type=transfer' in path
    checkpoint_available = any(
        file.checkpoint for file in model.framework_files
    )
    final_available = any(
        not file.checkpoint for file in model.framework_files
    )

    if transfer_request and checkpoint_available:
        # checkpoints are saved for transfer learning use cases,
        # return checkpoint if avaiable and requested
        return [file for file in model.framework_files if file.checkpoint][0]

    elif final_available:
        # default to returning final state, if available
        return [file for file in model.framework_files if not
        file.checkpoint][0]

    raise ValueError(f"Could not find a valid framework file for {path}")


@check_stub_before_invoke
def get_model_onnx_from_stub(stub: str):
    """
    Downloads model from stub and returns its onnx filepath

    :param stub: SparseZoo stub for the model
    :return: path to model.onnx for the specified stub
    """
    model = Zoo.load_model_from_stub(stub=stub)
    return model.onnx_file.downloaded_path()


@check_stub_before_invoke
def get_checkpoint_from_stub(stub: str) -> str:
    """
    Helper to download a model checkpoint from SparseZoo Stub

    NOTE: model and checkpoint are used interchangeably

    :param stub: A valid SparseZoo Stub
    :raises: ValueError if invalid stub given
    :return: path to model checkpoint (after downloading from SparseZoo)
    """

    model = Zoo.load_model_from_stub(stub=stub)
    file = _get_model_framework_file(model, stub)
    path = file.downloaded_path()
    return path
