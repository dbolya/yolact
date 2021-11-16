from sparsezoo import Zoo


def is_valid_stub(stub: str) -> bool:
    return stub.startswith('zoo:')


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


def download_checkpoint_from_stub(stub: str) -> str:
    """
    Helper to download a model checkpoint from SparseZoo Stub

    NOTE: model and checkpoint are used interchangeably

    :param stub: A valid SparseZoo Stub
    :raises: ValueError if invalid stub given
    :return: path to model checkpoint (after downloading from SparseZoo)
    """

    if is_valid_stub(stub=stub):
        model = Zoo.load_model_from_stub(stub=stub)
        file = _get_model_framework_file(model, stub)
        path = file.downloaded_path()
        return path

    raise ValueError("Invalid Stub, Check for typo!!!")

