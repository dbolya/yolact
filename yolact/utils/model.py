import os
import gdown

def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "yolact_resnet50_54_800000.pth": "1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0",
        "yolact_plus_resnet50_54_800000.pth": "1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP"
    }
    if model_name not in download_paths:
        raise Exception("Invalid Model Name")
    url = f'https://drive.google.com/uc?id={download_paths[model_name]}'
    output = f'models/{model_name}'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)