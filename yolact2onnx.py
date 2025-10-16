from data import cfg, MEANS, STD, set_cfg, mask_type
import torch.nn.functional as F
import torch
import numpy as np


def main(args):
    config = args.config
    trained_model = args.ckpt_path
    
    #load model
    set_cfg(config)
    cfg.mask_proto_debug = False
    cfg.export_onnx=True
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    print('Loading model...', end='')
    from yolact import Yolact
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    dummy = torch.randn(1,3,550,550)
    torch.onnx.export(net, dummy, "yolact_plus.onnx", opset_version=13)
    net.detect.use_fast_nms = True
    net.detect.use_cross_class_nms = False
    if cfg.use_maskiou:
        dummy2 = torch.randn(1,1,138,138)
        torch.onnx.export(net.maskiou_net, dummy2, "maskiou_net.onnx", 
                input_names = ['input'],   # the model's input names
                output_names = ['output'],
                dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},
                opset_version=13)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Yolact onnx export')
    parser.add_argument('--config', type=str, default='yolact_plus_base_config',
                    help='The config object to use.')
    parser.add_argument('--ckpt_path', type=str, default="",
                    help='Give the path to trained pytorch weights')
    parser.add_argument('--onnx_paths', type=str, default="",  nargs='+',
                    help='Give the path to exported ONNX weights')
    
    args = parser.parse_args()
    main(args)
    