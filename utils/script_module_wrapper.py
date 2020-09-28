import torch
import torch.nn

#  As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module #TODO remove once nn.Module supports JIT script modules
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn
