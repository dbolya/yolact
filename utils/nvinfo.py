# My version of nvgpu because nvgpu didn't have all the information I was looking for.
import re
import subprocess
import shutil
import os

def gpu_info() -> list:
    """
    Returns a dictionary of stats mined from nvidia-smi for each gpu in a list.
    Adapted from nvgpu: https://pypi.org/project/nvgpu/, but mine has more info.
    """
    gpus = [line for line in _run_cmd(['nvidia-smi', '-L']) if line]
    gpu_infos = [re.match('GPU ([0-9]+): ([^(]+) \(UUID: ([^)]+)\)', gpu).groups() for gpu in gpus]
    gpu_infos = [dict(zip(['idx', 'name', 'uuid'], info)) for info in gpu_infos]
    gpu_count = len(gpus)

    lines = _run_cmd(['nvidia-smi'])
    selected_lines = lines[7:7 + 3 * gpu_count]
    for i in range(gpu_count):
        mem_used, mem_total = [int(m.strip().replace('MiB', '')) for m in
                               selected_lines[3 * i + 1].split('|')[2].strip().split('/')]
        
        pw_tmp_info, mem_info, util_info = [x.strip() for x in selected_lines[3 * i + 1].split('|')[1:-1]]
        
        pw_tmp_info = [x[:-1] for x in pw_tmp_info.split(' ') if len(x) > 0]
        fan_speed, temperature, pwr_used, pwr_cap = [int(pw_tmp_info[i]) for i in (0, 1, 3, 5)]
        gpu_infos[i]['fan_spd' ] = fan_speed
        gpu_infos[i]['temp'    ] = temperature
        gpu_infos[i]['pwr_used'] = pwr_used
        gpu_infos[i]['pwr_cap' ] = pwr_cap

        mem_used, mem_total = [int(x) for x in mem_info.replace('MiB', '').split(' / ')]
        gpu_infos[i]['mem_used' ] = mem_used
        gpu_infos[i]['mem_total'] = mem_total

        utilization = int(util_info.split(' ')[0][:-1])
        gpu_infos[i]['util'] = utilization

        gpu_infos[i]['idx'] = int(gpu_infos[i]['idx'])

    return gpu_infos

def nvsmi_available() -> bool:
    """ Returns whether or not nvidia-smi is present in this system's PATH. """
    return shutil.which('nvidia-smi') is not None


def visible_gpus() -> list:
    """ Returns a list of the indexes of all the gpus visible to pytorch. """

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        return list(range(len(gpu_info())))
    else:
        return [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]




def _run_cmd(cmd:list) -> list:
    """ Runs a command and returns a list of output lines. """
    output = subprocess.check_output(cmd)
    output = output.decode('UTF-8')
    return output.split('\n')