import os
import json
import time
import sys

from collections import defaultdict
import matplotlib.pyplot as plt

# Because Python's package heierarchy system sucks
if __name__ == '__main__':
    from nvinfo import gpu_info, visible_gpus, nvsmi_available
else:
    from .nvinfo import gpu_info, visible_gpus, nvsmi_available

class Log:
    """
    A class to log information during training per information and save it out.
    It also can include extra debug information like GPU usage / temp automatically.

    Extra args:
     - session_data: If you have any data unique to this session, put it here.
     - overwrite: Whether or not to overwrite a pre-existing log with this name.
     - log_gpu_stats: Whether or not to log gpu information like temp, usage, memory.
                      Note that this requires nvidia-smi to be present in your PATH.
     - log_time: Also log the time in each iteration.
    """

    def __init__(self, log_name:str, log_dir:str='logs/', session_data:dict={},
                 overwrite:bool=False, log_gpu_stats:bool=True, log_time:bool=True):
        
        if log_gpu_stats and not nvsmi_available():
            print('Warning: Log created with log_gpu_stats=True, but nvidia-smi ' \
                  'was not found. Setting log_gpu_stats to False.')
            log_gpu_stats = False
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_path = os.path.join(log_dir, log_name + '.log')

        if os.path.exists(self.log_path) and overwrite:
            os.unlink(self.log_path)

        if os.path.exists(self.log_path):
            # Log already exists, so we're going to add to it. Increment the session counter.
            with open(self.log_path, 'r') as f:
                for last in f: pass

                if len(last) > 1:
                    self.session = json.loads(last)['session'] + 1
                else:
                    self.session = 0
        else:
            self.session = 0


        self.log_gpu_stats = log_gpu_stats
        self.log_time = log_time

        if self.log_gpu_stats:
            self.visible_gpus = visible_gpus()
    

        self._log_session_header(session_data)


    def _log_session_header(self, session_data:dict):
        """
        Log information that does not change between iterations here.
        This is to cut down on the file size so you're not outputing this every iteration.
        """
        info = {}
        info['type'] = 'session'
        info['session'] = self.session

        info['data'] = session_data

        if self.log_gpu_stats:
            keys = ['idx', 'name', 'uuid', 'pwr_cap', 'mem_total']

            gpus = gpu_info()
            info['gpus'] = [{k: gpus[i][k] for k in keys} for i in self.visible_gpus]
        
        if self.log_time:
            info['time'] = time.time()

        out = json.dumps(info) + '\n'

        with open(self.log_path, 'a') as f:
            f.write(out)


    def log(self, type:str, data:dict={}, **kwdargs):
        """
        Add an iteration to the log with the specified data points.
        Type should be the type of information this is (e.g., train, valid, etc.)
        
        You can either pass data points as kwdargs, or as a dictionary (or both!).
        Values should be json-serializable.
        """
        info = {}
        
        info['type'] = type
        info['session'] = self.session

        kwdargs.update(data)
        info['data'] = kwdargs

        if self.log_gpu_stats:
            keys = ['fan_spd', 'temp', 'pwr_used', 'mem_used', 'util']
            
            gpus = gpu_info()
            info['gpus'] = [{k: gpus[i][k] for k in keys} for i in self.visible_gpus]
        
        if self.log_time:
            info['time'] = time.time()
        
        out = json.dumps(info) + '\n'

        with open(self.log_path, 'a') as f:
            f.write(out)




class LogVisualizer():


    def __init__(self, path:str):
        self.log = defaultdict(lambda: [])

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    js = json.loads(line)
                    self.log[js['type']].append(js)
    
    def _decode(self, query:str) -> list:
        path, select = (query.split(';') + [''])[:2]
        
        if select.strip() == '':
            select = lambda x: True
        else:
            select = eval('lambda x: ' + select)

        path = path.split('.')
        
        for idx, val in enumerate(path):
            try:
                path[idx] = int(val)
            except ValueError:
                pass
        
        return path, select

    def _follow(self, entry:dict, query:list):
        path, select = query
        
        for p in path:
            try:
                if type(entry) is dict and type(p) is not str:
                    p = str(p)

                entry = entry[p]
            except (TypeError, IndexError, KeyError):
                return None
        
        if select(entry):
            return entry
        else:
            return None

    def plot(self, entry_type:str, x:str, y:str):
        log = self.log[entry_type]

        query_x = self._decode(x)
        query_y = self._decode(y)

        x = []
        y = []

        for datum in log:
            val_x = self._follow(datum, query_x)
            val_y = self._follow(datum, query_y)

            if val_x is not None and val_y is not None:
                x.append(val_x)
                y.append(val_y)
        
        plt.plot(x, y)
        plt.show()








if __name__ == '__main__':
    if len(sys.argv) < 4+1:
        print('Usage: python utils/logger.py <LOG_FILE> <TYPE> <X QUERY> <Y QUERY>')
        exit()
    
    vis = LogVisualizer(sys.argv[1])
    vis.plot(sys.argv[2], sys.argv[3], sys.argv[4])