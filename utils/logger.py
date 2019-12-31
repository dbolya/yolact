import os
import json
import time
import sys

from typing import Union
import datetime

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Because Python's package heierarchy system sucks
if __name__ == '__main__':
    from nvinfo import gpu_info, visible_gpus, nvsmi_available
    from functions import MovingAverage
else:
    from .nvinfo import gpu_info, visible_gpus, nvsmi_available
    from .functions import MovingAverage

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

        # if os.path.exists(self.log_path) and overwrite:
        #     os.unlink(self.log_path)

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


class LogEntry():
    """ A class that allows you to navigate a dictonary using x.a.b[2].c, etc. """

    def __init__(self, entry:Union[dict, list]):
        self._ = entry

    def __getattr__(self, name):
        if name == '_':
            return self.__dict__['_']

        res = self.__dict__['_'][name]

        if type(res) == dict or type(res) == list:
            return LogEntry(res)
        else:
            return res
    
    def __getitem__(self, name):
        return self.__getattr__(name)

    def __len__(self):
        return len(self.__dict__['_'])

class LogVisualizer():

    COLORS = [
        'xkcd:azure',
        'xkcd:coral',
        'xkcd:turquoise',
        'xkcd:orchid',
        'xkcd:orange',
        
        'xkcd:blue',
        'xkcd:red',
        'xkcd:teal',
        'xkcd:magenta',
        'xkcd:orangered'
    ]

    def __init__(self):
        self.logs = []
        self.total_logs = []
        self.log_names = []
    
    def _decode(self, query:str) -> list:
        path, select = (query.split(';') + [''])[:2]
        
        if select.strip() == '':
            select = lambda x, s: True
        else:
            select = eval('lambda x, s: ' + select)

        if path.strip() == '':
            path = lambda x, s: x
        else:
            path = eval('lambda x, s: ' + path)
        
        return path, select

    def _follow(self, entry:LogEntry, query:list):
        path, select = query

        try:
            if select(entry, entry._s):
                res = path(entry, entry._s)

                if type(res) == LogEntry:
                    return res.__dict__['_']
                else:
                    return res
            else:
                return None
        except (KeyError, IndexError):
            return None

    def _color(self, idx:int):
        return self.COLORS[idx % len(self.COLORS)]

    def sessions(self, path:str):
        """ Prints statistics about the sessions in the file. """

        if not os.path.exists(path):
            print(path + ' doesn\'t exist!')
            return

        cur_session = None
        cur_time = 0
        last_time = 0
        num_entries = 0

        def pop_session():
            delta = last_time - cur_time
            time_str = str(datetime.timedelta(seconds=delta)).split('.')[0]
            print('Session % 3d: % 8d entries | %s elapsed' % (cur_session, num_entries, time_str))

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    js = json.loads(line)
                    if js['type'] == 'session':
                        if cur_session is not None:
                            pop_session()
                        cur_time = js['time']
                        cur_session = js['session']
                        num_entries = 0
                    last_time = js['time']
                    num_entries += 1
        
        pop_session()

    def add(self, path:str, session:Union[int,list]=None):
        """ Add a log file to the list of logs being considered. """

        log = defaultdict(lambda: [])
        total_log = []

        if not os.path.exists(path):
            print(path + ' doesn\'t exist!')
            return

        session_idx = 0
        ignoring = True
        
        def valid(idx):
            if session is None:
                return True
            elif type(session) == int:
                return (idx == session)
            else:
                return idx in session

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    js = json.loads(line)
                    
                    _type = js['type']
                    if _type == 'session':
                        session_idx = js['session']
                        ignoring = not valid(session_idx)

                    if not ignoring:
                        ljs = LogEntry(js)
                        if _type == 'session':
                            js['_s'] = ljs
                        else:
                            js['_s'] =log['session'][-1]
                        log[_type].append(ljs)
                        total_log.append(ljs)
        
        name = os.path.basename(path)
        if session is not None:
            name += ' (Session %s)' % session

        self.logs.append(log)
        self.total_logs.append(total_log)
        self.log_names.append(name)

    def query(self, x:Union[str, list], entry_type:str=None, x_idx:int=None, log_idx:int=None) -> list:
        """
        Given a query string (can be already decoded for faster computation), query the entire log
        and return all values found by that query. If both log_idx and x_idx is None, this will be
        a list of lists in the form [log_idx][result_idx]. If x_idx is not None, then the result
        will be a list of [log_idx]. If both are not none, the return value will be a single query
        return value. With entry_type=None, this will search the entire log.
        """

        if type(x) is not list:
            x = self._decode(x)
        
        res = []

        for idx in (range(len(self.logs)) if log_idx is None else [log_idx]):
            candidates = []
            log = self.total_logs[idx] if entry_type is None else self.logs[idx][entry_type]

            for entry in log:
                candidate = self._follow(entry, x)
                if candidate is not None:
                    candidates.append(candidate)
            
            if x_idx is not None:
                candidates = candidates[x_idx]
            res.append(candidates)
        
        if log_idx is not None:
            res = res[0]
        return res

    def check(self, entry_type:str, x:str):
        """ Checks the log for the valid keys for this input. """
        keys = set()
        x = self._decode(x)

        for log in self.logs:
            for datum in log[entry_type]:
                res = self._follow(datum, x)

                if type(res) == dict:
                    for key in res.keys():
                        keys.add(key)
                elif type(res) == list:
                    keys.add('< %d' % len(res))
    
        return list(keys)

    def plot(self, entry_type:str, x:str, y:str, smoothness:int=0):
        """ Plot sequential log data. """

        query_x = self._decode(x)
        query_y = self._decode(y)

        for idx, (log, name) in enumerate(zip(self.logs, self.log_names)):
            log = log[entry_type]

            if smoothness > 1:
                avg = MovingAverage(smoothness)

            _x = []
            _y = []

            for datum in log:
                val_x = self._follow(datum, query_x)
                val_y = self._follow(datum, query_y)

                if val_x is not None and val_y is not None:
                    if smoothness > 1:
                        avg.append(val_y)
                        val_y = avg.get_avg()

                        if len(avg) < smoothness // 10:
                            continue
                        
                    _x.append(val_x)
                    _y.append(val_y)
            
            plt.plot(_x, _y, color=self._color(idx), label=name)
        
        plt.title(y.replace('x.', entry_type + '.'))
        plt.legend()
        plt.grid(linestyle=':', linewidth=0.5)
        plt.show()

    def bar(self, entry_type:str, x:str, labels:list=None, diff:bool=False, x_idx:int=-1):
        """ Plot a bar chart. The result of x should be list or dictionary. """

        query = self._decode(x)

        data_points = []

        for idx, (log, name) in enumerate(zip(self.logs, self.log_names)):
            log = log[entry_type]

            candidates = []

            for entry in log:
                test = self._follow(entry, query)

                if type(test) == dict:
                    candidates.append(test)
                elif type(test) == list:
                    candidates.append({idx: v for idx, v in enumerate(test)})
            
            if len(candidates) > 0:
                data_points.append((name, candidates[x_idx]))
        
        if len(data_points) == 0:
            print('Warning: Nothing to show in bar chart!')
            return

        names = [x[0] for x in data_points]
        data_points = [x[1] for x in data_points]

        # Construct the labels for the data
        if labels is not None:
            data_labels = labels
        else:
            data_labels = set()
            for datum in data_points:
                for k in datum:
                    data_labels.add(k)
                
            data_labels = list(data_labels)
            data_labels.sort()
        

        data_values = [[(datum[k] if k in datum else None) for k in data_labels] for datum in data_points]

        if diff:
            for idx in reversed(range(len(data_values))):
                for jdx in range(len(data_labels)):
                    if data_values[0][jdx] is None or data_values[idx][jdx] is None:
                        data_values[idx][jdx] = None
                    else:
                        data_values[idx][jdx] -= data_values[0][jdx]


        series_labels = names

        # Plot the graph now
        num_bars = len(series_labels)
        bar_width = 1 / (num_bars + 1)
        
        # Set position of bar on X axis
        positions = [np.arange(len(data_labels))]
        for _ in range(1, num_bars):
            positions.append([x + bar_width for x in positions[-1]])
        
        # Make the plot
        for idx, (series, data, pos) in enumerate(zip(series_labels, data_values, positions)):
            plt.bar(pos, data, color=self._color(idx), width=bar_width, edgecolor='white', label=series)
        
        # Add xticks on the middle of the group bars
        plt.title(x.replace('x.', entry_type + '.') + (' diff' if diff else ''))
        plt.xticks([r + bar_width for r in range(len(data_labels))], data_labels)
        
        # Create legend & Show graphic
        plt.legend()
        plt.show()

        

    def elapsed_time(self, cond1:str='', cond2:str='', legible:bool=True) -> list:
        """
        Returns the elapsed time between two entries based on the given conditionals.
        If a query isn't specified, the first / last entry will be used. The first query
        uses the first value and the second query uses the last value in the results.

        Setting legible to true returns human-readable results, while false returns seconds.
        """
        q1 = 'x.time; ' + cond1
        q2 = 'x.time; ' + cond2

        x1 = self.query(q1, x_idx=0)
        x2 = self.query(q2, x_idx=-1)
        
        diff = (lambda x: str(datetime.timedelta(seconds=x)).split('.')[0]) if legible else lambda x: x

        return [diff(b - a) for a, b in zip(x1, x2)]











if __name__ == '__main__':
    if len(sys.argv) < 4+1:
        print('Usage: python utils/logger.py <LOG_FILE> <TYPE> <X QUERY> <Y QUERY>')
        exit()
    
    vis = LogVisualizer()
    vis.add(sys.argv[1])
    vis.plot(sys.argv[2], sys.argv[3], sys.argv[4])
