import time
from collections import defaultdict

_total_times = defaultdict(lambda:  0)
_start_times = defaultdict(lambda: -1)
_disabled_names = set()

def disable(fn_name):
	""" Disables the given function name fom being considered for the average or outputted in print_stats. """
	_disabled_names.add(fn_name)

def enable(fn_name):
	""" Enables function names disabled by disable. """
	_disabled_names.remove(fn_name)

def reset():
	""" Resets the current timer. Call this at the start of an iteration. """
	_total_times.clear()
	_start_times.clear()

def start(fn_name):
	""" Start timing the specific function. """
	_start_times[fn_name] = time.time()

def stop(fn_name):
	""" Stop timing the specific function and add it to the total. """
	if _start_times[fn_name] > -1:
		_total_times[fn_name] += time.time() - _start_times[fn_name]
	else:
		print('Warning: timer for %s stopped before starting!' % fn_name)

def print_stats():
	""" Prints the current timing information into a table. """
	print()

	all_fn_names = [k for k in _total_times.keys() if k not in _disabled_names]

	max_name_width = max([len(k) for k in all_fn_names] + [4])
	if max_name_width % 2 == 1: max_name_width += 1
	format_str = ' {:>%d} | {:>10.4f} ' % max_name_width

	header = (' {:^%d} | {:^10} ' % max_name_width).format('Name', 'Time (ms)')
	print(header)

	sep_idx = header.find('|')
	sep_text = ('-' * sep_idx) + '+' + '-' * (len(header)-sep_idx-1)
	print(sep_text)

	for name in all_fn_names:
		print(format_str.format(name, _total_times[name]*1000))
	
	print(sep_text)
	print(format_str.format('Total', total_time()*1000))
	print()

def total_time():
	""" Returns the total amount accumulated across all functions. """ 
	return sum([elapsed_time for name, elapsed_time in _total_times.items() if name not in _disabled_names])
