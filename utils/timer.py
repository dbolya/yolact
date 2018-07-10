import time
from collections import defaultdict

_total_times = defaultdict(lambda:  0)
_start_times = defaultdict(lambda: -1)
_use_timer = False

def disable():
	_use_timer = False

def enable():
	_use_timer = True

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

	max_name_width = max([len(k) for k in _total_times.keys()] + [4])
	if max_name_width % 2 == 1: max_name_width += 1
	format_str = ' {:>%d} | {:>10.4f} ' % max_name_width

	header = (' {:^%d} | {:^10} ' % max_name_width).format('Name', 'Time (ms)')
	print(header)

	sep_idx = header.find('|')
	sep_text = ('-' * sep_idx) + '+' + '-' * (len(header)-sep_idx-1)
	print(sep_text)

	for name, elapsed_time in _total_times.items():
		print(format_str.format(name, elapsed_time*1000))
	
	print(sep_text)
	print(format_str.format('Total', total_time()*1000))
	print()

def total_time():
	""" Returns the total amount accumulated across all functions. """ 
	return sum(_total_times.values())
