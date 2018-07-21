import time
from collections import defaultdict

_total_times = defaultdict(lambda:  0)
_start_times = defaultdict(lambda: -1)
_disabled_names = set()
_timer_stack = []
_running_timer = None

def disable(fn_name):
	""" Disables the given function name fom being considered for the average or outputted in print_stats. """
	_disabled_names.add(fn_name)

def enable(fn_name):
	""" Enables function names disabled by disable. """
	_disabled_names.remove(fn_name)

def reset():
	""" Resets the current timer. Call this at the start of an iteration. """
	global _running_timer
	_total_times.clear()
	_start_times.clear()
	_timer_stack.clear()
	_running_timer = None

def start(fn_name, use_stack=True):
	"""
	Start timing the specific function.
	Note: If use_stack is True, only one timer can be active at a time.
	      Once you stop this timer, the previous one will start again.
	"""
	global _running_timer
	
	if use_stack:
		if _running_timer is not None:
			stop(_running_timer, use_stack=False)
			_timer_stack.append(_running_timer)
		start(fn_name, use_stack=False)
		_running_timer = fn_name
	else:
		_start_times[fn_name] = time.perf_counter()

def stop(fn_name=None, use_stack=True):
	"""
	If use_stack is True, this will stop the currently running timer and restore
	the previous timer on the stack if that exists. Note if use_stack is True,
	fn_name will be ignored.

	If use_stack is False, this will just stop timing the timer fn_name.
	"""
	global _running_timer

	if use_stack:
		if _running_timer is not None:
			stop(_running_timer, use_stack=False)
			if len(_timer_stack) > 0:
				_running_timer = _timer_stack.pop()
				start(_running_timer, use_stack=False)
			else:
				_running_timer = None
		else:
			print('Warning: timer stopped with no timer running!')
	else:
		if _start_times[fn_name] > -1:
			_total_times[fn_name] += time.perf_counter() - _start_times[fn_name]
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
	""" Returns the total amount accumulated across all functions in seconds. """ 
	return sum([elapsed_time for name, elapsed_time in _total_times.items() if name not in _disabled_names])


class env():
	"""
	A class that lets you go:
		with timer.env(fn_name):
			# (...)
	That automatically manages a timer start and stop for you.
	"""

	def __init__(self, fn_name, use_stack=True):
		self.fn_name = fn_name
		self.use_stack = use_stack

	def __enter__(self):
		start(self.fn_name, use_stack=self.use_stack)

	def __exit__(self, e, ev, t):
		stop(self.fn_name, use_stack=self.use_stack)

