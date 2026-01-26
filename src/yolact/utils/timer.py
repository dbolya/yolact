import time
from collections import defaultdict

_total_times = defaultdict(lambda: 0)
_start_times = defaultdict(lambda: -1)
_disabled_names = set()
_timer_stack = []
_running_timer = None
_disable_all = False


def disable_all():
    global _disable_all
    _disable_all = True


def enable_all():
    global _disable_all
    _disable_all = False


def disable(fn_name):
    """Disable `fn_name` from being tracked/printed."""
    _disabled_names.add(fn_name)


def enable(fn_name):
    """Enable a name previously disabled by `disable`."""
    _disabled_names.remove(fn_name)


def reset():
    """Reset all timer state. Call this at the start of an iteration."""
    global _running_timer
    _total_times.clear()
    _start_times.clear()
    _timer_stack.clear()
    _running_timer = None


def start(fn_name, use_stack=True):
    """
    Start timing `fn_name`.

    If `use_stack` is True, only one timer can be active at a time. When you stop
    the current timer, the previous timer (if any) will resume.
    """
    global _running_timer, _disable_all

    if _disable_all:
        return

    if use_stack:
        if _running_timer is not None:
            stop(_running_timer, use_stack=False)
            _timer_stack.append(_running_timer)
        start(fn_name, use_stack=False)
        _running_timer = fn_name
        return

    _start_times[fn_name] = time.perf_counter()


def stop(fn_name=None, use_stack=True):
    """
    Stop timing.

    If `use_stack` is True, this stops the currently running timer and resumes
    the previous timer on the stack (if any). In this mode, `fn_name` is ignored.

    If `use_stack` is False, this stops timing `fn_name`.
    """
    global _running_timer, _disable_all

    if _disable_all:
        return

    if use_stack:
        if _running_timer is None:
            print("Warning: timer stopped with no timer running!")
            return

        stop(_running_timer, use_stack=False)
        if _timer_stack:
            _running_timer = _timer_stack.pop()
            start(_running_timer, use_stack=False)
        else:
            _running_timer = None
        return

    if _start_times[fn_name] > -1:
        _total_times[fn_name] += time.perf_counter() - _start_times[fn_name]
    else:
        print(f"Warning: timer for {fn_name} stopped before starting!")


def print_stats():
    """Print the current timing information into a table."""
    print()

    all_fn_names = [name for name in _total_times.keys() if name not in _disabled_names]

    max_name_width = max([len(name) for name in all_fn_names] + [4])
    if max_name_width % 2 == 1:
        max_name_width += 1
    format_str = f" {{:>{max_name_width}}} | {{:>10.4f}} "

    header = f" {{:^{max_name_width}}} | {{:^10}} ".format("Name", "Time (ms)")
    print(header)

    sep_idx = header.find("|")
    sep_text = ("-" * sep_idx) + "+" + "-" * (len(header) - sep_idx - 1)
    print(sep_text)

    for name in all_fn_names:
        print(format_str.format(name, _total_times[name] * 1000))

    print(sep_text)
    print(format_str.format("Total", total_time() * 1000))
    print()


def total_time():
    """Return the total amount accumulated across all functions in seconds."""
    return sum(elapsed for name, elapsed in _total_times.items() if name not in _disabled_names)


class Env:
    """
    Context manager helper:

        with timer.env("fn"):
            ...
    """

    def __init__(self, fn_name, use_stack=True):
        self.fn_name = fn_name
        self.use_stack = use_stack

    def __enter__(self):
        start(self.fn_name, use_stack=self.use_stack)

    def __exit__(self, e, ev, t):
        stop(self.fn_name, use_stack=self.use_stack)


# Backwards-compatible alias (the old name was `env`).
env = Env
