# from time import time
import time
import torch
import functools


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class TimerError(Exception):
    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer:
    """A flexible Timer class.

    :Example:

    >>> import time
    >>> with Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with Timer(print_tmpl='it takes {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, start=True, print_tmpl=None, cuda_sync=False, round=None, unit="s"):
        assert unit in ["s", "ms"]
        assert round is None or isinstance(round, int)
        self._is_running = False
        self.round = round
        self.unit = unit
        self.print_tmpl = print_tmpl if print_tmpl else "{:.3f}"
        self.time = time_sync if cuda_sync else time.time
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self, reset=False):
        """Start the timer."""
        if not self._is_running:
            self._t_start = self.time()
            self._is_running = True
        self._t_start = self.time() if reset else self._t_start
        self._t_last = self.time()

    def since_start(self):
        """Total time since the timer is started.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError("timer is not running")
        self._t_last = self.time()
        dur = self._t_last - self._t_start
        if self.unit == "ms":
            dur = dur * 1000
        if self.round is not None:
            dur = round(dur, self.round)
        return dur

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError("timer is not running")
        dur = self.time() - self._t_last
        if self.unit == "ms":
            dur = dur * 1000
        if self.round is not None:
            dur = round(dur, self.round)
        self._t_last = self.time()
        return dur


def timer(cuda_sync=False):
    def out_func(old_func):
        time = Timer(start=False, cuda_sync=cuda_sync)

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            time.start()
            output = old_func(*args, **kwargs)
            print(f"timer: using {time.since_start()} s")
            return output

        return new_func

    return out_func


if __name__ == "__main__":
    timer = Timer(unit="ms", round=2)
    time.sleep(0.5)
    print(timer.since_start())
    time.sleep(0.5)
    print(timer.since_last_check())
    print(timer.since_start())
