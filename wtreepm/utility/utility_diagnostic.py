'''
Utility functions for run-time diagnostics.

Author[s]: Andrew Wetzel.

If in IPython, use timeit to time function and prun to profile function.
'''

# system -----
from __future__ import division
import time


#===================================================================================================
# timing, profiling
#===================================================================================================
def line_profile_func(func, args):
    '''
    Print time to run each line.
    *** There are issues after reloading code.

    Import function (with preceding module name, without trailing parentheses),
    its arguments (all within ()).
    '''
    import line_profiler
    reload(line_profiler)
    ppp = line_profiler.LineProfiler(func)
    ppp.runcall(func, *args)
    ppp.print_stats()


def time_this(func):
    '''
    Decorator that reports the execution time.
    '''
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('# run time = %.3f sec' % (end - start))
        return result
    return wrapper
