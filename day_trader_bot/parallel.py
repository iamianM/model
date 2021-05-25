import concurrent
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Executor
from functools import partial

from itertools import chain

import pandas as pd
from tqdm.autonotebook import tqdm


def num_cpus():
    """
    :return: number of CPUs available to the calling process
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def partition(iterable, size=None):
    """
    Splits iterable in equal parts.
    :param iterable: iterable, can be pd.Series
    :param size: size of one partition; None splits by number of cores
    :return: list of iterables comprising the original `iterable`
    """

    size = size or (len(iterable) // (num_cpus() - 1)) or 1
    return [iterable[i:i + size] for i in range(0, len(iterable), size)]


def _apply_func(iterable, func, tqdm_obj=None):
    """
    Applies a function to an iterable immutably.
    """
    if isinstance(iterable, (pd.DataFrame, pd.Series)):
        tqdm.pandas()
        return iterable.progress_apply(func)
    else:
        def update(*args):
            tqdm_obj.update()
            return func(*args)

        return map(update, iterable)


def exec_func(func, iterable, branching_factor=num_cpus() - 1, use_threading=False):
    """
    Execute a function to subsets of iterables leveraging multiple cpus or threading.
    Note: if multiprocessing is used, only a top-level function can be passed as a `func`.

    :param func: function to apply; it is executed asynchronously and several calls to `func` may be made concurrently.
    :param iterable: any iterable
    :param branching_factor: number of cpus or threads to use; None to use all but one core
    :param use_threading: True if threads should be used instead of processes
    :return: list of iterables after applying func to every one of them
    """
    return apply_func_mp(iterable, apply_function=func, apply_function_args=(), branching_factor=branching_factor,
                         use_threading=use_threading)


def apply_func(func, iterable, branching_factor=num_cpus() - 1, use_threading=False, apply_function=_apply_func):
    """
    Apply a function per element in iterable leveraging multiple cpus or threading.
    Note: if multiprocessing is used, only a top-level function can be passed as a `func`.

    :param func: function to apply; it is executed asynchronously and several calls to `func` may be made concurrently.
    :param iterable: an iterable, Pandas dataframe, a list of iterables
    :param branching_factor: number of cpus or threads to use; None to use all but one core
    :param use_threading: True if threads should be used instead of processes
    :param apply_function: function to apply; it is executed asynchronously and several calls to `func` may be made concurrently.
    :return: list of iterables after applying func to every one of them
    """
    return apply_func_mp(iterable, apply_function_args=(func,), branching_factor=branching_factor,
                         apply_function=apply_function, use_threading=use_threading)


def apply_func_mp(iterable, apply_function=_apply_func, apply_function_args=(), branching_factor=num_cpus() - 1,
                  use_threading=False):
    """
    Apply a function to iterables leveraging multiple cpus or threading.
    Note: if multiprocessing is used, only a top-level function can be passed as a `func`.

    :param iterable: an iterable, Pandas dataframe, a list of iterables
    :param apply_function: function to apply; it is executed asynchronously and several calls to `func` may be made concurrently.
    :param apply_function_args: arguments to pass to `apply_function`
    :param branching_factor: number of cpus or threads to use; None to use all but one core
    :param use_threading: True if threads should be used instead of processes
    :return: list of iterables after applying `apply_function` to every one of them
    """
    branching_factor = branching_factor or num_cpus() - 1
    executor_cls = ThreadPoolExecutor if use_threading else ProcessPoolExecutor

    tqdm_obj = tqdm(total=len(iterable))
    iterables = partition(iterable)
    with executor_cls(branching_factor) as e:
        args = [iterables]
        for arg in apply_function_args:
            args.append([arg] * len(iterables))

        if apply_function is _apply_func and use_threading:
            apply_function = partial(apply_function, tqdm_obj=tqdm_obj)

        res = list(chain.from_iterable(e.map(apply_function, *args)))
        tqdm_obj.close()
        return res


def exec_with_progress(executor: Executor, fn, *iterables, **tqdm_kwargs):
    """
    Equivalent to executor.map(fn, *iterables) but displays a tqdm-based progress bar.
    Does not support timeout or chunksize as executor.submit is used internally.
    Results are NOT returned in the same order as the iterables order.
    """
    futures_list = [executor.submit(fn, *iterable) for iterable in iterables]

    for future in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list), **tqdm_kwargs):
        yield future.result()
