#!/usr/bin/env python
from __future__ import print_function                                                                                                                                       

import numpy as np
import kernel_tuner
import gc
from collections import OrderedDict
import multiprocessing as mp

def initialize():
    with open('convolution.cu', 'r') as f:
        kernel_string = f.read()

    problem_size = (4096, 4096)
    size = np.prod(problem_size)
    input_size = (problem_size[0]+16) * (problem_size[1]+16)

    output = np.zeros(size).astype(np.float32)
    input_matrix = np.random.randn(input_size).astype(np.float32)
    filter_matrix = np.random.randn(17*17).astype(np.float32)

    cmem_args = {'d_filter': np.random.randn(17*17).astype(np.float32)}

    args = [output, input_matrix, filter_matrix]

    return kernel_string, problem_size, cmem_args, args


def main():
    mp.set_start_method('spawn')
    
    ranges = ((1, 9), (0, 6), (0, 3), (0, 3))

    kernel_string, problem_size, cmem_args, args = initialize()
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    log.file = open('convolution.log', 'w')
    log_mean.file = open('convolution_mean.log', 'w')

    print("#name threads it total setup execution speedup", file=log.file, flush=True)
    print("#name threads total setup execution speedup1 speedup2", file=log_mean.file, flush=True)
    for num_params in range(4, 5): #len(ranges)+1):
        tune_params = get_parameters(num_params, ranges)

        print("Running kernel tuner with the following parameter space:")
        print(tune_params)
        print("#", str(tune_params), file=log.file, flush=True)
        print("#", str(tune_params), file=log_mean.file, flush=True)
        # Run without noodles

        print("Without Noodles")
        total, execution, speedup = run_without_noodles(kernel_string, problem_size, args, tune_params, grid_div_y,
                                                        grid_div_x, cmem_args, verbose=False, use_predefined=True)

        mean_total_without = np.mean(total)
        mean_execution_without = np.mean(execution)
        log_mean("without", 1, mean_total_without, 0, mean_execution_without, 1, 1)
        print("Done")

        print("With noodles")
        # Run with noodles with different number of threads
        for num_threads in range(12, 17):
            total_with, setup_with, execution_with, speedup_with = run_with_noodles(kernel_string, problem_size, args,
                                                                                    tune_params, grid_div_y, grid_div_x,
                                                                                    cmem_args, mean_total_without,
                                                                                    num_threads, verbose=False)
            mean_total_with = np.mean(total_with)
            mean_setup_with = np.mean(setup_with)
            mean_execution_with = np.mean(execution_with)
            mean_speedup_with = np.mean(speedup_with)
            mean_speedup = mean_total_without / mean_total_with
            log_mean(num_threads, num_threads, mean_total_with, mean_setup_with, mean_execution_with, mean_speedup_with,
                     mean_speedup)
        print("Done")


def run_without_noodles(kernel_string, problem_size, args, tune_params, grid_div_y, grid_div_x, cmem_args,
                        verbose=False, use_predefined=False):
    total = []
    execution = []
    speedup = []
    if use_predefined:
        execution = total = [1722.18, 1719.15, 1720.81, 1717.88, 1705.14, 1718.1, 1718.2, 1716.38, 1712.28, 1701.67,
                             1711.55, 1721.1, 1705.16, 1714.7, 1714.46, 1711.97, 1710.28, 1716.93, 1719.11, 1722.74,
                             1713.18, 1709.53, 1711.1, 1713.31, 1708.85, 1713.38, 1704.57, 1709.63, 1713.77, 1711.57]
        speedup = np.ones((1, 30), dtype=float)
    else:
        for i in range(0, 30):
            answer, best, timing = kernel_tuner.tune_kernel("convolution_kernel",
                                                            kernel_string, problem_size, args, tune_params,
                                                            grid_div_y=grid_div_y, grid_div_x=grid_div_x,
                                                            verbose=verbose, cmem_args=cmem_args, answer=None,
                                                            num_threads=1, use_noodles=False)
            
            total_single = timing['total']
            execution_single = timing['execution']
            speedup_single = total_single / timing['total']  # this should be 1 always

            total.append(total_single)
            execution.append(execution_single)
            speedup.append(speedup_single)

            log('without', 1, i, timing['total'], timing['setup'], timing['execution'], speedup)

    return total, execution, speedup


def run_with_noodles(kernel_string, problem_size, args, tune_params, grid_div_y, grid_div_x, cmem_args,
                     mean_total_without, num_threads, verbose=False):
        total_with = []
        setup_with = []
        execution_with = []
        speedup_with = []

        for i in range(0, 30):
            #answer, best, timing = kernel_tuner.tune_kernel("convolution_kernel",
            #                                                kernel_string, problem_size, args, tune_params,
            #                                                grid_div_y=grid_div_y, grid_div_x=grid_div_x,
            #                                                verbose=verbose, cmem_args=cmem_args, answer=None,
            #                                                num_threads=num_threads, use_noodles=True)

            q = mp.Queue();
            kwargs = {
                'grid_div_y': grid_div_y,
                'grid_div_x': grid_div_x,
                'verbose': verbose,
                'cmem_args': cmem_args,
                'answer': None,
                'num_threads': num_threads,
                'use_noodles': True,
                'queue': q
            }
            p = mp.Process(target=kernel_tuner.tune_kernel, args=("convolution_kernel",kernel_string, problem_size, args, tune_params), kwargs=kwargs)
            p.start()
            answer, best, timing = q.get()
            p.join()

            total_with.append(timing['total'])
            setup_with.append(timing['setup'])
            execution_with.append(timing['execution'])

            speedup = mean_total_without / timing['total']
            speedup_with.append(speedup)

            log("noodles_"+str(num_threads), num_threads, i, timing['total'], timing['setup'], timing['execution'], speedup)

        return total_with, setup_with, execution_with, speedup_with


def get_parameters(num_params, ranges):
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16 * i for i in range(*ranges[0])]
    if num_params > 1:
        tune_params["block_size_y"] = [2 ** i for i in range(*ranges[1])]
    else:
        tune_params["block_size_y"] = [2]

    if num_params > 2:
        tune_params["tile_size_x"] = [2 ** i for i in range(*ranges[2])]
    else:
        tune_params["tile_size_x"] = [2]

    if num_params > 3:
        tune_params["tile_size_y"] = [2 ** i for i in range(*ranges[3])]
    else:
        tune_params["tile_size_y"] = [2]

    return tune_params


def log(name, threads, iteration, total, setup, execution, speedup):
    # Template for the log file:
    # name threads iteration total setup execution speedup
    print("{} {} {} {} {} {} {}".format(name, threads, iteration, total, setup, execution, speedup),
          file=log.file, flush=True)


def log_mean(name, threads, total, setup, execution, speedup1, speedup2):
    # Template for the mean log file.
    # name threads total setup execution speedup1 speedup2
    print("{} {} {} {} {} {} {}".format(name, threads, total, setup, execution, speedup1, speedup2),
          file=log_mean.file, flush=True)


if __name__ == "__main__":
    main()
