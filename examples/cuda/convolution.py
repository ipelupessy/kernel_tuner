#!/usr/bin/env python
import numpy as np
import kernel_tuner
import gc
from collections import OrderedDict

with open('convolution.cu', 'r') as f:
    kernel_string = f.read()

problem_size = (4096, 4096)
size = np.prod(problem_size)
input_size = (problem_size[0]+16) * (problem_size[1]+16)

output = np.zeros(size).astype(np.float32)
input = np.random.randn(input_size).astype(np.float32)
filter = np.random.randn(17*17).astype(np.float32)

print("Input size: ", input_size)
print("Filter size: ", 17*17)

cmem_args = {'d_filter': np.random.randn(17*17).astype(np.float32)}

args = [output, input, filter]

ranges = ((1, 9), (0, 6), (0, 3), (0, 3))

log = open('convolution.log', 'w')
mean_log = open('convolution_mean.log', 'w')
print("#name threads it total setup execution speedup", file=log, flush=True)
print("#name threads it total setup execution speedup1 speedup2", file=mean_log, flush=True)
for num_params in range(4, 5): #len(ranges)+1):
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16*i for i in range(1, 9)]
    if num_params > 1:
        tune_params["block_size_y"] = [2**i for i in range(6)]
    else:
        tune_params["block_size_y"] = [2]

    if num_params > 2:
        tune_params["tile_size_x"] = [2**i for i in range(3)]
    else:
        tune_params["tile_size_x"] = [2]

    if num_params > 3:
        tune_params["tile_size_y"] = [2**i for i in range(3)]
    else:
        tune_params["tile_size_y"] = [2]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    print("Running kernel tuner with the following parameter space:")
    print(tune_params)
    print("#", str(tune_params), file=log, flush=True)
    print("#", str(tune_params), file=mean_log, flush=True)
    # Run without noodles
    total = []
    execution = []
    print("Normal")
    #for i in range(0, 30):
    #    answer, best, timing = kernel_tuner.tune_kernel("convolution_kernel",
    #            kernel_string, problem_size, args, tune_params, 
    #            grid_div_y=grid_div_y, grid_div_x=grid_div_x, verbose=False,
    #            cmem_args=cmem_args, answer=None, num_threads=1,
    #            use_noodles=False)
    #    gc.collect()
    #    total_without = timing['total']
    #    total.append(total_without)
    #    execution_without = timing['execution']
    #    execution.append(execution_without)
    #    speedup = total_without / timing['total'] # this should be 1 always
    #    print("without {} {} {} {} {} {}".format(1, i, timing['total'], timing['setup'], timing['execution'], speedup), file=log, flush=True)

    execution = total = [1722.18, 1719.15, 1720.81, 1717.88, 1705.14, 1718.1, 1718.2, 1716.38, 1712.28, 1701.67, 1711.55, 1721.1, 1705.16, 1714.7, 1714.46, 1711.97, 1710.28, 1716.93, 1719.11, 1722.74, 1713.18, 1709.53, 1711.1, 1713.31, 1708.85, 1713.38, 1704.57, 1709.63, 1713.77, 1711.57]

    mean_total_without = np.mean(total)
    mean_execution_without = np.mean(execution)
    print("without {} {} {} {} {} {}".format(1, mean_total_without, 0,
                                          mean_execution_without,
                                          1, 1), file=mean_log, flush=True)

    print("With noodles")
    # Run with noodles with different number of threads
    for num_threads in range(8,9):
        total_with = []
        setup_with = []
        execution_with = []
        speedup_with = []

        for i in range(0, 30):
            answer, best, timing = kernel_tuner.tune_kernel("convolution_kernel",
                   kernel_string, problem_size, args, tune_params, 
                   grid_div_y=grid_div_y, grid_div_x=grid_div_x, verbose=False,
                   cmem_args=cmem_args, answer=None, num_threads=num_threads,
                   use_noodles=True)
            gc.collect()

            total_with.append(timing['total'])
            setup_with.append(timing['setup'])
            execution_with.append(timing['execution'])

            speedup = mean_total_without / timing['total']
            speedup_with.append(speedup)
            print("threads_{} {} {} {} {} {} {}".format(num_threads, num_threads, i, timing['total'],
                    timing['setup'], timing['execution'], speedup), file=log,
                    flush=True)

        mean_total_with = np.mean(total_with)
        mean_setup_with = np.mean(setup_with)
        mean_execution_with = np.mean(execution_with)
        mean_speedup_with = np.mean(speedup_with)
        mean_speedup = mean_total_without / mean_total_with
        print("threads_{} {} {} {} {} {} {}".format(num_threads, num_threads, mean_total_with, mean_setup_with, mean_execution_with, mean_speedup_with, mean_speedup), file=mean_log, flush=True)
