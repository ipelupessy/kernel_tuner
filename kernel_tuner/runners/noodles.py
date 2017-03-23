#module private functions
import subprocess

from collections import OrderedDict

from noodles import schedule, schedule_hint, has_scheduled_methods, gather, run_logging, run_parallel, run_process, serial, Storable
from noodles.run.runners import run_parallel_timing
from noodles.display import NCDisplay
from noodles.interface import AnnotatedValue

from kernel_tuner.util import detect_language, get_device_interface, compile_and_benchmark

class RefCopy:
        def __init__(self, obj):
            self.obj = obj

        def __deepcopy__(self, _):
            return self.obj


class NoodlesRunner:
    def run(self, kernel_name, original_kernel, problem_size, arguments,
            tune_params, parameter_space, grid_div_x, grid_div_y,
            answer, atol, verbose,
            lang, device, platform, cmem_args):
        """ Iterate through the entire parameter space using a multiple Python processes

        :param kernel_name: The name of the kernel in the code.
        :type kernel_name: string

        :param original_kernel: The CUDA, OpenCL, or C kernel code as a string.
        :type original_kernel: string

        :param problem_size: See kernel_tuner.tune_kernel
        :type problem_size: tuple(int or string, int or string)

        :param arguments: A list of kernel arguments, use numpy arrays for
                arrays, use numpy.int32 or numpy.float32 for scalars.
        :type arguments: list

        :param tune_params: See kernel_tuner.tune_kernel
        :type tune_params: dict( string : [int, int, ...] )

        :param parameter_space: A list of lists that contains the entire parameter space
                to be searched. Each list in the list represents a single combination
                of parameters, order is imported and it determined by the order in tune_params.
        :type parameter_space: list( list() )

        :param grid_div_x: See kernel_tuner.tune_kernel
        :type grid_div_x: list

        :param grid_div_y: See kernel_tuner.tune_kernel
        :type grid_div_y: list

        :param answer: See kernel_tuner.tune_kernel
        :type answer: list

        :param atol: See kernel_tuner.tune_kernel
        :type atol: float

        :param verbose: See kernel_tuner.tune_kernel
        :type verbose: boolean

        :param lang: See kernel_tuner.tune_kernel
        :type lang: string

        :param device: See kernel_tuner.tune_kernel
        :type device: int

        :param platform: See kernel_tuner.tune_kernel
        :type device: int

        :param cmem_args: See kernel_tuner.tune_kernel
        :type cmem_args: dict(string: numpy object)

        :returns: A dictionary of all executed kernel configurations and their
            execution times.
        :rtype: dict( string, float )
        """
        workflow = self._parameter_sweep(lang, device, arguments, verbose, RefCopy(cmem_args), RefCopy(answer), 
                                    RefCopy(tune_params), RefCopy(parameter_space), problem_size,
                                    grid_div_y, grid_div_x, original_kernel, kernel_name, atol, platform)

        if verbose:
            with NCDisplay(self.error_filter) as display:
                answer = run_logging(workflow, self.max_threads, display)
            #answer = run_parallel(workflow, self.max_threads)
        else:
            myId = uuid.uuid4().hex
            answer = run_parallel_timing(workflow, self.max_threads, "noodles"+myId+".json")

        if answer is None:
            print("Tuning did not return any results, did an error occur?")
            return None

        print(answer)
        answer = filter(None, answer)
        return dict(answer)


    def __init__(self, max_threads=1):
        self._max_threads = max_threads


    @property
    def max_threads(self):
        return self._max_threads


    @max_threads.setter
    def set_max_threads(self, max_threads):
        self._max_threads = max_threads


    def my_registry(self):
        return serial.pickle() + serial.base()


    def error_filter(self, type, value=None, tb=None):
        if type is subprocess.CalledProcessError:
            return value.stderr
        elif "cuCtxSynchronize" in str(value):
            return xcptn
        else:
            return None


    @schedule_hint(display="Batching ... ",
                ignore_error=True,
                confirm=True)
    def _parameter_sweep(self, lang, device, arguments, verbose, cmem_args, answer, tune_params, parameter_space,
                         problem_size, grid_div_y, grid_div_x, original_kernel, kernel_name, atol, platform):
        results = []
        for element in parameter_space:
            params = OrderedDict(zip(tune_params.keys(), element))

            instance_string = "_".join([str(i) for i in params.values()])

            time = self.run_single(lang, device, kernel_name, original_kernel, params,
                            problem_size, grid_div_y, grid_div_x,
                            cmem_args, answer, atol, instance_string, verbose, platform, arguments)

            if time is not None:
                results.append(time)

        return gather(*results)


    @schedule_hint(display="Testing {instance_string} ... ",
                ignore_error=True,
                confirm=True)
    def run_single(self, lang, device, kernel_name, original_kernel, params, problem_size, grid_div_y, grid_div_x, cmem_args, answer, atol, instance_string, verbose, platform, arguments):
        #detect language and create device function interface
        lang = detect_language(lang, original_kernel)
        dev = get_device_interface(lang, device, platform)

        #move data to the GPU
        gpu_args = dev.ready_argument_list(arguments)
        try:
            time = compile_and_benchmark(dev, gpu_args, kernel_name, original_kernel, params, problem_size, grid_div_y, grid_div_x, cmem_args, answer, atol, instance_string, False)
            if time is not None:
                result = (instance_string, time)
                return AnnotatedValue(result, None)
            else:
                return None
        except Exception as e:
            return AnnotatedValue(None, str(e))
