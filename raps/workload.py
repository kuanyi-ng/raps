"""
Module for generating workload traces and jobs.

This module provides functionality for generating random workload traces and
jobs for simulation and testing purposes.

Attributes
----------
TRACE_QUANTA : int
    The time interval in seconds for tracing workload utilization.
MAX_NODES_PER_JOB : int
    The maximum number of nodes required for a job.
JOB_NAMES : list
    List of possible job names for random job generation.
CPUS_PER_NODE : int
    Number of CPUs per node.
GPUS_PER_NODE : int
    Number of GPUs per node.
MAX_WALL_TIME : int
    Maximum wall time for a job in seconds.
MIN_WALL_TIME : int
    Minimum wall time for a job in seconds.
JOB_END_PROBS : list
    List of probabilities for different job end states.

"""

import math
import random
import numpy as np

from .config import load_config_variables

load_config_variables([
    'TRACE_QUANTA', 'MAX_NODES_PER_JOB', 'JOB_NAMES', 'CPUS_PER_NODE',\
    'GPUS_PER_NODE', 'MAX_WALL_TIME', 'MIN_WALL_TIME', 'JOB_END_PROBS',\
    'AVAILABLE_NODES' ], globals())

JOB_NAMES = ["LAMMPS", "GROMACS", "VASP", "Quantum ESPRESSO", "NAMD",\
             "OpenFOAM", "WRF", "AMBER", "CP2K", "nek5000", "CHARMM",\
             "ABINIT", "Cactus", "Charm++", "NWChem", "STAR-CCM+",\
             "Gaussian", "ANSYS", "COMSOL", "PLUMED", "nekrs",\
             "TensorFlow", "PyTorch", "BLAST", "Spark", "GAMESS",\
             "ORCA", "Simulink", "MOOSE", "ELK"]

MAX_PRIORITY = 500000

from .utils import truncated_normalvariate, determine_state, next_arrival, job_dict


class Workload(object):
    """ This class is responsible for generating random workload traces and jobs. """

    def compute_traces(self, cpu_util: float, gpu_util: float, wall_time: int) -> tuple[np.ndarray, np.ndarray]:
        """ Compute CPU and GPU traces based on mean CPU & GPU utilizations and wall time. """
        cpu_trace = cpu_util * np.ones(int(wall_time) // TRACE_QUANTA)
        gpu_trace = gpu_util * np.ones(int(wall_time) // TRACE_QUANTA)
        return (cpu_trace, gpu_trace)


    def generate_random_jobs(self, num_jobs: int) -> list[list[any]]:
        """ Generate random jobs with specified number of jobs. """
        jobs = []
        for job_index in range(num_jobs):
            nodes_required = random.randint(1, MAX_NODES_PER_JOB)
            name = random.choice(JOB_NAMES)
            cpu_util = random.random()*CPUS_PER_NODE
            gpu_util = random.random()*GPUS_PER_NODE
            mu = (MAX_WALL_TIME + MIN_WALL_TIME) / 2
            sigma = (MAX_WALL_TIME - MIN_WALL_TIME) / 6
            wall_time = truncated_normalvariate(mu, sigma, MIN_WALL_TIME, MAX_WALL_TIME) // 3600 * 3600
            end_state = determine_state(JOB_END_PROBS)
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time)
            priority = random.randint(0, MAX_PRIORITY)

            # Jobs arrive according to Poisson process
            time_to_next_job = next_arrival()

            jobs.append(job_dict(nodes_required, name, cpu_trace, gpu_trace, \
                        wall_time, end_state, None, time_to_next_job, None, priority))

        return jobs


    def random(self, **kwargs):
        """ Generate random workload """
        num_jobs = kwargs.get('num_jobs', 0)
        return self.generate_random_jobs(num_jobs=num_jobs)


    def peak(self, **kwargs):
        """Peak power test"""
        jobs = self.generate_random_jobs(num_jobs=0)
        cpu_util, gpu_util = CPUS_PER_NODE, GPUS_PER_NODE
        cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 10800)
        job_info = job_dict(AVAILABLE_NODES, "Max Test", cpu_trace, gpu_trace, \
                    len(gpu_trace)*TRACE_QUANTA, 'COMPLETED', None, 100, None)
        jobs.insert(0, job_info)
        return jobs


    def idle(self, **kwargs):
        """Idle power test"""
        jobs = self.generate_random_jobs(num_jobs=0)
        cpu_util, gpu_util = 0, 0
        cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 43200)
        job_info = job_dict(AVAILABLE_NODES, "Idle Test", cpu_trace, gpu_trace, \
                    len(gpu_trace)*TRACE_QUANTA, 'COMPLETED', None, 0, None)
        jobs.insert(0, job_info)
        return jobs


    def benchmark(self, **kwargs):
        """Benchmark tests"""

        jobs = self.generate_random_jobs(num_jobs=0)

        # Max test
        cpu_util, gpu_util = 1, 4
        cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 10800)
        job_info = jobs_dict(AVAILABLE_NODES, "Max Test", cpu_trace, gpu_trace,
                    len(gpu_trace)*TRACE_QUANTA, 'COMPLETED', None, 100, None)
        jobs.insert(0, job_info)
        # OpenMxP run
        cpu_util, gpu_util = 0, 4
        cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600)
        job_info = jobs_dict(AVAILABLE_NODES, "OpenMxP", cpu_trace, gpu_trace,
                    len(gpu_trace)*TRACE_QUANTA, 'COMPLETED', None, 300, None)
        jobs.insert(0, job_info)
        # HPL run
        cpu_util, gpu_util = 0.33, 0.79 * 4 # based on 24-01-18 run
        cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600)
        job_info = jobs_dict(AVAILABLE_NODES, "HPL", cpu_trace, gpu_trace,
                    len(gpu_trace)*TRACE_QUANTA, 'COMPLETED', None, 200, None)
        jobs.insert(0, job_info)
        # Idle test
        cpu_util, gpu_util = 0, 0
        cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600)
        job_info = jobs_dict(AVAILABLE_NODES, "Idle Test", cpu_trace, gpu_trace,
                    len(gpu_trace)*TRACE_QUANTA, 'COMPLETED', None, 0, None)
        jobs.insert(0, job_info)

        return jobs
    
