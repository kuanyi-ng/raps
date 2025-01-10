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

from .job import job_dict

JOB_NAMES = ["LAMMPS", "GROMACS", "VASP", "Quantum ESPRESSO", "NAMD",\
             "OpenFOAM", "WRF", "AMBER", "CP2K", "nek5000", "CHARMM",\
             "ABINIT", "Cactus", "Charm++", "NWChem", "STAR-CCM+",\
             "Gaussian", "ANSYS", "COMSOL", "PLUMED", "nekrs",\
             "TensorFlow", "PyTorch", "BLAST", "Spark", "GAMESS",\
             "ORCA", "Simulink", "MOOSE", "ELK"]

MAX_PRIORITY = 500000

from .utils import truncated_normalvariate, determine_state, next_arrival


class Workload:
    def __init__(self, *configs):
        """ Initialize Workload with multiple configurations.  """
        self.partitions = [config['system_name'] for config in configs]
        self.config_map = {config['system_name']: config for config in configs}

    def compute_traces(self, cpu_util: float, gpu_util: float, wall_time: int, trace_quanta: int) -> tuple[np.ndarray, np.ndarray]:
        """ Compute CPU and GPU traces based on mean CPU & GPU utilizations and wall time. """
        cpu_trace = cpu_util * np.ones(int(wall_time) // trace_quanta)
        gpu_trace = gpu_util * np.ones(int(wall_time) // trace_quanta)
        return (cpu_trace, gpu_trace)

    def generate_random_jobs(self, num_jobs: int) -> list[list[any]]:
        """ Generate random jobs with specified number of jobs. """
        jobs = []
        for job_index in range(num_jobs):
            # Randomly select a partition
            partition = random.choice(self.partitions)
            # Get the corresponding config for the selected partition
            config = self.config_map[partition]

            nodes_required = random.randint(1, config['MAX_NODES_PER_JOB'])
            name = random.choice(JOB_NAMES)
            cpu_util = random.random() * config['CPUS_PER_NODE']
            gpu_util = random.random() * config['GPUS_PER_NODE']
            mu = (config['MAX_WALL_TIME'] + config['MIN_WALL_TIME']) / 2
            sigma = (config['MAX_WALL_TIME'] - config['MIN_WALL_TIME']) / 6
            wall_time = truncated_normalvariate(mu, sigma, config['MIN_WALL_TIME'], config['MAX_WALL_TIME']) // 3600 * 3600
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = [], []

            # Jobs arrive according to Poisson process
            time_to_next_job = next_arrival(1 / config['JOB_ARRIVAL_TIME'])

            jobs.append(job_dict(nodes_required, name, cpu_trace, gpu_trace, net_tx, net_rx, \
                        wall_time, end_state, None, time_to_next_job, None, priority, partition))

        return jobs

    def random(self, **kwargs):
        """ Generate random workload """
        num_jobs = kwargs.get('num_jobs', 0)
        return self.generate_random_jobs(num_jobs=num_jobs)

    def peak(self, **kwargs):
        """Peak power test for multiple partitions"""
        jobs = []

        # Iterate through each partition and get its configuration
        for partition in self.partitions:
            # Fetch the config for the current partition
            config = self.config_map[partition]

            # Generate traces based on partition-specific configuration
            cpu_util = config['CPUS_PER_NODE']
            gpu_util = config['GPUS_PER_NODE']
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 10800, config['TRACE_QUANTA'])
            net_tx, net_rx = [], []

            # Create job info for this partition
            job_info = job_dict(
                config['AVAILABLE_NODES'],       # Nodes required
                f"Max Test {partition}",         # Name with partition label
                cpu_trace,                       # CPU trace
                gpu_trace,                       # GPU trace
                net_tx,                          # Network transmit trace
                net_rx,                          # Network receive trace
                len(gpu_trace) * config['TRACE_QUANTA'],  # Wall time
                'COMPLETED',                     # End state
                None,                            # Scheduled nodes
                0,                               # Time to next job
                None,                            # Job ID
                100,                             # Priority
                partition                        # Partition name
            )
            print(job_info)
            jobs.append(job_info)  # Add job to the list

        return jobs

    def idle(self, **kwargs):
        """Idle power test for multiple partitions"""

        # List to hold jobs for all partitions
        jobs = []

        # Iterate through each partition and get its configuration
        for partition in self.partitions:
            # Fetch partition-specific configuration
            config = self.config_map[partition]

            # Generate traces based on partition-specific configuration
            cpu_util, gpu_util = 0, 0  # Idle test has zero utilization
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 43200, config['TRACE_QUANTA'])  # 12 hours
            net_tx, net_rx = [], []

            # Create job info for this partition
            job_info = job_dict(
                config['AVAILABLE_NODES'],       # Nodes required
                f"Idle Test {partition}",        # Name with partition label
                cpu_trace,                       # CPU trace
                gpu_trace,                       # GPU trace
                net_tx,                          # Network transmit trace
                net_rx,                          # Network receive trace
                len(gpu_trace) * config['TRACE_QUANTA'],  # Wall time
                'COMPLETED',                     # End state
                None,                            # Scheduled nodes
                0,                               # Time to next job
                None,                            # Job ID
                100,                             # Priority
                partition                        # Partition name
            )
            jobs.append(job_info)  # Add job to the list

        return jobs

    def benchmark(self, **kwargs):
        """Benchmark tests for multiple partitions"""

        # List to hold jobs for all partitions
        jobs = []

        # Iterate through each partition and its config
        for partition in self.partitions:
            # Fetch partition-specific configuration
            config = self.config_map[partition]
            net_tx, net_rx = [], []

            # Max test
            cpu_util, gpu_util = 1, 4
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 10800, config['TRACE_QUANTA'])
            job_info = job_dict(
                config['AVAILABLE_NODES'], 
                f"Max Test {partition}", 
                cpu_trace, gpu_trace, net_tx, net_rx,
                len(gpu_trace) * config['TRACE_QUANTA'], 'COMPLETED', None, 100, None, 0, partition
            )
            jobs.append(job_info)

            # OpenMxP run
            cpu_util, gpu_util = 0, 4
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600, config['TRACE_QUANTA'])
            job_info = job_dict(
                config['AVAILABLE_NODES'], 
                f"OpenMxP {partition}", 
                cpu_trace, gpu_trace, net_tx, net_rx,
                len(gpu_trace) * config['TRACE_QUANTA'], 'COMPLETED', None, 300, None, 0, partition
            )
            jobs.append(job_info)

            # HPL run
            cpu_util, gpu_util = 0.33, 0.79 * 4  # based on 24-01-18 run
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600, config['TRACE_QUANTA'])
            job_info = job_dict(
                config['AVAILABLE_NODES'], 
                f"HPL {partition}", 
                cpu_trace, gpu_trace, net_tx, net_rx,
                len(gpu_trace) * config['TRACE_QUANTA'], 'COMPLETED', None, 200, None, 0, partition
            )
            jobs.append(job_info)

            # Idle test
            cpu_util, gpu_util = 0, 0
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600, config['TRACE_QUANTA'])
            job_info = job_dict(
                config['AVAILABLE_NODES'], 
                f"Idle Test {partition}", 
                cpu_trace, gpu_trace, net_tx, net_rx,
                len(gpu_trace) * config['TRACE_QUANTA'], 'COMPLETED', None, 0, None, 0, partition
            )
            jobs.append(job_info)

        return jobs

