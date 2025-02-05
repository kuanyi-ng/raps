from typing import Optional
import dataclasses
import numpy as np
import pandas as pd

from .job import Job, JobState
from .account import Accounts
from .network import network_utilization
from .utils import summarize_ranges, expand_ranges, write_dict_to_file
from .schedulers import load_scheduler


@dataclasses.dataclass
class TickData:
    """ Represents the state output from the simulation each tick """
    current_time: int
    completed: list[Job]
    running: list[Job]
    queue: list[Job]
    down_nodes: list[int]
    power_df: Optional[pd.DataFrame]
    p_flops: Optional[float]
    g_flops_w: Optional[float]
    system_util: float
    fmu_inputs: Optional[dict]
    fmu_outputs: Optional[dict]
    num_active_nodes: int
    num_free_nodes: int


class Engine:
    """Job scheduling simulation engine."""
    def __init__(self, *, power_manager, flops_manager, cooling_model=None, config, **kwargs):
        self.config = config
        self.down_nodes = summarize_ranges(self.config['DOWN_NODES'])
        self.available_nodes = list(set(range(self.config['TOTAL_NODES'])) - set(self.config['DOWN_NODES']))
        self.num_free_nodes = len(self.available_nodes)
        self.num_active_nodes = self.config['TOTAL_NODES'] - self.num_free_nodes - len(self.config['DOWN_NODES'])
        self.running = []
        self.queue = []
        self.accounts = Accounts()
        if 'accounts_json' in kwargs and kwargs['accounts_json']:
            self.accounts.initialize_accounts_from_json(kwargs.get('accounts_json'))
        self.jobs_completed = 0
        self.current_time = 0
        self.cooling_model = cooling_model
        self.sys_power = 0
        self.power_manager = power_manager
        self.flops_manager = flops_manager
        self.debug = kwargs.get('debug')
        self.output = kwargs.get('output')
        self.replay = kwargs.get('replay')
        self.sys_util_history = []
        
        # Get scheduler type from command-line args or default
        scheduler_type = kwargs.get('scheduler', 'default')
        self.scheduler = load_scheduler(scheduler_type)(config=self.config, policy=kwargs.get('policy'))
        print(f"Using scheduler: {scheduler_type}")


    def add_job(self, job):
        self.queue.append(job)
        self.queue = self.scheduler.sort_jobs(self.queue)


    def tick(self):
        """Simulate a timestep."""
        completed_jobs = [job for job in self.running if job.end_time is not None and job.end_time <= self.current_time]
        completed_job_stats = []
        
        # Simulate node failure
        newly_downed_nodes = self.node_failure(self.config['MTBF'])

        # Update active/free nodes
        self.num_free_nodes = len(self.available_nodes)
        self.num_active_nodes = self.config['TOTAL_NODES'] - self.num_free_nodes - len(expand_ranges(self.down_nodes))

        # Update running time for all running jobs
        for job in self.running:
            if job.end_time == self.current_time:
                job.state = JobState.COMPLETED

            if job.state == JobState.RUNNING:
                job.running_time = self.current_time - job.start_time
                time_quanta_index = (self.current_time - job.start_time) // self.config['TRACE_QUANTA']
                cpu_util = self.get_utilization(job.cpu_trace, time_quanta_index)
                gpu_util = self.get_utilization(job.gpu_trace, time_quanta_index)
                net_util = 0

                if len(job.ntx_trace) and len(job.nrx_trace):
                    net_tx = self.get_utilization(job.ntx_trace, time_quanta_index)
                    net_rx = self.get_utilization(job.nrx_trace, time_quanta_index)
                    net_util = network_utilization(net_tx, net_rx)

                self.flops_manager.update_flop_state(job.scheduled_nodes, cpu_util, gpu_util)
                job.power = self.power_manager.update_power_state(job.scheduled_nodes, cpu_util, gpu_util, net_util)

                if job.running_time % self.config['TRACE_QUANTA'] == 0:
                    job.power_history.append(job.power)

        for job in completed_jobs:
            self.running.remove(job)
            self.jobs_completed += 1
            job_stats = job.statistics()
            self.accounts.update_account_statistics(job_stats)

        # Ask scheduler to schedule any jobs waiting in queue
        self.scheduler.schedule(self.queue, self.running, self.available_nodes, self.current_time)

        # Update the power array UI component
        rack_power, rect_losses = self.power_manager.compute_rack_power()
        sivoc_losses = self.power_manager.compute_sivoc_losses()
        rack_loss = rect_losses + sivoc_losses

        # Update system utilization
        system_util = self.num_active_nodes / self.config['AVAILABLE_NODES'] * 100
        self.sys_util_history.append((self.current_time, system_util))

        # Render the updated layout
        power_df = None #self.power_manager.get_power_df() if self.power_manager else pd.DataFrame()
        cooling_inputs, cooling_outputs = None, None

        # Update power history every 15s
        if self.current_time % self.config['POWER_UPDATE_FREQ'] == 0:
            total_power_kw = sum(row[-1] for row in rack_power) + self.config['NUM_CDUS'] * self.config['POWER_CDU'] / 1000.0
            total_loss_kw = sum(row[-1] for row in rack_loss)
            self.power_manager.history.append((self.current_time, total_power_kw))
            self.sys_power = total_power_kw
            self.power_manager.loss_history.append((self.current_time, total_loss_kw))
            output_df = self.power_manager.get_power_df(rack_power, rack_loss)
            pflops = self.flops_manager.get_system_performance() / 1E15
            gflop_per_watt = pflops * 1E6 / (total_power_kw * 1000)
        else:
            pflops, gflop_per_watt = None, None

        if self.current_time % self.config['POWER_UPDATE_FREQ'] == 0:
            if self.cooling_model:
                # Power for NUM_CDUS (25 for Frontier)
                cdu_power = rack_power.T[-1] * 1000
                runtime_values = self.cooling_model.generate_runtime_values(cdu_power, self)

                # FMU inputs are N powers and the wetbulb temp
                fmu_inputs = self.cooling_model.generate_fmu_inputs(runtime_values,
                                uncertainties=self.power_manager.uncertainties)
                cooling_inputs, cooling_outputs = (
                    self.cooling_model.step(self.current_time, fmu_inputs, self.config['POWER_UPDATE_FREQ'])
                )

                # Get a dataframe of the power data
                power_df = self.power_manager.get_power_df(rack_power, rack_loss)
            else:
                # Get a dataframe of the power data
                power_df = self.power_manager.get_power_df(rack_power, rack_loss)

        tick_data = TickData(
            current_time=self.current_time,
            completed=completed_jobs,
            running=self.running,
            queue=self.queue,
            down_nodes=expand_ranges(self.down_nodes),
            power_df=power_df,
            p_flops=pflops,
            g_flops_w=gflop_per_watt,
            system_util=self.num_active_nodes / self.config['AVAILABLE_NODES'] * 100,
            fmu_inputs=cooling_inputs,
            fmu_outputs=cooling_outputs,
            num_active_nodes=self.num_active_nodes,
            num_free_nodes=self.num_free_nodes,
        )

        self.current_time += 1
        return tick_data



    def get_utilization(self, trace, time_quanta_index):
        """Retrieve utilization value for a given trace at a specific time quanta index."""
        if isinstance(trace, (list, np.ndarray)):
            return trace[time_quanta_index]
        elif isinstance(trace, (int, float)):
            return float(trace)
        else:
            raise TypeError(f"Invalid type for utilization: {type(trace)}.")


    def run_simulation(self, jobs, timesteps):
        """ Generator that yields after each simulation tick """
        last_submit_time = 0
        self.timesteps = timesteps

        for job_info in jobs:
            job = Job(job_info, self.current_time)
            self.add_job(job)

        for timestep in range(timesteps):
            while self.current_time >= last_submit_time and jobs:

                job = jobs.pop(0)
                job = Job(job_info, self.current_time)
                self.scheduler.schedule([job], self.running, self.available_nodes, self.current_time)

                if jobs:
                    last_submit_time = job.submit_time
                else:
                    last_submit_time = float('inf')  # Avoid infinite loop

            yield self.tick()

            # Stop the simulation if no more jobs are running or in the queue
            if not self.queue and not self.running and not self.replay:
                print(f"[DEBUG] {self.config['system_name']} - Stopping simulation at time {self.current_time}")
                break
            if self.debug and timestep % self.config['UI_UPDATE_FREQ'] == 0:
                    print(".", end="", flush=True)


    def get_stats(self):
        """ Return output statistics """
        sum_values = lambda values: sum(x[1] for x in values) if values else 0
        min_value = lambda values: min(x[1] for x in values) if values else 0
        max_value = lambda values: max(x[1] for x in values) if values else 0
        num_samples = len(self.power_manager.history) if self.power_manager else 0

        throughput = self.jobs_completed / self.timesteps * 3600 if self.timesteps else 0  # Jobs per hour
        average_power_mw = sum_values(self.power_manager.history) / num_samples / 1000 if num_samples else 0
        average_loss_mw = sum_values(self.power_manager.loss_history) / num_samples / 1000 if num_samples else 0
        min_loss_mw = min_value(self.power_manager.loss_history) / 1000 if num_samples else 0
        max_loss_mw = max_value(self.power_manager.loss_history) / 1000 if num_samples else 0

        loss_fraction = average_loss_mw / average_power_mw if average_power_mw else 0
        efficiency = 1 - loss_fraction if loss_fraction else 0
        total_energy_consumed = average_power_mw * self.timesteps / 3600 if self.timesteps else 0  # MW-hr
        emissions = total_energy_consumed * 852.3 / 2204.6 / efficiency if efficiency else 0
        total_cost = total_energy_consumed * 1000 * self.config.get('POWER_COST', 0)  # Total cost in dollars

        stats = {
            'num_samples': num_samples,
            'jobs completed': self.jobs_completed,
            'throughput': f'{throughput:.2f} jobs/hour',
            'jobs still running': [job.id for job in self.running],
            'jobs still in queue': [job.id for job in self.queue],
            'average power': f'{average_power_mw:.2f} MW',
            'min loss': f'{min_loss_mw:.2f} MW',
            'average loss': f'{average_loss_mw:.2f} MW',
            'max loss': f'{max_loss_mw:.2f} MW',
            'system power efficiency': f'{efficiency * 100:.2f}%',
            'total energy consumed': f'{total_energy_consumed:.2f} MW-hr',
            'carbon emissions': f'{emissions:.2f} metric tons CO2',
            'total cost': f'${total_cost:.2f}'
        }

        return stats


    def node_failure(self, mtbf):
        """Simulate node failure using Weibull distribution."""
        from scipy.stats import weibull_min
        shape_parameter = 1.5
        scale_parameter = mtbf * 3600  # Convert to seconds

        down_nodes = expand_ranges(self.down_nodes)
        all_nodes = np.setdiff1d(np.arange(self.config['TOTAL_NODES']), np.array(down_nodes, dtype=int))

        random_values = weibull_min.rvs(shape_parameter, scale=scale_parameter, size=all_nodes.size)
        failure_threshold = 0.1
        failed_nodes_mask = random_values < failure_threshold
        newly_downed_nodes = all_nodes[failed_nodes_mask]

        for node_index in newly_downed_nodes:
            if node_index in self.available_nodes:
                self.available_nodes.remove(node_index)
            self.down_nodes.append(str(node_index))
            self.power_manager.set_idle(node_index)

        return newly_downed_nodes.tolist()
