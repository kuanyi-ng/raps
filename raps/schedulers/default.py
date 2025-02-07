from enum import Enum
#from ..job import Job, JobState  # Unused
from ..utils import summarize_ranges

from ..workload import MAX_PRIORITY

class PolicyType(Enum):
    """Supported scheduling policies."""
    FCFS = 'fcfs'
    BACKFILL = 'backfill'
    PRIORITY = 'priority'
    SJF = 'sjf'


class Scheduler:
    """ Default job scheduler with various scheduling policies. """

    def __init__(self, config, policy, resource_manager=None):
        self.config = config
        self.policy = PolicyType(policy)
        if resource_manager is None:
            raise ValueError("Scheduler requires a ResourceManager instance")
        self.resource_manager = resource_manager
        self.debug = False

    def sort_jobs(self, queue, accounts=None):
        """Sort jobs based on the selected scheduling policy."""
        if self.policy == PolicyType.FCFS or self.policy == PolicyType.BACKFILL:
            return sorted(queue, key=lambda job: job.submit_time)
        elif self.policy == PolicyType.SJF:
            return sorted(queue, key=lambda job: job.wall_time)
        elif self.policy == PolicyType.PRIORITY:
            return sorted(queue, key=lambda job: job.priority, reverse=True)
        else:
            raise ValueError(f"Unknown policy type: {self.policy}")

    def schedule(self, queue, running, current_time, sorted=False, debug=False):
        # Sort the queue in place.
        if not sorted:
            queue[:] = self.sort_jobs(queue)

        # Iterate over a copy of the queue since we might remove items
        for job in queue[:]:

            # For synthetic jobs the number of requested nodes is given.
            # Make sure the available nodes count meets job.nodes_required.
            synthetic_bool = len(self.resource_manager.available_nodes) >= job.nodes_required

            # For telemetry replay jobs a list of requested nodes is provided.
            # Make sure the requested nodes are available.
            telemetry_bool = False
            if job.requested_nodes:
                telemetry_bool = set(job.requested_nodes).issubset(set(self.resource_manager.available_nodes))

            if synthetic_bool or telemetry_bool:
                self.resource_manager.assign_nodes_to_job(job, current_time)
                running.append(job)
                queue.remove(job)
                if debug:
                    scheduled_nodes = summarize_ranges(job.scheduled_nodes)
                    print(f"t={current_time}: Scheduled job {job.id} with wall time {job.wall_time} on nodes {scheduled_nodes}")
            else:
                if self.policy == PolicyType.BACKFILL:
                    # Try to find a backfill candidate from the entire queue.
                    backfill_job = self.find_backfill_job(queue, len(self.resource_manager.available_nodes), current_time)
                    if backfill_job:
                        self.assign_nodes_to_job(backfill_job, self.resource_manager.available_nodes, current_time)
                        running.append(backfill_job)
                        queue.remove(backfill_job)
                        if debug:
                            scheduled_nodes = summarize_ranges(backfill_job.scheduled_nodes)
                            print(f"t={current_time}: Backfilling job {backfill_job.id} with wall time {backfill_job.wall_time} on nodes {scheduled_nodes}")

    def find_backfill_job(self, queue, num_free_nodes, current_time):
        """Finds a backfill job based on available nodes and estimated completion times.

        Based on pseudocode from Leonenkov and Zhumatiy, 'Introducing new backfill-based
        scheduler for slurm resource manager.' Procedia computer science 66 (2015): 661-669.
        """

        if not queue:
            return None

        first_job = queue[0]

        for job in queue:
            job.end_time = current_time + job.wall_time  # Estimate end time

        # Sort jobs according to their termination time (end_time)
        sorted_queue = sorted(queue, key=lambda job: job.end_time)

        # Compute shadow time by accumulating nodes
        sum_nodes = 0
        shadow_time = None
        num_extra_nodes = 0

        for job in sorted_queue:
            sum_nodes += job.nodes_required
            if sum_nodes >= first_job.nodes_required:
                shadow_time = current_time + job.wall_time
                num_extra_nodes = sum_nodes - job.nodes_required
                break

        # Find backfill job
        for job in queue:
            condition1 = job.nodes_required <= num_free_nodes and current_time + job.wall_time < shadow_time
            condition2 = job.nodes_required <= min(num_free_nodes, num_extra_nodes)

            if condition1 or condition2:
                return job

        return None
