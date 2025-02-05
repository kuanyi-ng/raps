from enum import Enum
from ..job import Job, JobState
from ..utils import summarize_ranges


class PolicyType(Enum):
    """Supported scheduling policies."""
    FCFS = 'fcfs'
    BACKFILL = 'backfill'
    PRIORITY = 'priority'
    SJF = 'sjf'


class Scheduler:
    """ Default job scheduler with various scheduling policies. """
    

    def __init__(self, config, policy):
        self.config = config
        self.policy = PolicyType(policy)


    def sort_jobs(self, queue):
        """Sort jobs based on the selected scheduling policy."""
        if self.policy == PolicyType.FCFS or self.policy == PolicyType.BACKFILL:
            return sorted(queue, key=lambda job: job.submit_time)
        elif self.policy == PolicyType.SJF:
            return sorted(queue, key=lambda job: job.wall_time)
        elif self.policy == PolicyType.PRIORITY:
            return sorted(queue, key=lambda job: job.priority, reverse=True)
        else:
            raise ValueError(f"Unknown policy type: {self.policy}")


    def assign_nodes_to_job(self, job, available_nodes, current_time):
        """Assigns nodes to a job and updates available nodes."""
        if len(available_nodes) < job.nodes_required:
            raise ValueError(f"Not enough available nodes to schedule job {job.id}")

        if job.requested_nodes:  # Telemetry replay case
            job.scheduled_nodes = job.requested_nodes
            available_nodes[:] = [n for n in available_nodes if n not in job.scheduled_nodes]
        else:  # Synthetic or reschedule case
            job.scheduled_nodes = available_nodes[:job.nodes_required]
            available_nodes[:] = available_nodes[job.nodes_required:]

        # Set job start and end times
        job.start_time = current_time
        job.end_time = current_time + job.wall_time
        job.state = JobState.RUNNING  # Job is now running


    def schedule(self, queue, running, available_nodes, current_time, debug=False):
        # Sort the queue in place.
        queue[:] = self.sort_jobs(queue)

        # Iterate over a copy of the queue since we might remove items
        for job in queue[:]:
            synthetic_bool = len(available_nodes) >= job.nodes_required
            telemetry_bool = job.requested_nodes and set(job.requested_nodes).issubset(set(available_nodes))

            if synthetic_bool or telemetry_bool:
                self.assign_nodes_to_job(job, available_nodes, current_time)
                running.append(job)
                queue.remove(job)  # Remove the job from the queue
                if debug:
                    scheduled_nodes = summarize_ranges(job.scheduled_nodes)
                    print(f"t={current_time}: Scheduled job {job.id} with wall time {job.wall_time} on nodes {scheduled_nodes}")
            else:
                # Optionally, if you have a BACKFILL policy, attempt backfilling here.
                if self.policy == PolicyType.BACKFILL:
                    # Try to find a backfill candidate from the entire queue.
                    backfill_job = self.find_backfill_job(queue, len(available_nodes), current_time)
                    if backfill_job:
                        self.assign_nodes_to_job(backfill_job, available_nodes, current_time)
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
