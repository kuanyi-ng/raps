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


    def schedule(self, job_list, running, available_nodes, current_time):
        """Schedules jobs from the given job_list directly, modifying available_nodes."""
        
        while job_list:
            job = job_list.pop(0)

            if len(available_nodes) >= job.nodes_required:
                job.scheduled_nodes = available_nodes[:job.nodes_required]
                available_nodes[:] = available_nodes[job.nodes_required:]
                job.start_time = current_time
                job.end_time = current_time + job.wall_time
                job.state = JobState.RUNNING
                running.append(job)
            else:
                job_list.insert(0, job)  # Put job back at the front if it can't be scheduled
                break  # Stop scheduling if no nodes are available


    def schedule2(self, queue, running, available_nodes, current_time, debug=False):
        """Schedules jobs from the queue to available nodes."""
        queue = self.sort_jobs(queue)  # Ensure queue is sorted before scheduling

        while queue:

            # Try scheduling the first job in the queue
            job = queue.pop(0)
            synthetic_bool = len(available_nodes) >= job.nodes_required
            telemetry_bool = job.requested_nodes and set(job.requested_nodes).issubset(set(available_nodes))

            if synthetic_bool or telemetry_bool:

                # Schedule job
                self.assign_nodes_to_job(job, available_nodes, current_time)
                running.append(job)
                #self.history.append(dict(id=job.id, time=current_time, nodes=job.nodes_required, wall_time=job.wall_time))

                if debug:
                    scheduled_nodes = summarize_ranges(job.scheduled_nodes)
                    print(f"t={current_time}: Scheduled job with wall time",
                          f"{job.wall_time} on nodes {scheduled_nodes}")

            else:
                # If the job cannot be scheduled, either try backfilling or requeue it
                if queue and self.policy == PolicyType.BACKFILL:
                    queue.insert(0, job)
                    backfill_job = self.find_backfill_job(queue, len(available_nodes), current_time)
                    if backfill_job:
                        self.assign_nodes_to_job(backfill_job)
                        self.queue.remove(backfill_job)
                        if self.debug:
                            scheduled_nodes = summarize_ranges(backfill_job.scheduled_nodes)
                            print(f"t={self.current_time}: Backfilling job {backfill_job.id} with wall time",
                                  f"{backfill_job.wall_time} on nodes {scheduled_nodes}")
                else:
                    queue.append(job) # Note, this should be fixed. It shouldn't go to the end of the queue.
                break



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
