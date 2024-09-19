

def find_backfill_job(queue, num_free_nodes, current_time):
    """ This implementation is based on pseudocode from Leonenkov and Zhumatiy.
        "Introducing new backfill-based scheduler for slurm resource manager."
        Procedia computer science 66 (2015): 661-669. """

    # Compute shadow time
    first_job = queue[0]

    for job in queue: job.end_time = current_time + job.wall_time

    # Sort jobs according to their termination time (end_time)
    sorted_queue = sorted(queue, key=lambda job: job.end_time)

    # Loop over the list and collect nodes until the number of available nodes
    # is sufficient for the first job in the queue
    sum_nodes = 0
    shadow_time = None
    for job in sorted_queue:
        sum_nodes += job.nodes_required
        if sum_nodes >= first_job.nodes_required:
            shadow_time = current_time + job.wall_time
            num_extra_nodes = sum_nodes - job.nodes_required
            break

    # Find backfill job
    backfill_job = None
    for job in queue:
        # condition1 checks that the job ends before first_job starts
        condition1 = job.nodes_required <= num_free_nodes \
                     and current_time + job.wall_time < shadow_time
        # condition2 checks that the job does not interfere with first_job
        condition2 = job.nodes_required <= min(num_free_nodes, num_extra_nodes)

        if condition1 or condition2:
            backfill_job = job
            break

    return backfill_job
