import numpy as np
from .utils import linear_to_3d_index
from .config import initialize_config, load_config_variables

load_config_variables(['CPUS_PER_NODE', 
                       'GPUS_PER_NODE',
                       'CPU_PEAK_FLOPS', 
                       'GPU_PEAK_FLOPS', 
                       'CPU_FP_RATIO',
                       'GPU_FP_RATIO',
                       'TOTAL_NODES',
                       'DOWN_NODES',
                       'SC_SHAPE'
                      ], globals())

node_peak_flops = CPUS_PER_NODE*CPU_PEAK_FLOPS + GPUS_PER_NODE*GPU_PEAK_FLOPS
print(f"Node peak FLOPS: {node_peak_flops:.2e}")

num_nodes = TOTAL_NODES - len(DOWN_NODES)
system_peak_flops = num_nodes * node_peak_flops
print(f"System peak FLOPS: {system_peak_flops:.2e}")


def compute_node_flops(cpu_util, gpu_util):
    return CPU_FP_RATIO * cpu_util * CPU_PEAK_FLOPS + GPU_FP_RATIO * gpu_util * GPU_PEAK_FLOPS


class FLOPSManager():

    def __init__(self, sc_shape):
        self.sc_shape = sc_shape
        self.flop_state = np.zeros(sc_shape)

    def update_flop_state(self, scheduled_nodes, cpu_util, gpu_util):
        node_indices = linear_to_3d_index(scheduled_nodes, self.sc_shape)
        self.flop_state[node_indices] = compute_node_flops(cpu_util, gpu_util)

    def get_system_performance(self):
        return np.sum(self.flop_state)

if __name__ == "__main__":
    fm = FLOPManager(SC_SHAPE)
    print(fm.flop_state.shape)