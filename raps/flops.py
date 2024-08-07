import numpy as np
from .utils import linear_to_3d_index
from .config import initialize_config, load_config_variables

load_config_variables(['CPUS_PER_NODE', 
                       'GPUS_PER_NODE',
                       'CPU_PEAK_FLOPS', 
                       'GPU_PEAK_FLOPS', 
                       'CPU_FP_RATIO',
                       'GPU_FP_RATIO',
                       'AVAILABLE_NODES',
                       'DOWN_NODES',
                       'SC_SHAPE'
                      ], globals())


def compute_node_flops(cpu_util, gpu_util):
    return CPU_FP_RATIO * cpu_util * CPU_PEAK_FLOPS + GPU_FP_RATIO * gpu_util * GPU_PEAK_FLOPS


class FLOPSManager():

    def __init__(self, sc_shape):
        self.sc_shape = sc_shape
        self.flop_state = np.zeros(sc_shape)

    def update_flop_state(self, scheduled_nodes, cpu_util, gpu_util):
        node_indices = linear_to_3d_index(scheduled_nodes, self.sc_shape)
        self.flop_state[node_indices] = compute_node_flops(cpu_util, gpu_util)

    def get_rpeak(self):
        node_peak_flops = CPUS_PER_NODE*CPU_PEAK_FLOPS + GPUS_PER_NODE*GPU_PEAK_FLOPS
        system_peak_flops = AVAILABLE_NODES * node_peak_flops
        return system_peak_flops

    def get_system_performance(self):
        return np.sum(self.flop_state)


if __name__ == "__main__":
    fm = FLOPManager(SC_SHAPE)
    print(fm.flop_state.shape)
