import gc

import torch
from accelerate.utils import release_memory


def flush_gpu_memory_allocations():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    release_memory()
