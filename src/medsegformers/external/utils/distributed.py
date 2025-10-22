import torch.distributed as dist
import torch

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()

def all_reduce_tensor(t: torch.Tensor, op=dist.ReduceOp.SUM, async_op: bool = False):
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=op, async_op=async_op)
    return t
