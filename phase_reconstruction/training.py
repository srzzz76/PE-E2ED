import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler


def set_seed(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_distributed():
    distributed = all(name in os.environ for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
    if not distributed:
        return 0, 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA GPUs.")
    local_rank, rank, world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, rank, world_size, torch.device(f"cuda:{local_rank}")


def reduce_sum(value):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def current_stage(epoch, stages):
    stages = {int(key): value for key, value in stages.items()}
    for milestone in sorted(stages, reverse=True):
        if epoch >= milestone:
            return stages[milestone], milestone
    raise ValueError("At least one stage must start at epoch 0.")


def perturbation_parameters(epoch, config):
    section = config["early"] if epoch < config["early"]["until_epoch"] else config["late"]
    return tuple(section["amplitude"]), tuple(section["background"]), section["noise_std"]


def circular_mask(batch_size, height, width, device):
    y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device), torch.linspace(-1, 1, width, device=device), indexing="ij")
    return (x.square() + y.square() <= 1).float().view(1, 1, height, width).expand(batch_size, -1, -1, -1)


def make_loaders(dataset, batch_size, workers, rank, world_size, seed):
    train_size = int(0.8 * len(dataset))
    if train_size == 0 or train_size == len(dataset):
        raise ValueError("The dataset must contain at least two samples.")
    train_set, validation_set = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(seed))
    train_sampler = DistributedSampler(train_set, world_size, rank, shuffle=True, drop_last=True) if world_size > 1 else None
    validation_sampler = DistributedSampler(validation_set, world_size, rank, shuffle=False) if world_size > 1 else None
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, shuffle=train_sampler is None, drop_last=True, prefetch_factor=4 if workers > 0 else None, generator=torch.Generator().manual_seed(seed + rank), **common)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, sampler=validation_sampler, shuffle=False, drop_last=False, prefetch_factor=2 if workers > 0 else None, generator=torch.Generator().manual_seed(seed + 10000 + rank), **common)
    return train_loader, validation_loader, train_sampler
