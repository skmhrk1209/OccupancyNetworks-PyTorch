import torch
from torch import utils
from torch import distributed
import math


class DistributedSampler(utils.data.Sampler):
    r""" Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, shuffle=False, world_size=None, rank=None, last_epoch=-1):

        if world_size is None:
            if not distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = distributed.get_world_size()
        if rank is None:
            if not distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = distributed.get_rank()

        self.dataset = dataset
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.epoch = last_epoch
        self.num_samples = int(math.ceil(len(self.dataset) / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):

        self.epoch += 1

        if self.shuffle:
            # deterministically shuffle based on epoch
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):

        return self.num_samples


class BatchSampler(utils.data.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last, milestones, gamma, last_epoch=-1):

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.milestones = milestones
        self.gamma = gamma
        self.epoch = last_epoch

    def __iter__(self):

        self.epoch += 1

        if self.epoch in self.milestones:
            self.batch_size *= self.gamma

        batch = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):

        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
