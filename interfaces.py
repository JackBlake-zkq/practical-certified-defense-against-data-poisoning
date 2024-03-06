import abc
import torch
from torch.utils.data import Subset

class TrainModelFunction(abc.ABC):
    @abc.abstractmethod
    def __call__(self, partition_number: int, train_subset: Subset, mean: float, std: float) -> torch.nn.Module:
        pass