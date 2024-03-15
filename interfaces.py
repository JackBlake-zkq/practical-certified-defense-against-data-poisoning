import abc
from torch.nn import Module
from torch.utils.data import Subset
from torch import Tensor

class TrainModelFunction(abc.ABC):
    """
    Represents a function that trains a model on a subset of the training set.
    """
    @abc.abstractmethod
    def __call__(self, partition_number: int, train_subset: Subset) -> Module:
        pass