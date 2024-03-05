import abc
import torch

class IModelGenerator(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> torch.nn.Module:
        pass

class IDatasetGenerator(abc.ABC):
    @abc.abstractmethod
    def trainset(self) -> torch.utils.data.Dataset:
        pass
    @abc.abstractmethod
    def testset(self) -> torch.utils.data.Dataset:
        pass