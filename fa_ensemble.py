import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy
import random
import numpy as np
from torch.utils.data import Subset, Dataset
from interfaces import TrainModelFunction
from dataclasses import dataclass
import os

@dataclass
class FAPartitionInfo:
    idxs: list[int]
    mean: float
    std: float

class FiniteAggregationEnsemble:
    """
    This class represents an ensemble of base models using the Finite Aggregation method.

    Uses code from Wenxiao Wang's work on Finite Aggregation (FA):
    - [FA Hashing](https://github.com/wangwenxiao/FiniteAggregation/blob/main/FiniteAggregation_data_norm_hash.py)  
    - [FA Training](https://github.com/wangwenxiao/FiniteAggregation/blob/main/FiniteAggregation_train_cifar_nin_baseline.py)
    """
    def __init__(self, trainset: Dataset, train_function: TrainModelFunction, channels=3, k:int=50, d:int=1, base_model_dir:str='base_models'):
        self.k = k
        self.d = d
        self.n_subsets = self.k * self.d
        self.base_models = [None]*k
        self.train_function = train_function
        self.trainset = trainset
        self.base_model_dir = base_model_dir

        if not os.path.exists(base_model_dir):
            os.mkdir(base_model_dir)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print("Computing partitions...")
        imgs, labels = zip(*trainset)
        finalimgs = torch.stack(imgs)
        for_sorting = (finalimgs*255).int()
        intmagessum = for_sorting.reshape(for_sorting.shape[0],-1).sum(dim=1) % self.n_subsets

        random.seed(999999999+208)
        shifts = random.sample(range(self.n_subsets), self.d)


        idxgroup = [[] for i in range(self.n_subsets)]
        for i, h in enumerate(intmagessum):
            for shift in shifts:
                idxgroup[(h + shift)%self.n_subsets].append(i)


        idxgroup = [torch.LongTensor(idxs).view(-1, 1) for idxs in idxgroup]

        #force index groups into an order that depends only on image content (not indexes) so that (deterministic) training will not depend initial indices
        idxgroup = list([idxgroup[i][np.lexsort(torch.cat((torch.tensor(labels)[idxgroup[i]].int(),for_sorting[idxgroup[i]].reshape(idxgroup[i].shape[0],-1)),dim=1).numpy().transpose())] for i in range(self.n_subsets) ])

        self.partitions = list([x.squeeze().numpy() for x in idxgroup])
        self.means = torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).mean(dim=1) for i in range(self.n_subsets) ]))
        self.stds =  torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).std(dim=1) for i in range(self.n_subsets) ]))


    def train_base_model(self, partition_number: int):
        if partition_number < 0 or partition_number >= self.k:
            raise ValueError("patition_number must be in the range [0, k)")
        net = self.train_function(
            partition_number,
            Subset(self.trainset, torch.tensor(self.partitions[partition_number])),
            self.means[partition_number],
            self.stds[partition_number]
        )
        print('Saving..')
        torch.save(net.state_dict(), f'{self.base_model_dir}/model_{str(partition_number)}.pth')

    def eval(self):
        pass

    def certify(self):
        pass
