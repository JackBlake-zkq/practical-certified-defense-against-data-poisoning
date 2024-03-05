import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy
import random

class FiniteAggregationEnsembleTrainer:
    """
    This class trains an ensemble of base models using the Finite Aggregation method.

    Uses code from Wenxiao Wang's work on Finite Aggregation (FA):
    - [FA Hashing](https://github.com/wangwenxiao/FiniteAggregation/blob/main/FiniteAggregation_data_norm_hash.py)  
    - [FA Training](https://github.com/wangwenxiao/FiniteAggregation/blob/main/FiniteAggregation_train_cifar_nin_baseline.py)
    """
    def __init__(self, trainset:torch.utils.data.Dataset, testset:torch.utils.data.Dataset, model_repo:str, model:str, k:int=50, d:int=1, zero_seed:bool=False, output_dir="FA_ensemble"):
        print(architecture)
        self.k = k
        self.d = d
        self.zero_seed = zero_seed
        self.n_subsets = self.k * self.d
        self.architecture = architecture
        self.trainset = trainset
        self.testset = testset
        self.output_dir = output_dir

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        imgs, labels = zip(*self.trainset)
        finalimgs = torch.stack(list(map((lambda x: torchvision.transforms.ToTensor()(x)), list(imgs))))
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


    def train_base_models(self, start:int=0, range:int=250):

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        for part in range(start, start + range):
            seed = part
            if (self.zero_seed):
                seed = 0
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            curr_lr = 0.1
            print('\Partition: %d' % part)
            part_indices = torch.tensor(self.partitions[part])

            train_subset = torch.utils.data.Subset(self.trainset ,part_indices)
            train_subset.dataset.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.means[part], self.stds[part])
            ])

            # transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(self.means[part], self.stds[part])
            # ])

            nomtestloader = torch.utils.data.DataLoader(self.testset, batch_size=128, shuffle=True, num_workers=1)
            trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=1)
            
            net = self.architecture()

            net = net.to(device)

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.SGD(net.parameters(), lr=curr_lr, momentum=0.9, weight_decay=0.0005, nesterov= True)

        # Training
            net.train()
            for epoch in range(200):
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                if (epoch in [60,120,160]):
                    curr_lr = curr_lr * 0.2
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = curr_lr

            net.eval()

            (inputs, targets)  = next(iter(nomtestloader)) #Just use one test batch
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                total = targets.size(0)
            acc = 100.*correct/total
            print(f'Accuracy for base model {part}: {str(acc)}%')
            # Save checkpoint.
            print('Saving..')
            torch.save(net, self.output_dir + f'/base_model_{str(part)}.pkl')
