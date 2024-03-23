import torch
import random
import numpy as np
from torch.utils.data import Subset, Dataset
from interfaces import TrainModelFunction
import os
from torch.nn import Module
from tqdm import tqdm
from torch import nn, optim

class FiniteAggregationEnsemble:
    """
    This class represents an ensemble of base models using the Finite Aggregation method. Saves info about the ensemble
    into a folder called "ensembles/{state_dir}" where state_dir is fa_{k}_{d} by default.

    Uses code from [Wenxiao Wang's work on Finite Aggregation (FA)](https://github.com/wangwenxiao/FiniteAggregation/tree/main):
    @inproceedings{FiniteAggregation,
        title={Improved certified defenses against data poisoning with (deterministic) finite aggregation},
        author={Wang, Wenxiao and Levine, Alexander J and Feizi, Soheil},
        booktitle={International Conference on Machine Learning},
        pages={22769--22783},
        year={2022},
        organization={PMLR}
    }
    """
    def __init__(self, trainset: Dataset, testset: Dataset, train_function: TrainModelFunction, num_classes: int, channels=3, k:int=50, d:int=1, state_dir:str=None):
        self.k = k
        self.d = d
        self.n_subsets = self.k * self.d
        self.base_models = [None]*k
        self.train_function = train_function
        self.trainset = trainset
        self.testset = testset
        if state_dir == None:
            self.state_dir = f"ensembles/fa_k={str(k)}_d={str(d)}"
        else:
            self.state_dir = "ensembles/" + state_dir
        self.channels = channels
        self.num_classes = num_classes
        self.partitions = None

        size = list(self.trainset[0][0].size())
        while len(size) < 4:
            size.insert(0, 1)
        size = tuple(size)
        self.sample_input = torch.randn(size)

        if not os.path.exists(self.state_dir):
            os.mkdir(self.state_dir)
            os.mkdir(f'{self.state_dir}/base_models')
        else:
            if os.path.exists(f"{self.state_dir}/partition_info.pth"):
                partitions_file = torch.load(f"{self.state_dir}/partition_info.pth")
                self.partitions = partitions_file['idx']
            if not os.path.exists(f'{self.state_dir}/base_models'):
                os.mkdir(f'{self.state_dir}/base_models')

    def compute_partitions(self):
        """
        Computes the partitions of the dataset that each base model will be trained on, and the mean and standard deviation of each partition. Saves them to 
        """
        if self.partitions != None:
            print("Partitions already computed")
            return

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print("Computing partitions...")
        imgs, labels = zip(*self.trainset)

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
        out = {'idx': self.partitions }
        print(f"Finished computing partitions, saving to {self.state_dir}/partition_info.pth")
        torch.save(out, f"{self.state_dir}/partition_info.pth")
        print("Partitions saved")

    def train_base_model(self, partition_number: int):
        """
        Trains the base model for the specified partition number by calling the train_function
        that the class was instantiated with. Saves the base model to {state_dir}/base_models with
        file name model_{partition_number}.pkl
        """
        if os.path.exists(f'{self.state_dir}/base_models/model_{str(partition_number)}.pkl'):
            print(f"Base model {partition_number} already exists")
            return
        if self.partitions == None:
            print("Partitions not computed yet, computing now...")
            self.compute_partitions()
        if partition_number < 0 or partition_number >= self.n_subsets:
            raise ValueError("patition_number must be in the range [0, k)")
        print(f'Training Base model {partition_number}..')
        net = self.train_function(
            partition_number,
            Subset(self.trainset, torch.tensor(self.partitions[partition_number]))
        )
        print(f'Saving Base model {partition_number}..')
        torch.save(net, f'{self.state_dir}/base_models/model_{str(partition_number)}.pkl')
        print(f'Base model {partition_number} saved')
        

    def eval(self):
        """
        Evaluates the ensemble on the provided testset and saves results to {state_dir}/eval.pth. 
        All base models must have already been trained using train_base_model before calling this method.
        """
        if os.path.exists(f'{self.state_dir}/eval.pth'):
            print("Predictions already computed")
            return
        
        for i in range(self.n_subsets):
            if not os.path.exists(f'{self.state_dir}/base_models/model_{str(i)}.pkl'):
                print(f"Base model {i} not found. Aborting...")
                exit(1)

        print("Generating Predictions...")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        predictions = torch.zeros(len(self.testset), self.n_subsets, self.num_classes).to(device)
        labels = torch.zeros(len(self.testset)).type(torch.int).to(device)
        firstit = True
        for i in range(self.n_subsets):
            seed = i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            net  = torch.load(f'{self.state_dir}/base_models/model_{str(i)}.pkl')
            net = net.to(device)

            testloader = torch.utils.data.DataLoader(self.testset, batch_size=2000, shuffle=False, num_workers=1)
            
            net.eval()
            batch_offset = 0
            with torch.no_grad():
                for (inputs, targets) in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    out = net(inputs)
                    predictions[batch_offset:inputs.size(0)+batch_offset,i,:] = out
                    if firstit:
                        labels[batch_offset:batch_offset+inputs.size(0)] = targets
                    batch_offset += inputs.size(0)
            firstit = False

        
        print(f"Saving Predictions to {self.state_dir}/eval.pth")
        torch.save({'labels': labels, 'scores': predictions},f'{self.state_dir}/eval.pth')


    def certify(self):
        """
        Generates the roubstness cetificate for the ensemble, and accuracy  
        """

        if os.path.exists(f'{self.state_dir}/radii.pth'):
            print("Certificates already computed")
            return

        if not os.path.exists(f'{self.state_dir}/eval.pth'):
            self.eval()

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        random.seed(999999999+208)
        shifts = random.sample(range(self.n_subsets), self.d)

        filein = torch.load(f'{self.state_dir}/eval.pth', map_location=torch.device(device))
        labels = filein['labels']
        scores = filein['scores']

        num_classes = self.num_classes
        max_classes = scores.max(2).indices
        predictions = torch.zeros(max_classes.shape[0],num_classes).to(device)
        for i in range(max_classes.shape[1]):
            predictions[(torch.arange(max_classes.shape[0]).to(device),max_classes[:,i].to(device))] += 1
        predinctionsnp = predictions.cpu().numpy()
        idxsort = np.argsort(-predinctionsnp,axis=1,kind='stable')
        valsort = -np.sort(-predinctionsnp,axis=1,kind='stable')
        val =  valsort[:,0]
        idx = idxsort[:,0]
        valsecond =  valsort[:,1]
        idxsecond =  idxsort[:,1] 

        #original code from DPA
        #diffs = ((val - valsecond - (idxsecond <= idx))/2).astype(int)
        #certs = torch.tensor(diffs).cuda()
        #torchidx = torch.tensor(idx).cuda()
        #certs[torchidx != labels] = -1



        n_sample = labels.size(0)
        certs = torch.LongTensor(n_sample)

        #prepared for indexing
        shifted = [
            [(h + shift)%self.n_subsets for shift in shifts] for h in range(self.n_subsets)
        ]
        shifted = torch.LongTensor(shifted)

        for i in tqdm(range(n_sample)):
            if idx[i] != labels[i]:
                certs[i] = -1
                continue
            
            certs[i] = self.n_subsets #init value
            label = int(labels[i])

            #max_classes corresponding to diff h
            max_classes_given_h = max_classes[i][shifted.view(-1)].view(-1, self.d)

            for c in range(num_classes): #compute min radius respect to all classes
                if c != label:
                    diff = predictions[i][labels[i]] - predictions[i][c] - (1 if c < label else 0)
                    
                    deltas = (1 + (max_classes_given_h == label).long() - (max_classes_given_h == c).long()).sum(dim=1)
                    deltas = deltas.sort(descending=True)[0]
                    
                    radius = 0
                    while diff - deltas[radius] >= 0:
                        diff -= deltas[radius].item()
                        radius += 1
                    certs[i] = min(certs[i], radius)


        base_acc = 100 *  (max_classes == labels.unsqueeze(1)).sum().item() / (max_classes.shape[0] * max_classes.shape[1])
        print('Base classifier accuracy: ' + str(base_acc))
        torch.save(certs,f'{self.state_dir}/radii.pth')
        a = certs.cpu().sort()[0].numpy()
        accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/predictions.shape[0]
        print('Smoothed classifier accuracy: ' + str(accs[0] * 100.) + '%')
        print('Robustness certificate: ' + str(sum(accs >= .5)))


    def distill(self, student: Module, seed: int=0):
        """
        Distills the ensemble into a single model and saves to {state_dir}/student.pkl
        Code adapted from [Konrad Zuchniak's work on Multi-teacher distillation (with keras)](https://github.com/ZuchniakK/MTKD)
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        alpha = 0.1
        epochs = 50
        curr_lr = 0.1
        epochs = 10
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=True, num_workers=1)
        student.to(device)

        student_criterion = nn.CrossEntropyLoss()
        distillation_criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(student.parameters(), lr=curr_lr, momentum=0.9, weight_decay=0.0005, nesterov= True)

        teachers = []
        for i in range(self.n_subsets):
            teacher = torch.load(f'{self.state_dir}/base_models/model_{str(i)}.pkl')
            teacher.to(device)
            teacher.eval()
            teachers.append(teacher)

        # Training
        student.train()
        for epoch in tqdm(range(epochs)):
            for (inputs, targets) in tqdm(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                teacher_votes = [{} for _ in range(len(inputs))]
                most_voted = [(-1, -1) for i in range(len(inputs))]
                for teacher in teachers:
                    with torch.no_grad():
                        for i, output in enumerate(teacher(inputs).max(1)[1]):
                            if teacher_votes[i].get(output, None) is None:
                                teacher_votes[i][output] = 1
                            else: 
                                teacher_votes[i][output] += 1
                            if teacher_votes[i][output] > most_voted[i][1]:
                                most_voted[i] = (output, teacher_votes[i][output])
                ensemble_outputs = torch.tensor([t[0] for t in most_voted]).to(device)
                student_outputs = student(inputs)
                loss = alpha * student_criterion(student_outputs, targets) + (1-alpha) * distillation_criterion(student_outputs, ensemble_outputs)
                loss.backward()
                optimizer.step()
            if (epoch in [60,120,160]):
                curr_lr = curr_lr * 0.2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_lr
        print(f"Finished training student, saving to {self.state_dir}/student.pkl")
        torch.save(student, f'{self.state_dir}/student.pkl')
        print("Evaluating Student")
        student.eval()
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=1)
        correct = 0
        total = 0
        for (inputs, targets) in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        acc = 100.*correct/total
        print(f'Accuracy for student: {str(acc)}%')
