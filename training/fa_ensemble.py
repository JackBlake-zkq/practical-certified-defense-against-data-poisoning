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
    def __init__(self, trainset: Dataset, testset: Dataset, train_function: TrainModelFunction, num_classes: int, state_dir:str, k:int=50, d:int=1):
        self.k = k
        self.d = d
        self.n_subsets = self.k * self.d
        self.base_models = [None]*k
        self.train_function = train_function
        self.trainset = trainset
        self.testset = testset
        self.state_dir = "ensembles/" + state_dir
        self.num_classes = num_classes
        self.partitions = None

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
        

    def get_base_model_predictions(self, test=True):
        """
        Runs each base model on the provided testset and saves results to {state_dir}/{train/test}_predictions.pth.
        It is not necessary to directly call this method, as it is called by eval() if predictions have not been computed yet.
        All base models must have already been trained using train_base_model before calling this method.
        """
        ds_str = ('test' if test else 'train') + "set"
        dataset = self.testset if test else self.trainset
        path = f'{self.state_dir}/{ds_str}_predictions.pth'
        if os.path.exists(path):
            print(f"{ds_str} predictions already computed, using those...")
            return torch.load(path)

        print(f"Generating base model predictions on {ds_str}...")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logits_by_base_model = torch.zeros(len(dataset), self.n_subsets, self.num_classes).to(device)
        logits_by_class = torch.zeros(len(dataset), self.num_classes, self.n_subsets).to(device)
        softmaxes_by_base_model = torch.zeros(len(dataset), self.n_subsets, self.num_classes).to(device)
        softmaxes_by_class = torch.zeros(len(dataset), self.num_classes, self.n_subsets).to(device)
        labels = torch.zeros(len(dataset)).type(torch.int).to(device)
        firstit = True
        for i in tqdm(range(self.n_subsets)):
            seed = i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            net  = torch.load(f'{self.state_dir}/base_models/model_{str(i)}.pkl')
            net = net.to(device)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=2000, shuffle=False, num_workers=1)
            
            net.eval()
            batch_offset = 0
            with torch.no_grad():
                for (inputs, targets) in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    out = net(inputs)
                    softmaxes = nn.Softmax(dim=1)(out)
                    logits_by_base_model[batch_offset:inputs.size(0)+batch_offset,i,:] = out
                    softmaxes_by_base_model[batch_offset:inputs.size(0)+batch_offset,i,:] = softmaxes
                    for c in range(self.num_classes):
                        logits_by_class[batch_offset:inputs.size(0)+batch_offset, c, i] = out[:,c]
                        softmaxes_by_class[batch_offset:inputs.size(0)+batch_offset, c, i] = softmaxes[:,c]
                    if firstit:
                        labels[batch_offset:batch_offset+inputs.size(0)] = targets
                    batch_offset += inputs.size(0)
            firstit = False

        print(f"Saving Predictions to {path}")
        output = {
            'labels': labels, 
            'logits_by_base_model': logits_by_base_model, 
            'logits_by_class': logits_by_class,
            'softmaxes_by_base_model': softmaxes_by_base_model,
            'softmaxes_by_class': softmaxes_by_class
            }
        torch.save(output, path)
        return output


    def eval(self, mode:str):
        """
        Generates accuracy and robustness info about the ensemble
        """

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        random.seed(999999999+208)
        shifts = random.sample(range(self.n_subsets), self.d)

        filein = self.get_base_model_predictions(test=True)
        labels = filein['labels']
        logits_by_base_model = filein['logits_by_base_model']
        logits_by_class = filein['logits_by_class']
        softmaxes_by_base_model = filein['softmaxes_by_base_model']
        softmaxes_by_class = filein['softmaxes_by_class']

        num_classes = self.num_classes
        n_sample = labels.shape[0]
        max_classes = logits_by_base_model.max(2).indices

        certs_path = f'{self.state_dir}/{mode}_certs.pth'

        if os.path.exists(certs_path):
            print("Certificates already computed, using those...")
            certs = torch.load(certs_path)
        elif mode == 'label_voting':
            print("Computing Certificates...")
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
            torch.save(certs, certs_path)
        elif mode == 'logit_median':
            logits,_ = logits_by_class.to(device).sort(dim=2)
            certs = torch.Tensor(n_sample)
            for i in tqdm(range(n_sample)):
                certs[i] = 9999999999999999
                mid = self.n_subsets//2
                predicted_class = torch.argmax(logits[i,:,mid])
                correct_class = labels[i]
                if predicted_class != correct_class: certs[i] = -1
                else:
                    for c in range(num_classes):
                        if c == correct_class: continue
                        shift = self.d
                        r = 0
                        while shift < mid and logits[i][correct_class][mid - shift] > logits[i][c][mid + shift]:
                            shift += self.d
                            r += 1
                        if r < certs[i]: certs[i] = r
            torch.save(certs, certs_path)
        elif mode == 'softmax_median':
            softmaxes,_ = softmaxes_by_class.to(device).sort(dim=2)
            certs = torch.Tensor(n_sample)
            for i in tqdm(range(n_sample)):
                certs[i] = 9999999999999999
                mid = self.n_subsets//2
                predicted_class = torch.argmax(softmaxes[i,:,mid])
                correct_class = labels[i]
                if predicted_class != correct_class: certs[i] = -1
                else:
                    for c in range(num_classes):
                        if c == correct_class: continue
                        shift = self.d
                        r = 0
                        while shift < mid and softmaxes[i][correct_class][mid - shift] > softmaxes[i][c][mid + shift]:
                            shift += self.d
                            r += 1
                        if r < certs[i]: certs[i] = r
            torch.save(certs, certs_path)
        
        base_acc = 100 *  (max_classes == labels.unsqueeze(1)).sum().item() / (max_classes.shape[0] * max_classes.shape[1])
        print('Base classifier accuracy: ' + str(base_acc))

        a = certs.cpu().sort()[0].numpy()
        accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/n_sample
        print('Ensembe Accuracy: ' + str(accs[0] * 100.) + '%')
        print('Certified Radius (for at least half of inputs): ' + str(sum(accs >= 0.5)))
        

    def distill(self, student: Module, mode: str, seed: int=0, lr=0.1, epochs=1):
        """
        Distills the ensemble into a single model and saves to {state_dir}/student.pkl
        """
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        preds = self.get_base_model_predictions(test=False)
        logits_by_base_model = preds['logits_by_base_model'].to(device)
        logits_by_class = preds['logits_by_class'].to(device)
        softmaxes_by_base_model = preds['softmaxes_by_base_model'].to(device)
        softmaxes_by_class = preds['softmaxes_by_class'].to(device)

        if mode == 'label_voting':
            num_classes = self.num_classes
            max_classes = logits_by_base_model.max(2).indices
            predictions = torch.zeros(max_classes.shape[0],num_classes).to(device)
            for i in range(max_classes.shape[1]):
                predictions[(torch.arange(max_classes.shape[0]).to(device),max_classes[:,i].to(device))] += 1
            ensemble_outputs = torch.argmax(predictions, dim=1)
        elif mode == 'logit_median':
            mid = self.n_subsets//2
            logits, _ = logits_by_class.to(device).sort(dim=2)
            ensemble_outputs = logits[:,:,mid]
        elif mode == 'softmax_median':
            mid = self.n_subsets//2
            softmaxes, _ = softmaxes_by_class.to(device).sort(dim=2)
            ensemble_outputs = softmaxes[:,:,mid]
        else:
            raise ValueError("Invalid distillation mode")

        student_path = f'{self.state_dir}/student_{mode}.pkl'
        if os.path.exists(student_path):
            print(f"Student for {mode} distillation mode already trained")
            student = torch.load(student_path)
        else:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=False, num_workers=1)
            student.to(device)

            criterion = torch.nn.MSELoss() if mode == 'logit_median' else nn.CrossEntropyLoss()

            optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

            # Training
            student.train()
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                batch_offset = 0
                correct = 0
                total = 0
                for (inputs, targets) in tqdm(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    teachers_outputs = ensemble_outputs[batch_offset:batch_offset + inputs.size(0)]
                    student_outputs = student(inputs)
                    correct += torch.argmax(student_outputs, dim=1).eq(targets).sum().item()
                    total += inputs.shape[0]
                    loss = criterion(student_outputs, teachers_outputs)
                    loss.backward()
                    optimizer.step()
                    batch_offset += inputs.size(0)
                print(f"Trainset Accuracy: {str(correct/total*100)}%")
            print(f"Finished training student, saving to { student_path}")
            torch.save(student, student_path)
        print("Evaluating Student")
        student.to(device)
        student.eval()
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=1)
        correct = 0
        total = 0
        for (inputs, targets) in tqdm(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        acc = 100.*correct/total
        print(f'Accuracy for student: {str(acc)}%')

        