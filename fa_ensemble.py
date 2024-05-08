import torch
import random
import numpy as np
from torch.utils.data import Subset, Dataset
from interfaces import TrainModelFunction
import os
from torch.nn import Module
from tqdm import tqdm
from torch import nn, optim
import numpy.typing as npt
from threading import Lock

class FiniteAggregationEnsemble:
    """
    This class represents an ensemble of base models using the Finite Aggregation method. Saves info about the ensemble
    into state_dir. The ensemble can be trained, evaluated, and distilled into a single model. The base models are trained
    trained on partitions of the dataset, and using the train_function that is passed in.

    Uses code from [Wenxiao Wang's work on Finite Aggregation (FA)](https://github.com/wangwenxiao/FiniteAggregation/tree/main)

    Introduces two novel aggregation methods:

    1. Median of Logits - To generate the prediction for a sample: for each class, takes the median logit value across all base models e.g. for 3 base models with logits [1, 2, 3], [2, 3, 4], [3, 4, 5], the median logit for class 0 would be 2, for class 1 would be 3, and for class 2 would be 4, so the ensemble outputs [2, 3, 4] as its logits.
    2. Median of Softmaxes - Same thing, but take the softmax before taking the median. Notably, the output is not a softmax,
    so we should take the softmax of the output to get the final softmax prediction.
    """
    def __init__(self, state_dir:str, trainset: Dataset, testset: Dataset, num_classes: int, k:int, d:int=1):
        self.k = k
        self.d = d
        self.n_subsets = self.k * self.d
        self.base_models = [None]*k
        self.trainset = trainset
        self.testset = testset
        self.state_dir = state_dir
        self.num_classes = num_classes
        self.partitions = None
        os.makedirs(state_dir, exist_ok=True)
        self.lock = Lock()

    def get_partitions(self) -> list[npt.NDArray[np.int64]]:
        """
        Computes the partitions of the dataset that each base model will be trained on, 
        saves to `{state_dir}/partition_info.pth`. If the partitions have already been computed,
        it will load them in instead of recomputing them. 
        Returns indices of the partitions such that the indices
        into the trainset for partiton `i` are in `get_partitions()[i]`

        Based on [Wenxiao Wang's hasing logic](https://github.com/wangwenxiao/FiniteAggregation/blob/main/FiniteAggregation_data_norm_hash.py)
        """
        with self.lock:
            path = f"{self.state_dir}/partition_info.pth"
            if os.path.exists(path):
                print("Partitions already computed, using those...")
                return torch.load(path, map_location=torch.device('cpu'))

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

            partitions = list([x.squeeze().numpy() for x in idxgroup])
            print(f"Finished computing partitions, saving to {self.state_dir}/partition_info.pth")
            torch.save(partitions, f"{self.state_dir}/partition_info.pth")
            print("Partitions saved")
            return partitions

    def train_base_model(self, partition_number: int, train_function: TrainModelFunction) -> None:
        """
        Trains the base model for the specified partition number by calling the train_function
        that the class was instantiated with. Saves the base model to `{state_dir}/base_models` with
        file name `model_{partition_number}.pkl`
        """
        path = f'{self.state_dir}/base_models/model_{str(partition_number)}.pkl'
        if os.path.exists(path):
            print(f"Base model {partition_number} already exists")
            return
        if not os.path.exists(f'{self.state_dir}/base_models'):
            os.mkdir(f'{self.state_dir}/base_models')
        if partition_number < 0 or partition_number >= self.n_subsets:
            raise ValueError("patition_number must be in the range [0, k*d)")
        
        partitions = self.get_partitions()
        print(f'Training Base model {partition_number}..')
        net = train_function(
            partition_number,
            Subset(self.trainset, torch.tensor(partitions[partition_number]))
        )
        print(f'Saving Base model {partition_number} to {path}..')
        torch.save(net, path)
        print(f'Base model {partition_number} saved')

    def get_single_base_model_predictions(self, partition_number, test: bool) -> dict:
        ds_str = ('test' if test else 'train') + "set"
        dataset = self.testset if test else self.trainset

        path = f'{self.state_dir}/predictions/model_{partition_number}_{ds_str}_predictions.pth'

        if os.path.exists(path):
            return torch.load(path, map_location=torch.device('cpu'))
        
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        os.makedirs(f'{self.state_dir}/predictions', exist_ok=True)

        path = f'{self.state_dir}/predictions/model_{partition_number}_{ds_str}_predictions.pth'
        
        seed = partition_number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        net  = torch.load(f'{self.state_dir}/base_models/model_{str(partition_number)}.pkl', map_location=torch.device('cpu'))
        net = net.to(device)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2000, shuffle=False, num_workers=1)

        logits = torch.zeros(len(dataset), self.num_classes).to(device)

        net.eval()
        batch_offset = 0
        with torch.no_grad():
            for (inputs, targets) in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                out = net(inputs)
                next_batch_offset = batch_offset + inputs.size(0)
                logits[batch_offset:next_batch_offset,:] = out
                batch_offset = next_batch_offset
        torch.save(logits, path)
        return logits
        

    def get_all_predictions(self, test=True ) -> dict:
        """
        Runs each base model on the provided testset and saves results to `{state_dir}/{train/test}_predictions.pth`.
        It is not necessary to directly call this method, as it is called by `eval` if predictions have not been computed yet.
        All base models must have already been trained using train_base_model before calling this method.
        """
        ds_str = ('test' if test else 'train') + "set"
        dataset = self.testset if test else self.trainset
        path = f'{self.state_dir}/{ds_str}_predictions.pth'
        if os.path.exists(path):
            print(f"{ds_str} predictions already computed, using those...")
            return torch.load(path, map_location=torch.device('cpu'))

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
        labels = torch.zeros(len(dataset)).to(device)
        for i in tqdm(range(self.n_subsets)):
            logits = self.get_single_base_model_predictions(i, test)
            softmaxes = nn.Softmax(dim=1)(logits)
            logits_by_base_model[:,i,:] = logits
            softmaxes_by_base_model[:,i,:] = softmaxes
            for c in range(self.num_classes):
                logits_by_class[:,c,i] = logits[:,c]
                softmaxes_by_class[:,c,i] = softmaxes[:,c]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2000, shuffle=False, num_workers=1)
        batch_offset = 0
        with torch.no_grad():
            for (inputs, targets) in dataloader:
                labels[batch_offset:inputs.size(0)+batch_offset] = targets
                batch_offset += inputs.size(0)

        print(f"Saving Predictions to {path}")
        output = {
            'logits_by_base_model': logits_by_base_model, 
            'logits_by_class': logits_by_class,
            'softmaxes_by_base_model': softmaxes_by_base_model,
            'softmaxes_by_class': softmaxes_by_class,
            'labels': labels
        }
        torch.save(output, path)
        return output

    def eval(self, mode:str) -> None:
        """
        Generates accuracy and robustness info about the ensemble using the specified mode.
        `mode` can be `label_voting`, `logit_median`, or `softmax_median`.
        Saves certified radii to `{state_dir}/{mode}_certs.pth`
        Outputs accuracy and robustness info to the console.

        Compare radii with different modes to see which one is most robust.

        `softmax_median`, a novel technique, tends to get the best robustness in our experiments.

        `label_voting` code is based on [Wenxiao Wang's certificate logic](https://github.com/wangwenxiao/FiniteAggregation/blob/main/FiniteAggregation_cerfity.py)
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

        filein = self.get_all_predictions(test=True)
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
            certs = torch.load(certs_path, map_location=torch.device('cpu'))
        elif mode == 'label_voting':
            # See https://github.com/wangwenxiao/FiniteAggregation/blob/main/FiniteAggregation_cerfity.py
            # 
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
        elif mode == 'label_runoff':
            # Code from [Keivan Rezaei's work on Run Off Election](https://github.com/k1rezaei/Run-Off-Election/blob/main/fa_roe_certify.py)
            INF = int(10 ** 9)
            # CertV1 
            def get_sample_cert(gap, gap_reducers):
                gap_reducers = gap_reducers.sort(descending=True)[0]
                sample_cert = 0
                while gap > 0:
                    gap -= gap_reducers[sample_cert].item()
                    sample_cert += 1

                return sample_cert

            scores = logits_by_base_model
            num_of_classes = self.num_classes
            num_of_samples = n_sample
            num_of_models = scores.shape[1]

            dec_classes = torch.argsort(scores, dim=2, descending=True)
            idx_roe_fa = torch.zeros((num_of_samples, ), dtype=torch.int) # roe+fa prediction

            predictions = torch.zeros(num_of_samples, num_of_classes).to(device) # number of first-round votes for each class
            for i in range(num_of_models):
                predictions[(torch.arange(num_of_samples),dec_classes[:,i, 0])] += 1

            certs = torch.LongTensor(num_of_samples)

            #prepared for indexing
            shifted = [
                [(h + shift)%self.n_subsets for shift in shifts] for h in range(self.n_subsets)
            ]
            shifted = torch.LongTensor(shifted)

            for i in tqdm(range(num_of_samples)):
                # FA+ROE
                
                # votes in 1st round
                prediction = predictions[i].cpu().numpy()
                ordered_classes = np.argsort(-prediction, kind='stable')

                # top two classes
                m1 = ordered_classes[0].item()
                m2 = ordered_classes[1].item()

                # votes in 2nd round
                m1_election = np.zeros(num_of_classes)
                m2_election = np.zeros(num_of_classes)

                for cls in range(num_of_classes):
                    m1_election[cls] = 2 * (scores[i, :, m1] > scores[i, :, cls]).sum().item() - num_of_models
                    m2_election[cls] = 2 * (scores[i, :, m2] > scores[i, :, cls]).sum().item() - num_of_models
                
                # FA+ROE prediction
                elec = m1_election[m2]
                if elec > 0:
                    idx_roe_fa[i] = m1
                elif elec == 0: # tie
                    if m1 <= m2:
                        idx_roe_fa[i] = m1
                    else:
                        idx_roe_fa[i] = m2
                else:
                    idx_roe_fa[i] = m2


                if idx_roe_fa[i] != labels[i]: # wrong prediction
                    certs[i] = -1
                    continue
                
                c_pred = idx_roe_fa[i]
                c_sec = m1 + m2 - c_pred
                
                certs[i] = self.n_subsets #init value
                label = int(labels[i])

                max_classes_given_h = dec_classes[i, shifted.view(-1), 0].view(-1, self.d)
                m1_election_given_h = torch.zeros((num_of_classes, self.d * self.k, self.d,))
                m2_election_given_h = torch.zeros((num_of_classes, self.d * self.k, self.d,))

                m1_to_m3 = np.zeros(num_of_classes)
                m2_to_m3 = np.zeros(num_of_classes)

                # pw_{b, m1, m3} and pw_{b, m2, m3} in Round 1
                for m3 in range(num_of_classes):
                    gap = prediction[m1] - prediction[m3] + (m3 > m1) # this gap should become non-positive
                    pw = (1 + (max_classes_given_h == m1).long() - (max_classes_given_h == m3).long()).sum(dim=1) # how much each partition can contribute to reduce gap
                    m1_to_m3[m3] = get_sample_cert(gap, pw) # greedy approach

                    gap = prediction[m2] - prediction[m3] + (m3 > m2) # this gap should become non-positive
                    pw = (1 + (max_classes_given_h == m2).long() - (max_classes_given_h == m3).long()).sum(dim=1) # how much each partition can contribute to reduce gap
                    m2_to_m3[m3] = get_sample_cert(gap, pw) # greedy approach
                
                # pw_{b, m1, cls} in Round 2
                for cls in range(num_of_classes):
                    m1_election_given_h[cls] = (scores[i, shifted.view(-1), m1] > scores[i, shifted.view(-1), cls]).view(-1, self.d)
                    m2_election_given_h[cls] = (scores[i, shifted.view(-1), m2] > scores[i, shifted.view(-1), cls]).view(-1, self.d)
                
                
                if c_pred == m1:
                    R1 = m1_to_m3
                    R1_csec = m2_to_m3
                    R2_gaps = m1_election
                    R2_pw = m1_election_given_h
                else:
                    R1 = m2_to_m3
                    R1_csec = m1_to_m3
                    R2_gaps = m2_election
                    R2_pw = m2_election_given_h
                
                # CertR1 = min Certv2(c_pred, c1, c2)
                CertR1 = INF
                
                for c1 in range(num_of_classes):
                    if c1 == c_pred:
                        continue
                    
                    for c2 in range(c1):
                        if c2 == c_pred or c2 == c1:
                            continue
                        
                        n1 = R1[c1]
                        n2 = R1[c2]

                        gap = prediction[c_pred] - prediction[c1] + (c1 > c_pred) + prediction[c_pred] - prediction[c2] + (c2 > c_pred)
                        pw = (1 + 2 * (max_classes_given_h == c_pred).long() - (max_classes_given_h == c1).long() - (max_classes_given_h == c2).long()).sum(dim=1)
                        nsum = get_sample_cert(gap, pw)
                        
                        Certv2_c1_c2 = max(max(n1, n2), nsum)
                        CertR1 = min(CertR1, Certv2_c1_c2)
                
                # CertR2 = min max{Certv1({f_i}, c_sec, c), Certv1({g_i}, c_pred, c)}
                CertR2 = INF
                
                for c in range(num_of_classes):
                    if c == c_pred:
                        continue
                    
                    CertR2_c_1 = R1_csec[c]
                    gap = R2_gaps[c] + (c > c_pred)
                    pw = 2 * R2_pw[c].sum(dim=1)
                    CertR2_c_2 = get_sample_cert(gap, pw)

                    CertR2_c = max(CertR2_c_1, CertR2_c_2)
                    CertR2 = min(CertR2, CertR2_c)

                assert(CertR1 > 0 and CertR2 > 0)
                certs[i] = min(CertR1, CertR2) - 1
            
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
            softmaxes = nn.Softmax(dim=1)(softmaxes)
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
        print('Clean Accuracy: ' + str(accs[0] * 100.) + '%')
        print('Median Certified Radius: ' + str(sum(accs >= 0.5)))

    def certified_accuracy(self, mode:str, attack_size:int) -> float:
        """
        Returns the certified accuracy of the ensemble using the specified mode and attack size.
        """
        certs_path = f'{self.state_dir}/{mode}_certs.pth'
        if not os.path.exists(certs_path):
            raise Exception("Certificates not computed yet, please call eval first")
        certs = torch.load(certs_path, map_location=torch.device('cpu'))
        a = certs.cpu().sort()[0].numpy()
        accs = np.array([(i <= a).sum() for i in np.arange(np.amax(a)+1)])/len(certs)
        if attack_size > len(accs):
            return 0
        return accs[attack_size]

    def distill(self, student: Module, mode: str, lr: float, seed: int=0, epochs:int=10) -> Module:
        """
        Distills the ensemble into a single model and saves to `{state_dir}/students/student_{id}.pkl`
        where `id` is generated based on the hyperparameters of the distillation that are passed in.

       `mode` can be `label_voting`, `logit_median`, or `softmax_median`.

       Compare accuracies with different parameters and modes. 
       
       `logit_median`, a novel technique, tends to get the best accuracy in our experiments
        """

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        id = mode + str(hash(lr)) + str(epochs) + str(seed)
        student_path = f'{self.state_dir}/students/student_{str(id)}.pkl'
        print(student_path)
        if os.path.exists(student_path):
            print(f"Student already trained with those parameters at {student_path}, loading it in instead of training again...")
            student = torch.load(student_path, map_location=torch.device('cpu'))
        else:
            preds = self.get_all_predictions(test=False)
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
                ensemble_outputs = nn.Softmax(dim=1)(softmaxes[:,:,mid])
            elif mode == "label_runoff":
                print("Computing ensemble outputs for label runoff. This work is also done during evaluation, so this is redundant and will be removed in the future via storing the results.")
                random.seed(999999999+208)
                shifts = random.sample(range(self.n_subsets), self.d)
                scores = logits_by_base_model
                num_of_classes = self.num_classes
                num_of_samples = logits_by_base_model.shape[0]
                num_of_models = logits_by_base_model.shape[1]

                dec_classes = torch.argsort(scores, dim=2, descending=True)
                ensemble_outputs = torch.zeros((num_of_samples, ), dtype=torch.long).to(device)
                predictions = torch.zeros(num_of_samples, num_of_classes).to(device) # number of first-round votes for each class
                for i in range(num_of_models):
                    predictions[(torch.arange(num_of_samples),dec_classes[:,i, 0])] += 1

                #prepared for indexing
                shifted = [
                    [(h + shift)%self.n_subsets for shift in shifts] for h in range(self.n_subsets)
                ]
                shifted = torch.LongTensor(shifted)

                for i in tqdm(range(num_of_samples)):
                    # FA+ROE
                    
                    # votes in 1st round
                    prediction = predictions[i].cpu().numpy()
                    ordered_classes = np.argsort(-prediction, kind='stable')

                    # top two classes
                    m1 = ordered_classes[0].item()
                    m2 = ordered_classes[1].item()

                    # votes in 2nd round
                    m1_election = np.zeros(num_of_classes)
                    m2_election = np.zeros(num_of_classes)

                    for cls in range(num_of_classes):
                        m1_election[cls] = 2 * (scores[i, :, m1] > scores[i, :, cls]).sum().item() - num_of_models
                        m2_election[cls] = 2 * (scores[i, :, m2] > scores[i, :, cls]).sum().item() - num_of_models
                    
                    # FA+ROE prediction
                    elec = m1_election[m2]
                    if elec > 0:
                        ensemble_outputs[i] = m1
                    elif elec == 0: # tie
                        if m1 <= m2:
                            ensemble_outputs[i] = m1
                        else:
                            ensemble_outputs[i] = m2
                    else:
                        ensemble_outputs[i] = m2
            else:
                raise ValueError("Invalid distillation mode")
            
            if not os.path.exists(f'{self.state_dir}/students'):
                os.mkdir(f'{self.state_dir}/students')

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128, shuffle=False)
            student.to(device)

            criterion = (
                torch.nn.MSELoss() if mode == 'logit_median' 
                else nn.CrossEntropyLoss()
            )

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

        