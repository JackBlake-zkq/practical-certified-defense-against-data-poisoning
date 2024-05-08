# Practical Certified Defense against Data Poisoning

Existing work on certified defenses against data poisoning yields systems that are prohibitively computationally expensive at inference time.

Existing work is also inconvinient to adapt to a different dataset (especially), architecture, or training method.

We aim to integrated current defenses with knowledge distillation, and develop new aggregation methods that are better suited for distillation.

## Prior Work

This project is base on work related to certified defenses against data poisining, particularly ones using ensembles of models trained on partitioned data.

We build on the following works:

- [DEEP PARTITION AGGREGATION: PROVABLE DEFENSES AGAINST GENERAL POISONING ATTACKS - Alexander Levine & Soheil Feizi](https://arxiv.org/pdf/2006.14768)
- [Improved Certified Defenses against Data Poisoning with (Deterministic) Finite Aggregation - Wenxiao Wang, Alexander J Levine, Soheil Feizi](https://proceedings.mlr.press/v162/wang22m.html)
- [Run-Off Election: Improved Provable Defense against Data Poisoning Attacks - Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi](https://arxiv.org/pdf/2302.02300)

We also referenced:

- [Distillation Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo?tab=readme-ov-file)

## Idea

This repo provides a system to:

- Automatically partition dataset
- Train base models of any architecture, with any dataset (samples need to be tensors e.g. via a ToTensor transformation)
- Evaluate Enseble Accuracy and Robustness (with multiple aggregation techniques)
- Distill into a single model by training on ensemble outputs (with any/each of the aggregation techniques)

## Solution

The class provided in `fa_ensemble.py` provides the tools necessary to train, evaluate, and distill. See notebooks for example usage.

We also introduce 2 novel aggregation methods, which we call Logit Median Aggregation, and Softmax Median Aggregation.

### Logit Aggregation

To generate the prediction for a sample: for each class, takes the median logit value across all base models e.g. for 3 base models with logits [1, 2, 3], [2, 3, 4], [3, 4, 5], the median logit for class 0 would be 2, for class 1 would be 3, and for class 2 would be 4, so the ensemble outputs [2, 3, 4] as its logits.

This 

### Softmax Median

Same thing, but take the softmax before taking the median. Notably, the output is not a softmax, so we should take the softmax of the output to get the final softmax prediction.

### "Proof" of Certified Defense

From a single poison training sample, at most d partitions will be affected, thus d base models. We'll assume pessimistically that the adversary gains full control over the output of each of these d base models. 

In the worst case, the adversary will make the logit for the correctly predicted class -inf, and all others inf, since they are trying to change the prediction. Thus, the median for the predicted class will shift back by at most d, and the median for all other classes will shift forward by at most d. If the max of the medians does not change after this shift, we can sustain that attack.

The certified radius for an input is the amount of times we can iterate this process without the max of the medians changing.

## Results

See `benchmark.py` for results on MNIST with k=1200, d=1, same training as previous work.

Logit Median gets poor certified accuracies, and poor accuracy under distillation.

Softmax Median establishes a new state of the art, so long as the proof is correct. It achieves higher certified accuracies for all attack sizes, and has good accuracy under distillation.