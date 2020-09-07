# [Sparse and Imperceivable Adversarial Attacks](https://arxiv.org/abs/1909.05040)
Francesco Croce, Matthias Hein\
*University of Tübingen*\
\
Accepted to ICCV 2019

## Metrics and attacks
We consider three threat models:
+ **L0**: aims at changing the smallest number of pixels, with no constraints on the size of the modification of each pixel\
(except for the usual [0,1] box constraints), to get adversarial examples,
+ **L0+Linf**: aims at changing the smallest number of pixels with perturbations of bounded Linf-norm,
+ **L0+sigma**: aims at changing the smallest number of pixels with imperceivable perturbations.

We propose two adversarial attacks, each one able to handle the three scenarios mentioned above:
+ **CornerSearch**: a black-box attack, which minimize the L0-norm of the perturbations,
+ **PGD**: an extention of usual Projected Gradient Descent white-box attack to the L0-norm. It requires to fix the sparsity level of the\
adversarial perturbations (the number of pixels changed).

Our attacks wrt L0 achieve state-of-the-art results, outperforming both black- and white-box attacks.
<p align="center"><img src="https://github.com/fra31/sparse-imperceivable-attacks/blob/master/images/pl_robacc_mnist_3.png" width="400" \>
<img src="https://github.com/fra31/sparse-imperceivable-attacks/blob/master/images/pl_robacc_cifar10_correct_3.png" width="400">

With the constraints given by the sigma-map we introduce, we can craft sparse and imperceivable adversarial perturbations.
<p align="center"><img src="https://github.com/fra31/sparse-imperceivable-attacks/blob/master/images/img_gh_1.png" width="800">

## Running the attacks
We provide an implementation of CornerSearch and PGD for all types of attacks, i.e. `L0`, `L0+Linf` and `L0+sigma`, with versions for TensorFlow and PyTorch models.

The threat model with relative parameters can be set for CornerSearch [here](https://github.com/fra31/sparse-imperceivable-attacks/blob/2e443b84ae55ba2677173dec6bf92d0b7cebe6d5/run_attack.py#L67) and [here](https://github.com/fra31/sparse-imperceivable-attacks/blob/2e443b84ae55ba2677173dec6bf92d0b7cebe6d5/run_attack_pt.py#L54), while for PGD [here](https://github.com/fra31/sparse-imperceivable-attacks/blob/2e443b84ae55ba2677173dec6bf92d0b7cebe6d5/run_attack.py#L50) and [here](https://github.com/fra31/sparse-imperceivable-attacks/blob/2e443b84ae55ba2677173dec6bf92d0b7cebe6d5/run_attack_pt.py#L37).

We include pretrained Tensorflow and PyTorch models (see `models`) to run the attacks with the following examples. With
```python
python run_attack.py --dataset=[cifar10 | mnist] --attack=[CS | PGD] --n_examples=100 --data_dir=/path/to/data
```
one can run CornerSearch (CS) or PGD on a model implemented in TensorFlow, trained on either CIFAR-10 or MNIST, while with
```python
python run_attack_pt.py --attack=[CS | PGD] --n_examples=100 --data_dir=/path/to/data
```
a model implemented in PyTorch and trained on CIFAR-10 is used. Additionally, adding `--path_results=/path/to/results` sets where the results are saved.

**Note**: all the parameters of the attacks can be set in `run_attacks.py` and `run_attacks_pt.py`. A more detailed description of
each of them is available in `cornersearch_attack.py` and `pgd_attacks.py`. Please note they might need to be tuned to achieve the best performance on new models, depending on the dataset, threat model and characteristics of the classifier.

## Citations
```
@inproceedings{croce2019sparse,
  title={Sparse and Imperceivable Adversarial Attacks},
  author={Croce, Francesco and Hein, Matthias},
  booktitle={ICCV},
  year={2019}
}
```
