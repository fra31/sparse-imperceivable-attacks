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

With the constraints given by the sigma-map we introduce, we can craft sparse and imperceivable adversarial perturbations.
<p align="center"><img src="https://raw.githubusercontent.com/fra31/sparse-imperceivable-attacks/master/images/pl_gh_1.png" width="800">

## Running the attacks
With python `run_attack.py --dataset=cifar10 --attack=CS --path_results=/path/to/results` one can run CornerSearch (CS) on a CIFAR-10
model (note that the datasets need to be loaded separately). All the parameters can be set in `run_attacks.py` and the description of
each of them
is available in `cornersearch_attack.py` and `pgd_attacks.py`.

## Citations
```
@inproceedings{croce2019sparse,
  title={Sparse and Imperceivable Adversarial Attacks},
  author={Croce, Francesco and Hein, Matthias},
  booktitle={ICCV},
  year={2019}
}
```
