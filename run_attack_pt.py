import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import argparse

from resnet import ResNet18
from utils_pt import load_data

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Define hyperparameters.')
  parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, mnist')
  parser.add_argument('--attack', type=str, default='CS', help='PGD, CS')
  parser.add_argument('--path_results', type=str, default='none')
  parser.add_argument('--n_examples', type=int, default=50)
  parser.add_argument('--data_dir', type=str, default= './data')
  
  hps = parser.parse_args()
  
  # load model
  model = ResNet18().cuda()
  ckpt = torch.load('./models/model_test.pt')
  model.load_state_dict(ckpt)
  model.eval()
  
  # load data
  x_test, y_test = load_data(hps.dataset, hps.n_examples, hps.data_dir)
  
  # x_test, y_test are images and labels on which the attack is run
  # x_test in the format bs (batch size) x heigth x width x channels
  # y_test in the format bs
  
  if hps.attack == 'PGD':
    import pgd_attacks_pt
    
    args = {'type_attack': 'L0',
                'n_restarts': 5,
                'num_steps': 100,
                'step_size': 120000.0/255.0,
                'kappa': -1,
                'epsilon': -1,
                'sparsity': 5}
            
    attack = pgd_attacks_pt.PGDattack(model, args)
    
    adv, pgd_adv_acc = attack.perturb(x_test, y_test)
    
    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)  
  
  elif hps.attack == 'CS':
    import cornersearch_attacks_pt
    
    args = {'type_attack': 'L0',
            'n_iter': 1000,
            'n_max': 100,
            'kappa': -1,
            'epsilon': -1,
            'sparsity': 10,
            'size_incr': 1}
    
    attack = cornersearch_attacks_pt.CSattack(model, args)
    
    adv, pixels_changed, fl_success = attack.perturb(x_test, y_test)
    
    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)
    
