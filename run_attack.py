import tensorflow as tf
import scipy.io
import numpy as np
import argparse

def load_model_and_dataset(dataset):
  if dataset == 'mnist':
    import mnist_NiN_bn
    model = mnist_NiN_bn.NiN_Model()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint('/home/repository_mnist_relu/Tensorflow_version/nin_model/')
    saver.restore(sess, checkpoint)
    
  elif dataset == 'cifar10':
    import cifar_NiN_bn
    model = cifar_NiN_bn.NiN_Model()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint('/home/cifar-10-cnn/Tensorflow_version/nin_model/')
    saver.restore(sess, checkpoint)
    
  else:
    raise ValueError('unknown dataset')
    
  return model
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Define hyperparameters.')
  parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, mnist')
  parser.add_argument('--attack', type=str, default='CS', help='PGD, CS')
  parser.add_argument('--path_results', type=str, default='none')
  
  hps = parser.parse_args()
  
  sess = tf.InteractiveSession()
  model = load_model(hps.dataset)
  
  # x_test, y_test are images and labels on which the attack is run (to be loaded)
  # x_test in the format bs (batch size) x heigth x width x channels
  # y_test in the format bs
  
  if hps.attack == 'PGD':
    import pgd_attacks
    
    args = {'type_attack': 'L0+sigma',
            'n_restarts': 3,
            'num_steps': 20,
            'step_size': 120000.0/255.0/2.0,
            'kappa': 0.8,
            'epsilon': -1,
            'sparsity': 50}
            
    attack = pgd_attacks.PGDattack(model, args)
    
    adv, pgd_adv_acc = attack.perturb(x_test, y_test, sess)
    
    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)  
  
  elif hps.attack == 'CS':
    import cornersearch_attacks
    
    args = {'type_attack': 'L0+sigma',
            'n_iter': 1000,
            'n_max': 150,
            'kappa': 0.8,
            'epsilon': -1,
            'sparsity': 100,
            'size_incr': 5}
    
    attack = cornersearch_attacks.CSattack(model, args)
    
    adv, pixels_changed, fl_success = attack.perturb(x_test, y_test, sess)
    
    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)
    
  sess.close()