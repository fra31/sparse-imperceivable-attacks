import tensorflow as tf
import scipy.io
import numpy as np

def project_L0_box(y, k, lb, ub):
  ''' projection of the batch y to a batch x such that:
        - each image of the batch x has at most k pixels with non-zero channels
        - lb <= x <= ub '''
      
  x = np.copy(y)
  p1 = np.sum(x**2, axis=-1)
  p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
  p2 = np.sum(p2**2, axis=-1)
  p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
  x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
  x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
    
  return x
  
def project_L0_sigma(y, k, sigma, kappa, x_nat):
  ''' projection of the batch y to a batch x such that:
        - 0 <= x <= 1
        - each image of the batch x differs from the corresponding one of
          x_nat in at most k pixels
        - (1 - kappa*sigma)*x_nat <= x <= (1 + kappa*sigma)*x_nat '''
  
  x = np.copy(y)
  p1 = 1.0/np.maximum(1e-12, sigma)*(x_nat > 0).astype(float) + 1e12*(x_nat == 0).astype(float)
  p2 = 1.0/np.maximum(1e-12, sigma)*(1.0/np.maximum(1e-12, x_nat) - 1)*(x_nat > 0).astype(float) + 1e12*(x_nat == 0).astype(float) + 1e12*(sigma == 0).astype(float)
  lmbd_l = np.maximum(-kappa, np.amax(-p1, axis=-1, keepdims=True))
  lmbd_u = np.minimum(kappa, np.amin(p2, axis=-1, keepdims=True))
  
  lmbd_unconstr = np.sum((y - x_nat)*sigma*x_nat, axis=-1, keepdims=True)/np.maximum(1e-12, np.sum((sigma*x_nat)**2, axis=-1, keepdims=True))
  lmbd = np.maximum(lmbd_l, np.minimum(lmbd_unconstr, lmbd_u))
  
  p12 = np.sum((y - x_nat)**2, axis=-1, keepdims=True)
  p22 = np.sum((y - (1 + lmbd*sigma)*x_nat)**2, axis=-1, keepdims=True)
  p3 = np.sort(np.reshape(p12 - p22, [x.shape[0],-1]))[:,-k]
  
  x  = x_nat + lmbd*sigma*x_nat*((p12 - p22) >= p3.reshape([-1, 1, 1, 1]))
  
  return x
  
def perturb_L0_box(attack, x_nat, y_nat, lb, ub, sess):
  ''' PGD attack wrt L0-norm + box constraints
  
      it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
      such that:
        - each image of the batch adv differs from the corresponding one of
          x_nat in at most k pixels
        - lb <= adv - x_nat <= ub
      
      it returns also a vector of flags where 1 means no adversarial example found
      (in this case the original image is returned in adv) '''
  
  if attack.rs:
    x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
    x2 = np.clip(x2, 0, 1)
  else:
    x2 = np.copy(x_nat)
      
  adv_not_found = np.ones(y_nat.shape)
  adv = np.zeros(x_nat.shape)

  for i in range(attack.num_steps):
    if i > 0:
      pred, grad = sess.run([attack.model.correct_prediction, attack.model.grad], feed_dict={attack.model.x_input: x2, attack.model.y_input: y_nat})
      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
      
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + attack.step_size * grad, casting='unsafe')
      
    x2 = x_nat + project_L0_box(x2 - x_nat, attack.k, lb, ub)
    
  return adv, adv_not_found
  
def perturb_L0_sigma(attack, x_nat, y_nat, sess):
  ''' PGD attack wrt L0-norm + sigma-map constraints
  
      it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
      such that:
        - each image of the batch adv differs from the corresponding one of
          x_nat in at most k pixels
        - (1 - kappa*sigma)*x_nat <= adv <= (1 + kappa*sigma)*x_nat
      
      it returns also a vector of flags where 1 means no adversarial example found
      (in this case the original image is returned in adv) '''
  
  if attack.rs:
    x2 = x_nat + np.random.uniform(-attack.kappa, attack.kappa, x_nat.shape)
    x2 = np.clip(x2, 0, 1)
  else:
    x2 = np.copy(x_nat)
      
  adv_not_found = np.ones(y_nat.shape)
  adv = np.zeros(x_nat.shape)

  for i in range(attack.num_steps):
    if i > 0:
      pred, grad = sess.run([attack.model.correct_prediction, attack.model.grad], feed_dict={attack.model.x_input: x2, attack.model.y_input: y_nat})
      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
      
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + attack.step_size * grad, casting='unsafe')
      
    x2 = project_L0_sigma(x2, attack.k, attack.sigma, attack.kappa, x_nat)
    
  return adv, adv_not_found
  
def sigma_map(x):
  ''' creates the sigma-map for the batch x '''
  
  sh = [4]
  sh.extend(x.shape)
  t = np.zeros(sh)
  t[0,:,:-1] = x[:,1:]
  t[0,:,-1] = x[:,-1]
  t[1,:,1:] = x[:,:-1]
  t[1,:,0] = x[:,0]
  t[2,:,:,:-1] = x[:,:,1:]
  t[2,:,:,-1] = x[:,:,-1]
  t[3,:,:,1:] = x[:,:,:-1]
  t[3,:,:,0] = x[:,:,0]

  mean1 = (t[0] + x + t[1])/3
  sd1 = np.sqrt(((t[0]-mean1)**2 + (x-mean1)**2 + (t[1]-mean1)**2)/3)

  mean2 = (t[2] + x + t[3])/3
  sd2 = np.sqrt(((t[2]-mean2)**2 + (x-mean2)**2 + (t[3]-mean2)**2)/3)

  sd = np.minimum(sd1, sd2)
  sd = np.sqrt(sd)
  
  return sd

class PGDattack():
  def __init__(self, model, args):
    self.model = model
    self.type_attack = args['type_attack'] # 'L0', 'L0+Linf', 'L0+sigma'
    self.num_steps = args['num_steps']
    self.step_size = args['step_size']
    self.n_restarts = args['n_restarts']
    self.rs = True                         # random starting point
    self.epsilon = args['epsilon']         # for L0+Linf
    self.kappa = args['kappa']             # for L0+sigma
    self.k = args['sparsity']
    
  def perturb(self, x_nat, y_nat, sess):
    adv = np.copy(x_nat)
    
    if self.type_attack == 'L0+sigma': self.sigma = sigma_map(x_nat)
      
    for counter in range(self.n_restarts):
      if counter == 0:
        corr_pred = sess.run(self.model.correct_prediction, {self.model.x_input: x_nat, self.model.y_input: y_nat})
        pgd_adv_acc = np.copy(corr_pred)
        
      if self.type_attack == 'L0':
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, -x_nat, 1.0 - x_nat, sess)
      
      elif self.type_attack == 'L0+Linf':
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, np.maximum(-self.epsilon, -x_nat), np.minimum(self.epsilon, 1.0 - x_nat), sess)
      
      elif self.type_attack == 'L0+sigma' and x_nat.shape[3] == 3:
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_sigma(self, x_nat, y_nat, sess)
      
      elif self.type_attack == 'L0+sigma' and x_nat.shape[3] == 1:
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, np.maximum(-self.kappa*self.sigma, -x_nat), np.minimum(self.kappa*self.sigma, 1.0 - x_nat), sess)
      
      
      pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)

      print("Restart {} - Robust accuracy: {}".format(counter + 1, np.sum(pgd_adv_acc)/x_nat.shape[0]))
      adv[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv[np.logical_not(curr_pgd_adv_acc)]
    
    pixels_changed = np.sum(np.amax(np.abs(adv - x_nat) > 1e-10, axis=-1), axis=(1,2))
    print('Pixels changed: ', pixels_changed)
    corr_pred = sess.run(self.model.correct_prediction, {self.model.x_input: adv, self.model.y_input: y_nat})
    print('Robust accuracy at {} pixels: {:.2f}%'.format(self.k, np.sum(corr_pred)/x_nat.shape[0]*100.0))
    print('Maximum perturbation size: {:.5f}'.format(np.amax(np.abs(adv - x_nat))))
    
    return adv, pgd_adv_acc