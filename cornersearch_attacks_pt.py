#import tensorflow as tf
import scipy.io
import numpy as np
import torch
from utils_pt import get_logits, get_predictions

def onepixel_perturbation(attack, orig_x, pos, sigma):
  ''' returns a batch with the possible perturbations of the pixel in position pos '''
    
  if attack.type_attack == 'L0':
    if orig_x.shape[-1] == 3:
      batch_x = np.tile(orig_x,(8,1,1,1))
      t = np.zeros([3])
      for counter in range(8):
        t2 = counter + 0
        for c in range(3):
          t[c] = t2 % 2
          t2 = (t2 - t[c])/2
        batch_x[counter,pos[0],pos[1]] = t.astype(np.float32)
    elif orig_x.shape[-1] == 1:
      batch_x = np.tile(orig_x,(2,1,1,1))
      batch_x[0,pos[0],pos[1],0] = 0.0
      batch_x[1,pos[0],pos[1],0] = 1.0
  
  elif attack.type_attack == 'L0+Linf':
    if orig_x.shape[-1] == 3:
      batch_x = np.tile(orig_x,(8,1,1,1))
      t = np.zeros([3])
      for counter in range(8):
        t2 = counter + 0
        for c in range(3):
          t3 = t2 % 2
          t[c] = (t3*2.0 - 1.0)*attack.epsilon
          t2 = (t2 - t3)/2
        batch_x[counter,pos[0],pos[1]] = np.clip(t.astype(np.float32) + orig_x[pos[0],pos[1]], 0.0, 1.0)
    elif orig_x.shape[-1] == 1:
      batch_x = np.tile(orig_x,(2,1,1,1))
      batch_x[0,pos[0],pos[1],0] = np.clip(batch_x[0,pos[0],pos[1],0] - attack.epsilon, 0.0, 1.0)
      batch_x[1,pos[0],pos[1],0] = np.clip(batch_x[1,pos[0],pos[1],0] + attack.epsilon, 0.0, 1.0)
  
  elif attack.type_attack == 'L0+sigma':
    batch_x = np.tile(orig_x,(2,1,1,1))
    if orig_x.shape[-1] == 3:
      batch_x[0,pos[0],pos[1]] = np.clip(batch_x[0,pos[0],pos[1]]*(1.0 - attack.kappa*sigma[pos[0],pos[1]]), 0.0, 1.0)
      batch_x[1,pos[0],pos[1]] = np.clip(batch_x[0,pos[0],pos[1]]*(1.0 + attack.kappa*sigma[pos[0],pos[1]]), 0.0, 1.0)
    
    elif orig_x.shape[-1] == 1:
      batch_x[0,pos[0],pos[1]] = np.clip(batch_x[0,pos[0],pos[1]] - attack.kappa*sigma[pos[0],pos[1]], 0.0, 1.0)
      batch_x[1,pos[0],pos[1]] = np.clip(batch_x[0,pos[0],pos[1]] + attack.kappa*sigma[pos[0],pos[1]], 0.0, 1.0)
    
  else:
    raise ValueError('unknown attack')
  
  return batch_x
    
def onepixel_perturbation_image(attack, orig_x, sigma):
  ''' returns a batch with all the possible perturbations of the image orig_x '''
  
  n_channels = orig_x.shape[-1]
  assert n_channels in [1, 3]
  n_corners = 2**n_channels if attack.type_attack in ['L0', 'L0+Linf'] else 2
  
  batch_x = np.zeros([n_corners*orig_x.shape[0]*orig_x.shape[1], orig_x.shape[0], orig_x.shape[1], orig_x.shape[2]])
  for counter in range(orig_x.shape[0]):
      for counter2 in range(orig_x.shape[1]):
        batch_x[(counter*orig_x.shape[0]+counter2)*n_corners:(counter*orig_x.shape[1]+counter2)*n_corners+n_corners] = np.clip(onepixel_perturbation(attack, orig_x, [counter,counter2], sigma), 0.0, 1.0)
  
  return batch_x

def flat2square(attack, ind):
  ''' returns the position and the perturbation given the index of an image
      of the batch of all the possible perturbations '''
  
  if attack.type_attack in ['L0', 'L0+Linf']:
    if attack.shape_img[-1] == 3:
      new_pixel = ind % 8
      ind = (ind - new_pixel)//8
      c = ind % attack.shape_img[1]
      r = (ind - c)//attack.shape_img[1]
      t = np.zeros([ind.shape[0],3])
      for counter in range(3):
        t[:,counter] = new_pixel % 2
        new_pixel = (new_pixel - t[:,counter])/2
    elif attack.shape_img[-1] == 1:
      t = ind % 2
      ind = (ind-t)//2
      c = ind % attack.shape_img[1]
      r = (ind-c)//attack.shape_img[1]
  
  elif attack.type_attack == 'L0+sigma':
      t = ind % 2
      c = ((ind - t)//2) % attack.shape_img[1]
      r = ((ind - t)//2 - c)//attack.shape_img[1]
    
  return r, c, t

def npixels_perturbation(attack, orig_x, ind, k, sigma):
  ''' creates n_iter images which differ from orig_x in at most k pixels '''
  
  # sampling the n_iter k-pixels perturbations
  ind2 = np.random.randint(0, attack.n_max**2, (attack.n_iter, k))
  ind2 = attack.n_max - np.floor(ind2**0.5).astype(int) - 1
  
  # creating the n_iter k-pixels perturbed images
  batch_x = np.tile(orig_x,(attack.n_iter,1,1,1))
  if attack.type_attack == 'L0':
    for counter in range(attack.n_iter):
      p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
      batch_x[counter,p11,p12] = d1 + 0 if attack.shape_img[-1] == 3 else np.expand_dims(d1 + 0, 1)
  
  elif attack.type_attack == 'L0+Linf':
    for counter in range(attack.n_iter):
      p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
      d1 = d1 + 0 if attack.shape_img[-1] == 3 else np.expand_dims(d1 + 0, 1)
      batch_x[counter,p11,p12] = np.clip(batch_x[counter,p11,p12]+(2.0*d1 - 1.0)*attack.epsilon, 0.0, 1.0)
  
  elif attack.type_attack == 'L0+sigma':
    for counter in range(attack.n_iter):
      p11, p12, d1 = flat2square(attack, ind[ind2[counter]])
      d1 = np.expand_dims(d1,1)
      if attack.shape_img[-1] == 3: batch_x[counter,p11,p12] = np.clip(batch_x[counter,p11,p12] - attack.kappa*sigma[p11,p12]*(1-d1) + attack.kappa*sigma[p11,p12]*d1, 0.0, 1.0)
      elif attack.shape_img[-1] == 1: batch_x[counter,p11,p12] = np.clip(batch_x[counter,p11,p12] - attack.kappa*sigma[p11,p12]*(1-d1) + attack.kappa*sigma[p11,p12]*d1, 0.0, 1.0)
      
  return batch_x

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
  
class CSattack():
  def __init__(self, model, args):
    self.model = model
    self.type_attack = args['type_attack'] # 'L0', 'L0+Linf', 'L0+sigma'
    self.n_iter = args['n_iter']           # number of iterations (N_iter in the paper)
    self.n_max = args['n_max']             # the modifications for k-pixels perturbations are sampled among the best n_max (N in the paper)
    self.epsilon = args['epsilon']         # for L0+Linf, the bound on the Linf-norm of the perturbation
    self.kappa = args['kappa']             # for L0+sigma (see kappa in the paper), larger kappa means easier and more visible attacks
    self.k = args['sparsity']              # maximum number of pixels that can be modified (k_max in the paper)
    self.size_incr = args['size_incr']     # size of progressive increment of sparsity levels to check  
  
  def perturb(self, x_nat, y_nat):
    adv = np.copy(x_nat)
    fl_success = np.ones([x_nat.shape[0]])
    self.shape_img = x_nat.shape[1:]
    self.sigma = sigma_map(x_nat)
    self.n_classes = 10
    self.n_corners = 2**self.shape_img[2] if self.type_attack in ['L0', 'L0+Linf'] else 2
    #corr_pred = sess.run(self.model.correct_prediction, {self.model.x_input: x_nat, self.model.y_input: y_nat})
    corr_pred = get_predictions(self.model, x_nat, y_nat)
    bs = self.shape_img[0]*self.shape_img[1]
    
    for c in range(x_nat.shape[0]):
      if corr_pred[c]:
        sigma = np.copy(self.sigma[c])
        batch_x = onepixel_perturbation_image(self, x_nat[c], sigma)
        batch_y = np.squeeze(y_nat[c])
        logit_2 = np.zeros([batch_x.shape[0], self.n_classes])
        found = False
        
        # checks one-pixels modifications
        for counter in range(self.n_corners):
          #logit_2[counter*bs:(counter+1)*bs], pred = sess.run([self.model.y, self.model.correct_prediction], feed_dict={self.model.x_input: batch_x[counter*bs:(counter+1)*bs], self.model.y_input: np.tile(batch_y,(bs))})
          logit_2[counter*bs:(counter+1)*bs] = get_logits(self.model, batch_x[counter*bs:(counter+1)*bs])
          pred = logit_2[counter*bs:(counter+1)*bs].argmax(axis=-1) == np.tile(batch_y,(bs))
          if not pred.all() and not found:
            ind_adv = np.where(pred.astype(int)==0)
            adv[c] = batch_x[counter*bs + ind_adv[0][0]]
            found = True
            print('Point {} - adversarial example found changing 1 pixel'.format(c))
        
        # creates the orderings
        t1 = np.copy(logit_2[:, batch_y])
        logit_2[:, batch_y] = -1000.0*np.ones(np.shape(logit_2[:, batch_y]))
        t2 = np.amax(logit_2, axis=1)
        t3 = t1 - t2
        logit_3 = np.tile(np.expand_dims(t1,axis=1),(1,self.n_classes))-logit_2
        logit_3[:, batch_y] = t3
        ind = np.argsort(logit_3, axis=0)
        
        # checks multiple-pixels modifications
        for n3 in range(2,self.k,self.size_incr):
          if not found:
             for c2 in range(self.n_classes):
               if not found:
                 ind_cl = np.copy(ind[:, c2])

                 batch_x = npixels_perturbation(self, x_nat[c], ind_cl, n3, sigma)
                 #pred = sess.run(self.model.correct_prediction, feed_dict={self.model.x_input: batch_x, self.model.y_input: np.tile(batch_y,(batch_x.shape[0]))})
                 pred = get_predictions(self.model, batch_x, np.tile(batch_y,(batch_x.shape[0])))
                 
                 if np.sum(pred.astype(np.int32)) < self.n_iter and not found:
                   found = True
                   ind_adv = np.where(pred.astype(int)==0)
                   adv[c] = batch_x[ind_adv[0][0]]
                   print('Point {} - adversarial example found changing {} pixels'.format(c, np.sum(np.amax(np.abs(adv[c] - x_nat[c]) > 1e-10, axis=-1), axis=(0,1))))
        
        if not found:
          fl_success[c] = 0
          print('Point {} - adversarial example not found'.format(c))
      
      else:
        print('Point {} - misclassified'.format(c))
    
    pixels_changed = np.sum(np.amax(np.abs(adv - x_nat) > 1e-10, axis=-1), axis=(1,2))
    print('Pixels changed: ', pixels_changed)
    #print('attack successful: ', fl_success)
    #print('attack successful: {:.2f}%'.format((1.0 - np.mean(fl_success))*100.0))
    #corr_pred = sess.run(self.model.correct_prediction, {self.model.x_input: adv, self.model.y_input: y_nat})
    corr_pred = get_predictions(self.model, adv, y_nat)
    print('Robust accuracy at {} pixels: {:.2f}%'.format(self.k, np.sum(corr_pred)/x_nat.shape[0]*100.0))
    print('Maximum perturbation size: {:.5f}'.format(np.amax(np.abs(adv - x_nat))))
    
    return adv, pixels_changed, fl_success