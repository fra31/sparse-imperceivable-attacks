import tensorflow as tf
import scipy.io
import numpy as np

def conv(x, phase, shape):
    he_initializer = tf.contrib.keras.initializers.he_normal()
    W = tf.get_variable('weights', shape=shape, initializer=he_initializer)
    b = tf.get_variable('bias', shape=[shape[3]], initializer=tf.zeros_initializer)
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x,b)
    # return tf.contrib.layers.batch_norm(x,is_training=phase)    
    return tf.layers.batch_normalization(x,axis=-1,training=phase,name="bn")

def activation(x):
    return tf.nn.relu(x) 

def max_pool(input, k_size=3, stride=2):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME')

def global_avg_pool(input, k_size=1, stride=1):
    return tf.nn.avg_pool(input, ksize=[1,k_size,k_size,1], strides=[1,stride,stride,1], padding='VALID')

def inference(x, phase=False, keep_prob=1.0):
    with tf.variable_scope('conv1'):
        x = conv(x, phase, [5, 5, 1, 192])
        x = activation(x)

    with tf.variable_scope('mlp1-1'):
        x = conv(x, phase, [1, 1, 192, 160])
        x = activation(x)

    with tf.variable_scope('mlp1-2'):
        x = conv(x, phase, [1, 1, 160, 96])
        x = activation(x)

    with tf.name_scope('max_pool-1'):
        x  = max_pool(x, 3, 2)

    with tf.variable_scope('conv2'):
        x = conv(x, phase, [5, 5, 96, 192])
        x = activation(x)

    with tf.variable_scope('mlp2-1'):
        x = conv(x, phase, [1, 1, 192, 192])
        x = activation(x)

    with tf.variable_scope('mlp2-2'):
        x = conv(x, phase, [1, 1, 192, 192])
        x = activation(x)

    with tf.name_scope('max_pool-2'):
        x  = max_pool(x, 3, 2)

    with tf.variable_scope('conv3'):
        x = conv(x, phase, [3, 3, 192, 192])
        x = activation(x)

    with tf.variable_scope('mlp3-1'):
        x = conv(x, phase, [1, 1, 192, 192])
        x = activation(x)

    with tf.variable_scope('mlp3-2'):
        x = conv(x, phase, [1, 1, 192, 10])
        x = activation(x)

    with tf.name_scope('global_avg_pool'):
        x  = global_avg_pool(x, 7, 1)
        output  = tf.reshape(x,[-1,10])

    return output
    
class NiN_Model():
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    self.y_input = tf.placeholder(tf.int64, shape=None)
    self.bs = tf.placeholder(tf.int32, shape=None)
      
    self.y = inference(self.x_input, phase=False, keep_prob=1.0)
    self.predictions = tf.argmax(self.y, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    
    self.corr_pred = self.correct_prediction
    
    self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_input)
    self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
    self.grad = tf.gradients(self.xent, self.x_input)[0]