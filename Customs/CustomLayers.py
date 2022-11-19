import numpy as np
from tensorflow import keras
import tensorflow as tf
import keras.layers as layers
import keras.initializers as initializers


#Time Embedding from paper "Time2Vec: Learning a Vector Representation of Time(2019)"
class Time2Vector(layers.Layer):
  def __init__(self, seq_len, kernel_initializer="glorot_uniform", bias_initializer="zeros", **kwargs):
    super(Time2Vector, self).__init__(**kwargs)
    self.seq_len = seq_len
    self.kernel_init = initializers.get(kernel_initializer)
    self.bias_init = initializers.get(bias_initializer)

  def build(self, input_shape):
    assert len(input_shape) == 3

    w_init = self.kernel_init
    b_init = self.bias_init
    
    self.w_linear = tf.Variable(name='w_linear', 
                                initial_value=w_init(shape=(self.seq_len,), dtype=tf.float32),
                                trainable=True)
    
    self.b_linear = tf.Variable(name='b_linear', 
                                initial_value=b_init(shape=(self.seq_len,), dtype=tf.float32),
                                trainable=True)
    
    self.w_periodic = tf.Variable(name='w_periodic', 
                                initial_value=w_init(shape=(self.seq_len,), dtype=tf.float32),
                                trainable=True)

    self.b_periodic = tf.Variable(name='b_periodic', 
                                initial_value=b_init(shape=(self.seq_len,), dtype=tf.float32),
                                trainable=True)

  def call(self, inputs):
    '''Calculate linear and periodic time features'''
    x = tf.math.reduce_mean(inputs, axis=-1) 
    x_linear = tf.multiply(x, self.w_linear) + self.b_linear # Linear time embedding shape = (batch, seq_len)
    x_linear = tf.expand_dims(x_linear, axis=-1) # shape =  (batch, seq_len, 1)
    
    #Linear time embedding 
    x_periodic = tf.math.sin(tf.multiply(x, self.w_periodic) + self.b_periodic) #shape = (batch, seq_len)
    x_periodic = tf.expand_dims(x_periodic, axis=-1) # shape= (batch, seq_len, 1)

    return tf.concat([x_linear, x_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config


#Basic Attention from paper "Hierarchical Attention Networks for Document Classification(2016)"
class AttentionLayer(layers.Layer):
    def __init__(self, attention_dim, kernel_initializer="glorot_uniform", bias_initializer="zeros", supports_masking=True, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
        self.kernel_init = initializers.get(kernel_initializer)
        self.bias_init = initializers.get(bias_initializer)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        assert len(input_shape) == 3

        w_init = self.kernel_init
        b_init = self.kernel_init

        self.w = tf.Variable(name="w",
                            initial_value=w_init(shape=(input_shape[-1], self.attention_dim), dtype=tf.float32),
                            trainable=True)
        
        self.u = tf.Variable(name="u", 
                            initial_value=w_init(shape=(self.attention_dim, 1), dtype=tf.float32), 
                            trainable=True)

        self.b = tf.Variable(name="b", 
                            initial_value=b_init(shape=(self.attention_dim, ), dtype=tf.float32),
                            trainable=True)
   

    def call(self, inputs):
        
        x0 = inputs #shape = (batch, seq_len, dim)

        x = tf.nn.tanh(tf.add(tf.matmul(x0, self.w), self.b)) #tanh(xW + b) shape= (batch, seq_len, attention_dim)
        x = tf.matmul(x, self.u) # xU shape = (batch, seq_len, 1)

        x = tf.nn.softmax(x, axis=1) #softmax along sequence

        weighted_input = x * x0

        output = tf.reduce_sum(weighted_input, axis=1)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({'attention_dim': self.attention_dim})
        return config