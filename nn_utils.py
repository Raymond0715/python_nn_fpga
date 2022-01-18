# import pdb

import tensorflow as tf
from tensorflow.keras import regularizers
from quantization import QuantilizeFn, tangent

class QConv2D(tf.keras.layers.Layer):
  def __init__(
      self,
      kernel_depth,
      kernel_size,
      strides = [1, 1],
      padding = 'SAME',
      quantize = 'full',
      quantize_w = 32,
      quantize_x = 32,
      weight_decay = 0.0005,
      use_bias = True,
      name = None,
      alpha = None):

    # super(QConv2D, self).__init__()
    super().__init__()
    self.kernel_depth = kernel_depth
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.quantize = quantize
    self.weight_decay = weight_decay
    self.use_bias = use_bias
    if self.quantize != 'full':
      self.QuantilizeWeight, self.QuantizeActivation = \
          QuantilizeFn(quantize_w, quantize_x)
      self.alpha = alpha # For nature gradient quantilization
    else:
      # print('[DEBUG][nn_utils.py] init QConv2D full')
      pass

  def build(self, input_shape):
    self.filters = self.add_weight(
        shape = [
          self.kernel_size,
          self.kernel_size,
          int(input_shape[-1]),
          self.kernel_depth],
        initializer = tf.keras.initializers.GlorotUniform(),
        regularizer = regularizers.l2(self.weight_decay),
        name = 'weights')
        # regularizer = regularizers.l2(self.weight_decay))

    if self.use_bias:
      self.bias = self.add_weight(
          shape = [self.kernel_depth],
          # initializer = tf.keras.initializers.Zeros(),
          # initializer = 'random_normal',
          initializer = tf.keras.initializers.GlorotUniform(),
          name = 'bias')

  def call(self, input_tensor):
    if self.quantize != 'full':
      # print('[DEBUG][nn_utils.py] QConv2D call ng')
      filters = tf.clip_by_value(self.filters, -1, 1)
      quantize_filters = self.QuantilizeWeight(filters)
      # quantize_filters = self.QuantilizeWeight(self.filters)
      filters = tangent(self.filters, quantize_filters, self.alpha)
      input_tensor_quantize = self.QuantizeActivation(input_tensor)
      input_tensor = tangent(
          input_tensor, input_tensor_quantize, self.alpha)

      if self.use_bias:
        bias = self.QuantizeActivation(self.bias)
    else:
      # print('[DEBUG][nn_utils.py] QConv2D call full')
      filters = self.filters
      if self.use_bias:
        bias = self.bias

    output = tf.nn.conv2d(
        input_tensor, filters, self.strides, self.padding)

    if self.use_bias:
      output = tf.nn.bias_add(output, bias)

    # if self.quantize != 'full':
      # output = self.QuantizeActivation(output)

    return output
