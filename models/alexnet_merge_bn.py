import tensorflow as tf
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D

from nn_utils import QConv2D
from quantization import QuantilizeFn 
import pdb

class AlexNet(tf.keras.Model):
  def __init__(
      self,
      weight_decay,
      class_num,
      quantize   = 'full',
      quantize_w = 32,
      quantize_x = 32,
      num_epochs   = 250):
    super(AlexNet, self).__init__(name = '')
    self.weight_decay = weight_decay
    self.quantize   = quantize
    self.quantize_w = quantize_w
    self.quantize_x = quantize_x
    # self.alpha        = tf.Variable(0., trainable = False, name = 'alpha')
    self.alpha        = 0
    self.num_epochs   = num_epochs

    self.conv1 = QConv2D(
        64, 11, 4,
        padding      = 'VALID',
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = True,
        alpha        = self.alpha)

    self.conv2 = QConv2D(
        192, 5,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = True,
        alpha        = self.alpha)

    self.conv3 = QConv2D(
        384, 3,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = True,
        alpha        = self.alpha)

    self.conv4 = QConv2D(
        384, 3,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = True,
        alpha        = self.alpha)

    self.conv5 = QConv2D(
        256, 3,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = True,
        alpha        = self.alpha)

    self.conv6 = QConv2D(
        4096, 5,
        padding      = 'VALID',
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = True,
        alpha        = self.alpha)

    self.conv7 = QConv2D(
        4096, 1,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = True,
        alpha        = self.alpha)

    self.conv8 = QConv2D(
        class_num, 1,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = True,
        alpha        = self.alpha)

  def call(self, input_tensor):
    x = input_tensor
    # Layer 1
    x = self.conv1(x)
    x = Activation('relu')(x)

    # Layer 2
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)

    # Layer 3
    x = self.conv2(x)
    x = Activation('relu')(x)

    # Layer 4
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)

    # Layer 5
    x = self.conv3(x)
    x = Activation('relu')(x)

    # Layer 6
    x = self.conv4(x)
    x = Activation('relu')(x)

    # Layer 7
    x = self.conv5(x)
    x = Activation('relu')(x)

    # Layer 8
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)

    # Layer 9
    x = self.conv6(x)
    x1 = Activation('relu')(x)

    # Layer 10
    x = self.conv7(x1)
    x = Activation('relu')(x)

    # Layer 11
    x = self.conv8(x)
    x = Flatten()(x)

    # x = Activation('softmax')(x)

    # return x
    return x1

class_num    = 1000
# quantize   = 'full'
# quantize_w = 32
# quantize_x = 32
quantize   = 'ste'
quantize_w = 8
quantize_x = 8
weight_decay = 0.0005
num_epochs   = 250

model_merge_bn = AlexNet(
    weight_decay, class_num, quantize = quantize,
    quantize_w = quantize_w, quantize_x = quantize_x,
    num_epochs = num_epochs)
