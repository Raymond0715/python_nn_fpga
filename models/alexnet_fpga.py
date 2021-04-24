import tensorflow as tf
from tensorflow.keras.layers import Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization

from nn_utils import QConv2D

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
        use_bias     = False,
        alpha        = self.alpha)
    self.bn1 = BatchNormalization()
    self.conv2 = QConv2D(
        192, 5,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = False,
        alpha        = self.alpha)
    self.bn2 = BatchNormalization()
    self.conv3 = QConv2D(
        384, 3,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = False,
        alpha        = self.alpha)
    self.bn3 = BatchNormalization()
    self.conv4 = QConv2D(
        384, 3,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = False,
        alpha        = self.alpha)
    self.bn4 = BatchNormalization()
    self.conv5 = QConv2D(
        256, 3,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = False,
        alpha        = self.alpha)
    self.bn5 = BatchNormalization()
    self.conv6 = QConv2D(
        4096, 5,
        padding      = 'VALID',
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = False,
        alpha        = self.alpha)
    self.bn6 = BatchNormalization()
    self.conv7 = QConv2D(
        4096, 1,
        quantize   = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = False,
        alpha        = self.alpha)
    self.bn7 = BatchNormalization()
    self.conv8 = QConv2D(
        class_num, 1,
        quantize   = self.quantize,
        # quantize   = 'full',
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = weight_decay,
        use_bias     = False,
        alpha        = self.alpha)
    self.bn8 = BatchNormalization()

  def call(self, input_tensor):
    x = input_tensor
    x = self.conv1(x)
    x = self.bn1(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = Activation('relu')(x)

    x = self.conv4(x)
    x = self.bn4(x)
    x = Activation('relu')(x)

    x = self.conv5(x)
    x = self.bn5(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = 3, strides = 2)(x)

    x = self.conv6(x)
    x = self.bn6(x)
    x = Activation('relu')(x)

    x = self.conv7(x)
    x = self.bn7(x)
    x = Activation('relu')(x)

    x = self.conv8(x)
    x = self.bn8(x)
    x = Flatten()(x)

    x = Activation('softmax')(x)

    return x

class_num    = 1000
quantize   = 'full'
quantize_w = 32
quantize_x = 32
# quantize   = 'ste'
# quantize_w = 8
# quantize_x = 8
weight_decay = 0.0005
num_epochs   = 250

model = AlexNet(
    weight_decay, class_num, quantize = quantize,
    quantize_w = quantize_w, quantize_x = quantize_x,
    num_epochs = num_epochs)
