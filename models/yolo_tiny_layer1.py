import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers

from nn_utils import QConv2D
from utils import QStore4DTensor, QStore2DTensor

class YoloTiny(Model):
  def __init__(
      self,
      quantize,
      quantize_w,
      quantize_x):

    super().__init__(name = '')
    self.quantize = quantize
    self.quantize_w = quantize_w
    self.quantize_x = quantize_x
    self.alpha = Variable(0., trainable = False)

    self.conv1 = QConv2D(
        64, 3, 1,
        quantize = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = 0,
        use_bias = True,
        alpha = self.alpha)

  def call(self, input_tensor):
    x = input_tensor

    x = self.conv1(x)

    return x

def GenerateModel(quantize, quantize_w, quantize_x):
  return YoloTiny(quantize, quantize_w, quantize_x)
