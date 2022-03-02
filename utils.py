from tensorflow.keras.layers import BatchNormalization
from tensorflow import math as tfMath

from quantization import Round2Fixed, RoundPower2Exp
import nn_utils
import numpy as np


def MergeBN(dst, src):
  merge_list = []
  for i in range(len(src.layers)):
    layer = src.get_layer(index = i)
    if isinstance(layer, nn_utils.QConv2D):
      merge_list.append(layer.weights[0].numpy())
      if len(layer.weights) == 2:
        merge_list.append(layer.weights[1].numpy())
    elif isinstance(layer, BatchNormalization):
      merge_weight = \
          merge_list[-1] \
          * layer.weights[0].numpy()[None, None, None, :] \
          / np.sqrt(
              layer.weights[3].numpy() + 0.001)[None, None, None, :]
      merge_list[-1] = merge_weight

      bias = layer.weights[1].numpy() \
          - layer.weights[0].numpy() \
          * layer.weights[2].numpy() \
          / np.sqrt(layer.weights[3].numpy() + 0.001)
      merge_list.append(bias)
    else:
      pass

  dst.set_weights(merge_list)

  return dst


def RoundPower24Store(x, k=4):
  s, p = RoundPower2Exp(x, k)
  p_minus = p * -1
  s_pro = (s - 1) * -4
  p_pro = s_pro + p_minus

  # if s == -1:
    # # p = 0x8 + p.astype(tfInt16)
    # p = 0x8 + p

  return p_pro


def Fix2Int(x, integer=16, k=32):
  # x_quantize_float = Round2Fixed(x, integer, k)
  fractrion = k - integer
  n = tfMath.pow(2, fraction)
  x_quantize_int = x_quantize_float * n

  return x_quantize_int
