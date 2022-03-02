from tensorflow.keras.layers import BatchNormalization
from tensorflow import math as tfMath

from quantization import RoundPower2Exp, Round2Fixed
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


def Fix2Int(x, integer=16, k=32):
  fraction = k - integer
  n = tfMath.pow(2.0, fraction)
  x_quantize_int = x * n

  return x_quantize_int


def GenerateRoundFn(w_int, w_bit, flag_shift):

  def Round2FixedWrap(x):
    x_quantize = Round2Fixed(x, w_int, w_bit)
    x_int = Fix2Int(x_quantize, w_int, w_bit)

    return x_int

  def RoundPower2Wrap(x):
    s, p = RoundPower2Exp(x, w_bit)
    p_minus = p * -1
    s_pro = (s - 1) * -4
    p_pro = s_pro + p_minus

    return p_pro

  if flag_shift == 'shift':
    print('[INFO][utilt.py] '
        'GenerateRoundFn for shift. w_bit is {}'.format(w_bit))
    return RoundPower2Wrap
  elif flag_shift == 'mul':
    print('[INFO][utilt.py] '
        'GenerateRoundFn for multiplication. '
        'w_int is {}. w_bit is {}.'.format(w_int, w_bit))
    return Round2FixedWrap
  else:
    return None


def ConvertTensor(tensor, paral):
  '''
  Brief:
    Return a flattened 1D tensorflow tensor.
    Especially convert tf-style weight to FPGA-style weight, which is
      (row, col, in_channels, output_channels) ->
      (output_channels/paral, in_channels, row, col, paral)
  '''
  if len(tensor.shape) == 4:
    print('[INFO][utils.py] Convert 4D tensor {}'.format(tensor.name))
    height = tensor.shape[0]
    width = tensor.shape[1]
    in_channels = tensor.shape[2]
    tensor_paral = tfReshape(
        tensor, [height, width, in_channels, -1, paral])
    tensor_reorder = tfTranspose(tensor_paral, [3, 2, 0, 1, 4])
    tensor_1d = tfReshape(tensor_reorder, [-1])
  elif len(tensor.shape) == 1:
    print('[INFO][utils.py] 1D tensor {}, '
        'no need to be converted'.format(tensor.name))
    tensor_1d = tensor
  else:
    print('[INFO][utils.py] Wrong tensor shape!!! '
        'Variable name is {}'.format(tensor.name))
    tensor_1d = None

  return tensor_1d
