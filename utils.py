from tensorflow.keras.layers import BatchNormalization
from tensorflow import math as tfMath
from tensorflow import transpose as tfTranspose
from tensorflow import reshape as tfReshape
from tensorflow import reverse as tfReverse

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
    print('[INFO][utilt.py][GenerateRoundFn] '
        'GenerateRoundFn for shift. w_bit is {}'.format(w_bit))
    return RoundPower2Wrap
  elif flag_shift == 'mul':
    print('[INFO][utilt.py][GenerateRoundFn] '
        'GenerateRoundFn for multiplication. '
        'w_int is {}. w_bit is {}.'.format(w_int, w_bit))
    return Round2FixedWrap
  else:
    return None


def ConvertTensor(tensor, axis, paral):
  '''
  Brief:
    Return a flattened 1D tensorflow tensor.

    For weight tensor, axis should be set to [3, 2, 0, 1, 4], which is
      (row, col, in_channels, out_channels/paral, paral) ->
      (out_channels/paral, in_channels, row, col, paral)

    For image tensor, axis should be set to [2, 0, 3, 1, 4]
      (row, col, 1, ch/paral, paral) -> (1, row, ch/paral, col, paral)
  '''
  if len(tensor.shape) == 4:
    # print('[INFO][utils.py] Convert 4D tensor {}'.format(tensor.name))
    print('[INFO][utils.py] Convert 4D tensor.')
    height = tensor.shape[0]
    width = tensor.shape[1]
    dim = tensor.shape[2]
    tensor_paral = tfReshape(
        tensor, [height, width, dim, -1, paral])
    # tensor_transpose = tfTranspose(tensor_paral, [3, 2, 0, 1, 4])
    tensor_transpose = tfTranspose(tensor_paral, axis)
    tensor_1d = tfReshape(tensor_transpose, [-1])
  elif len(tensor.shape) == 1:
    # print('[INFO][utils.py][ConvertTensor] 1D tensor {}, '
        # 'no need to be converted'.format(tensor.name))
    print('[INFO][utils.py][ConvertTensor] 1D tensor, no need to be converted')
    tensor_1d = tensor
  else:
    print('[INFO][utils.py][ConvertTensor] Wrong tensor shape!!!')
    tensor_1d = None

  return tensor_1d


def Store4DTensor(tensor, output_path, integer, width):
  # Activation tensor.
  # (batch, row, col, channels) -> (row, col, batch, channels)
  img_transpose = tfTranspose(tensor, perm = [1, 2, 0, 3])

  # (row, col, 1, ch, 1) -> (1, 1, ch, row, col)
  data_1d = ConvertTensor(img_transpose, [2, 4, 3, 0, 1], 1)

  data_tensor = Round2Fixed(data_1d, integer, width)

  with open(str(output_path), 'wb') as f:
    for npiter in np.nditer(data_tensor.numpy().astype(np.float32)):
      f.write(npiter)

def Store2DTensor(tensor, output_path, integer, width):

  tensor_quantize = Round2Fixed(tensor, integer, width)

  with open(str(output_path), 'wb') as f:
    for npiter in np.nditer(tensor_quantize.numpy().astype(np.float32)):
      f.write(npiter)

def StoreFormatTxt(it, f):
  for npiter in it:
    if it.multi_index[1] == 0:
      data_str = ''

    if npiter >= 0:
      pixel_str = '{:0>4x}'.format(npiter)
    else:
      pixel_str = '{:0>4x}'.format(0x10000+npiter)

    data_str = data_str + pixel_str

    if it.multi_index[1] == 7 or it.finished:
      f.write(data_str + '\n')
