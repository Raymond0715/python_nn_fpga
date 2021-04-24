from tensorflow.keras.layers import BatchNormalization

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


def Store4DTensor(tensor, output_path):
  tensor_h = tensor.shape[1]
  tensor_w = tensor.shape[2]
  tensor_c = tensor.shape[3]
  tensor_b = tensor.shape[0]

  step_b = tensor_h * tensor_w * tensor_c
  step_c = tensor_h * tensor_w
  step_h = tensor_h
  step_w = 1

  num_pixel   = tensor_h * tensor_w * tensor_c * tensor_b
  tensor_np   = tensor.numpy()
  data_tensor = np.zeros(num_pixel, dtype=np.float32)

  for b in range(tensor_b):
    for row in range(tensor_h):
      for col in range(tensor_w):
        for k in range(tensor_c):
          index = b * step_b + k * step_c + row * step_h + col * step_w
          data_tensor[index] = tensor_np[b, row, col, k]

  f = open(str(output_path), 'w')
  for i, npiter in enumerate(np.nditer(data_tensor)):
    # f.write(str(npiter) + ' ')
    f.write(str(npiter) + ', ')
    if (i + 1) % 9 == 0:
      f.write('\n')

  if num_pixel % 9 != 0:
    f.write('\n')

  f.close()


def Store4D(weight, f):
  weight_h = weight.shape[0]
  weight_w = weight.shape[1]
  weight_c = weight.shape[2]
  weight_f = weight.shape[3]

  step_f   = weight_h * weight_w * weight_c  # Step for filter
  step_c   = weight_h * weight_w             # Step for channel
  step_h   = weight_h                        # Step for col
  step_w   = 1                               # Step for row

  num_pixel = weight_f * weight_c * weight_h * weight_w
  weight_np   = weight.numpy()
  data_weight = np.zeros(num_pixel, dtype=np.float32)
  for b in range(weight_f):
    for row in range(weight_h):
      for col in range(weight_w):
        for k in range(weight_c):
          index = b * step_f + k * step_c + row * step_h + col * step_w
          data_weight[index] = weight_np[row, col, k, b]

  for npiter in np.nditer(data_weight):
    f.write(str(npiter) + ' ')


def Store2DTensor(tensor, output_path):
  f = open(str(output_path), 'w')

  tensor_len  = tensor.shape[1]
  tensor_np   = tensor.numpy()
  for i in range(tensor_len):
    f.write(str(tensor_np[0,i]) + ' ')
    if (i + 1) % 9 == 0:
      f.write('\n')

  if tensor_len % 9 != 0:
    f.write('\n')

  f.close()


def Store1D(bn, f):
  bn_len  = bn.shape[0]
  bn_np   = bn.numpy()
  for i in range(bn_len):
    f.write(str(bn_np[i]) + ' ')


def StoreWeight(weight, f):
  if len(weight.shape) == 4:
    print('[INFO][store_weight.py] Store 4D tensor {}'.format(weight.name))
    Store4D(weight, f)
  elif len(weight.shape) == 1:
    print('[INFO][store_weight.py] Store 1D tensor '
        'weight {}'.format(weight.name))
    Store1D(weight, f)
  else:
    print('[INFO][store_weight.py] Wrong weight shape!!! '
        'Variable name is {}'.format(weight.name))


def Store4DBin(weight, f):
  weight_h = weight.shape[0]
  weight_w = weight.shape[1]
  weight_c = weight.shape[2]
  weight_f = weight.shape[3]

  step_f = weight_h * weight_w * weight_c  # Step for filter
  step_c = weight_h * weight_w             # Step for channel
  step_h = weight_h                        # Step for col
  step_w = 1                               # Step for row

  num_pixel = weight_f * weight_c * weight_h * weight_w
  weight_np   = weight.numpy()
  data_weight = np.zeros(num_pixel, dtype=np.float32)
  for b in range(weight_f):
    for row in range(weight_h):
      for col in range(weight_w):
        for k in range(weight_c):
          index = b * step_f + k * step_c + row * step_h + col * step_w
          data_weight[index] = weight_np[row, col, k, b]

  for npiter in np.nditer(data_weight):
    # f.write(str(npiter) + ' ')
    f.write(npiter)


def Store1DBin(bn, f):
  bn_len  = bn.shape[0]
  bn_np   = bn.numpy()
  for i in range(bn_len):
    # f.write(str(bn_np[i]) + ' ')
    f.write(bn_np[i])


def StoreWeightBin(weight, f):
  if len(weight.shape) == 4:
    print('[INFO][utils.py] Store 4D tensor {}'.format(weight.name))
    Store4DBin(weight, f)
  elif len(weight.shape) == 1:
    print('[INFO][utils.py] Store 1D tensor '
        'weight {}'.format(weight.name))
    Store1DBin(weight, f)
  else:
    print('[INFO][utils.py] Wrong weight shape!!! '
        'Variable name is {}'.format(weight.name))
