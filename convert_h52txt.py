import argparse

from pathlib import Path
import numpy as np

from tensorflow import int16 as tfInt16
from tensorflow import transpose as tfTranspose
from tensorflow import reshape as tfReshape
from tensorflow import reverse as tfReverse

from utils import RoundPower24Store
from test_conv import OneConvNet
from quantization import RoundPower2Exp

import pdb


# def RoundPower2(x, k=4):
  # bound = np.power(2.0, k - 1)
  # min_val = np.power(2.0, -bound + 1.0)
  # s = np.sign(x)
  # # # Check README.md for why `x` need to be divided by `8`
  # # x = np.clip(np.absolute(x / 8), min_val, 1.0)
  # x = np.clip(np.absolute(x), min_val, 1.0)
  # p = -1 * np.around(np.log(x) / np.log(2.))
  # if s == -1:
    # p = 0x8 + p.astype(np.int8)
  # return p.astype(np.int8)


# def Round2Fixed(x, integer=16, k=32):
  # assert integer >= 1, integer
  # fraction = k - integer
  # bound = np.power(2.0, k - 1)
  # n = np.power(2.0, fraction)
  # min_val = -bound
  # max_val = bound
  # # ??? Is this round function correct
  # x_round = np.around(x * n)
  # clipped_value = np.clip(x_round, min_val, max_val).astype(np.int32)
  # return clipped_value


# def Store4DBinConvert(weight, f):
  # weight_h = weight.shape[0]
  # weight_w = weight.shape[1]
  # weight_c = weight.shape[2]
  # weight_f = weight.shape[3]
  # weight_p = 64
  # # weight_p = 1

  # num_pixel = weight_f * weight_c * weight_h * weight_w
  # step_f = weight_c * weight_h * weight_w * weight_p       # Step for filter
  # step_c = weight_h * weight_w * weight_p                  # Step for col
  # step_h = weight_w * weight_p                             # Step for channel
  # step_w = weight_p                                        # Step for row

  # data_weight = np.zeros(int(num_pixel), dtype=np.int16)
  # for b in range(int(weight_f / weight_p)):
    # for k in range(weight_c):
      # for row in range(weight_h):
        # for col in range(weight_w):
          # for p in range(weight_p):
            # index = int(
                # b * step_f + k * step_c + row * step_h + col * step_w + p)
            # ######################
            # ### Shift          ###
            # ######################
            # data_shift = \
                # RoundPower24Store(weight[row, col, k, b * weight_p + p], 4)
            # data_weight[index] = data_shift.numpy().astype(np.int16)

            # ######################
            # ### Multiplication ###
            # ######################
            # # data_weight[index] = \
                # # Fix2Int(
                    # # Round2Fixed(weight[row, col, k, b * weight_p + p], 4, 12),
                    # # 4, 12).numpy().astype(np.int16)

  # if args.bin:
    # for i in range(int(num_pixel / 2)):
      # temp = data_weight[2 * i]
      # data_weight[2 * i] = data_weight[2 * i + 1]
      # data_weight[2 * i + 1] = temp

    # for npiter in np.nditer(data_weight):
      # f.write(npiter)

  # else:
    # for j in range(int(num_pixel/args.paral_in)):
      # for i in range(args.paral_in):
        # pixel = data_weight[j*args.paral_in+i]
        # if pixel >= 0:
          # pixel_str = '{:0>4x}'.format(pixel)
        # else:
          # pixel_str = '{:0>4x}'.format(0x10000+pixel)
        # f.write(pixel_str)

      # f.write('\n')


# def Store1DBinConvert(weight, f):
  # weight_c = weight.shape[0]

  # data_weight = np.zeros(int(weight_c), dtype=np.int32)
  # for b in range(int(weight_c)):
    # data_weight[b] = Fix2Int(Round2Fixed(weight[b], 8, 24), 8, 24)

  # for npiter in np.nditer(data_weight):
    # f.write(npiter)


# def StoreWeightBinConvert(weight, f):
  # if len(weight.shape) == 4:
    # print('[INFO][utils.py] Store 4D tensor {}'.format(weight.name))
    # Store4DBinConvert(weight, f)
  # elif len(weight.shape) == 1:
    # print('[INFO][utils.py] Store 1D tensor {}'.format(weight.name))
    # Store1DBinConvert(weight, f)
  # else:
    # print('[INFO][utils.py] Wrong weight shape!!! '
        # 'Variable name is {}'.format(weight.name))


def ConvertTensor(tensor, paral):
  '''
  Brief:
    Return a flattened 1D tensorflow tensor.
  '''
  if len(tensor.shape) == 4:
    print('[INFO][utils.py] Convert 4D tensor {}'.format(tensor.name))
    height = tensor.shape[0]
    width = tensor.shape[1]
    in_channels = tensor.shape[2]
    tensor_paral = tfReshape(
        tensor, [height, width, in_channels, -1, paral])
    # tensor_reorder = tfTranspose(tensor_paral, [1, 2, 4, 3, 0])
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = 'Store model as .dat. Default running command is'
      '`python convert_h52txt.py`')
  parser.add_argument(
      '--paral_in', default = 8, type = int,
      help = 'Parallelism degree of input. No need to change in most case.')
  parser.add_argument(
      '--paral_w', default = 64, type = int,
      help = 'Parallelism degree of weight. No need to change in most case.')
  parser.add_argument(
      '--img_w', default = 56, type = int,
      help = 'Image width and height.')
  parser.add_argument(
      '--img_ch', default = 256, type = int,
      help = 'Number of image channels.')
  parser.add_argument(
      '--input_file', default = 'weight_56_256.h5',
      help = 'Input file name.')
  parser.add_argument(
      '--output_file', default = 'weight_56_256_shift_process_16bit.dat',
      help = 'Output file name.')
  parser.add_argument(
      '--bin', dest='bin', action='store_true',
      help = 'Output binary for on-board test.')
  parser.add_argument(
      '--txt', dest='bin', action='store_false',
      help = 'Output text for simulation.')
  parser.set_defaults(bin=True)
  args = parser.parse_args()


  ckpt_path = Path('.') / 'ckpt' / args.input_file
  output_path = Path('.') / 'ckpt_dat' / args.output_file

  input_tensor_shape = (None, args.img_w, args.img_w, args.img_ch)
  # Don't change. Quantization will be executed in store function.
  model = OneConvNet('ste', 4, 32, 4, 32)
  model.build(input_tensor_shape)
  model.load_weights(str(ckpt_path))

  if args.bin:
    print('[INFO][convert_h52txt.py] Open file as binary.')
    file_mode = 'wb'
  else:
    print('[INFO][convert_h52txt.py] Open file as text.')
    file_mode = 'w'

  with open(str(output_path), mode = file_mode) as f:
    for i, weight in enumerate(model.weights):
      weight_1d = ConvertTensor(weight, args.paral_w)
      if weight_1d == None:
        continue

      if args.bin:
        print('[INFO][convert_h52txt.py] Store weight as binary.')
        weight_reshape = \
            tfReshape(tfReverse(tfReshape(weight_1d, [-1, 2]), [1]), [-1])
        weight_quantize = RoundPower24Store(weight_reshape)
        for npiter in np.nditer(weight_quantize.numpy().astype(np.int16)):
          f.write(npiter)
