import argparse

from pathlib import Path
import numpy as np

from tensorflow import int16 as tfInt16
from tensorflow import reshape as tfReshape
from tensorflow import reverse as tfReverse

import utils
from test_conv import OneConvNet
from models.yolo_tiny import GenerateModel
import pdb


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = 'Store model as .dat. Default running command is'
      '`python convert_h52txt.py`')
  parser.add_argument(
      '--paral_w', default = 64, type = int,
      help = 'Parallelism degree of weight. No need to change in most case.')

  parser.add_argument(
      '--quantize_w_method', default = 'mul',
      help = 'Choose quantization mode. '
      '`shfit` for shift and `mul` for multiply')
  parser.add_argument(
      '--quantize_w_integer', default = 4, type = int,
      help = 'Specify integer data width of weight.')
  parser.add_argument(
      '--quantize_w', default = 12, type = int,
      help = 'Specify data width of weight.')
  parser.add_argument(
      '--quantize_b_method', default = 'mul',
      help = 'Choose quantization mode. '
      '`shfit` for shift and `mul` for multiply')
  parser.add_argument(
      '--quantize_b_integer', default = 8, type = int,
      help = 'Specify integer data width of bias tensor.')
  parser.add_argument(
      '--quantize_b', default = 24, type = int,
      help = 'Specify data width of bias tensor.')

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
      '--output_file_weight', default = 'None',
      help = 'Output file name.')
  parser.add_argument(
      '--output_file_bias', default = 'None',
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
  output_weight_path = Path('.') / 'ckpt_dat' / args.output_file_weight
  output_bias_path = Path('.') / 'ckpt_dat' / args.output_file_bias

  input_tensor_shape = (None, args.img_w, args.img_w, args.img_ch)
  # Don't change. Quantization will be executed in store function.
  # model = OneConvNet('shift', 4, 32, 4, 32)
  model = GenerateModel('shift', 4, 4, 3, 8)
  model.build(input_tensor_shape)
  model.load_weights(str(ckpt_path))

  if args.bin:
    print('[INFO][convert_h52txt.py] Open file as binary.')
    file_mode = 'wb'
  else:
    print('[INFO][convert_h52txt.py] Open file as text.')
    file_mode = 'w'

  QuantizeFuncWeight = utils.GenerateRoundFn(
      args.quantize_w_integer, args.quantize_w, args.quantize_w_method)

  QuantizeFuncBias = utils.GenerateRoundFn(
      args.quantize_b_integer, args.quantize_b, args.quantize_b_method)

  '''
  Tensor process table

                on-board (bin)               simulation (txt)
  weight        Swap adjacent elements       tfReshape
                QuantizeFuncWeight           QuantizeFuncWeight
                np.int16                     utils.StoreFormatTxt
                fw                           np.int16
                                             fw

  bias          QuantizeFuncBias             QuantizeFuncBias
                np.int32                     utils.StoreFormatTxt
                fb                           np.int32
                                             fb
  '''
  with open(str(output_weight_path), mode = file_mode) as fw, \
      open(str(output_bias_path), mode = file_mode) as fb:
    for i, weight in enumerate(model.weights):
      weight_1d = utils.ConvertTensor(weight, [3, 2, 0, 1, 4], args.paral_w)
      if weight_1d is None:
        continue

      if args.bin and 'weights' in weight.name:
        print('[INFO][convert_h52txt.py] '
            'Store weight {} as binary.'.format(weight.name))
        weight_reshape = \
            tfReshape(tfReverse(tfReshape(weight_1d, [-1, 2]), [1]), [-1])
        weight_quantize = QuantizeFuncWeight(weight_reshape)
        for npiter in np.nditer(weight_quantize.numpy().astype(np.int16)):
          fw.write(npiter)

      elif args.bin and 'bias' in weight.name:
        print('[INFO][convert_h52txt.py] '
            'Store weight {} as binary.'.format(weight.name))
        weight_quantize = QuantizeFuncBias(weight_1d)
        for npiter in np.nditer(weight_quantize.numpy().astype(np.int32)):
          fb.write(npiter)

      elif not args.bin and 'weights' in weight.name:
        print('[INFO][convert_h52txt.py] '
            'Store weight {} as text.'.format(weight.name))
        weight_reshape = tfReshape(weight_1d, [-1, 8])
        weight_quantize = QuantizeFuncWeight(weight_reshape)
        it = np.nditer(
            weight_quantize.numpy().astype(np.int16), flags=['multi_index'])
        utils.StoreFormatTxt(it, '{:0>4x}', 0x10000, 8, fw)

      elif not args.bin and 'bias' in weight.name:
        print('[INFO][convert_h52txt.py] '
            'Store bias {} as text.'.format(weight.name))
        weight_reshape = tfReshape(weight_1d, [-1, 4])
        weight_quantize = QuantizeFuncBias(weight_reshape)
        it = np.nditer(
            weight_quantize.numpy().astype(np.int32), flags=['multi_index'])
        utils.StoreFormatTxt(it, '{:0>8x}', 0x100000000, 4, fb)

      else:
        print('[ERROR][convert_h52txt.py] '
            'Wrong store formant or wrong variable type.')
