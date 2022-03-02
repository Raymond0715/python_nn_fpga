import argparse

from pathlib import Path
import numpy as np

from tensorflow import int16 as tfInt16
from tensorflow import transpose as tfTranspose
from tensorflow import reshape as tfReshape
from tensorflow import reverse as tfReverse

from utils import GenerateRoundFn, ConvertTensor
from test_conv import OneConvNet


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
      '--quantize', default = 'shift',
      help = 'Choose quantization mode. '
      '`shfit` for shift and `mul` for multiply')
  parser.add_argument(
      '--quantize_w_integer', default = 4, type = int,
      help = 'Specify integer data width of weight.')
  parser.add_argument(
      '--quantize_w', default = 12, type = int,
      help = 'Specify data width of weight.')

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

  QuantizeFunc = GenerateRoundFn(
      args.quantize_w_integer, args.quantize_w, args.quantize)

  with open(str(output_path), mode = file_mode) as f:
    for i, weight in enumerate(model.weights):
      weight_1d = ConvertTensor(weight, args.paral_w)
      if weight_1d is None:
        continue

      if args.bin:
        print('[INFO][convert_h52txt.py] '
            'Store weight {} as binary.'.format(weight.name))
        weight_reshape = \
            tfReshape(tfReverse(tfReshape(weight_1d, [-1, 2]), [1]), [-1])
        weight_quantize = QuantizeFunc(weight_reshape)
        for npiter in np.nditer(weight_quantize.numpy().astype(np.int16)):
          f.write(npiter)

      else:
        print('[INFO][convert_h52txt.py] '
            'Store weight {} as text.'.format(weight.name))
        weight_reshape = tfReshape(weight_1d, [-1, 8])
        weight_quantize = QuantizeFunc(weight_reshape)
        it = np.nditer(
            weight_quantize.numpy().astype(np.int16), flags=['multi_index'])
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
