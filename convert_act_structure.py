import argparse
from pathlib import Path
import numpy as np

from tensorflow import convert_to_tensor as tfConverttoTensor
from tensorflow import transpose as tfTranspose
from tensorflow import reshape as tfReshape
from tensorflow import float32 as tfFloat32
from tensorflow import reverse as tfReverse

import utils


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = 'For FPGA, convert activation data layout')

  # Input
  parser.add_argument(
      '--paral_in', default = 8, type = int,
      help = 'Input degree of parallelism. No need to change in most case.')
  parser.add_argument(
      '--input', default = 'img_56_256.bin',
      help = 'Input file name.')
  parser.add_argument(
      '--img_size', default = 56, type = int,
      help = 'Input image size.')
  parser.add_argument(
      '--img_channels', default = 256, type = int,
      help = 'Image channels.')

  # Quantize
  parser.add_argument(
      '--quantize_x_integer', default = 4, type = int,
      help = 'Specify integer data width of input tensor.')
  parser.add_argument(
      '--quantize_x', default = 12, type = int,
      help = 'Specify data width of input tensor.')

  # Output
  parser.add_argument(
      '--output', default = 'img_56_256_process_fix.dat',
      help = 'Output file name.')
  parser.add_argument(
      '--bin', dest='bin', action='store_true',
      help = 'Output binary for on-board test.')
  parser.add_argument(
      '--txt', dest='bin', action='store_false',
      help = 'Output text for simulation.')
  parser.set_defaults(bin=True)
  args = parser.parse_args()

  # PARAMETER
  paral_in = args.paral_in
  img_w = args.img_size
  img_h = args.img_size
  img_ch = args.img_channels

  dat_raw_path = Path('.') / 'fig' / args.input
  dat_path = Path('.') / 'fig' / args.output

  img_raw = np.fromfile(dat_raw_path, dtype=np.float32)
  img_tf = tfConverttoTensor(img_raw, dtype=tfFloat32)
  # img_raw = np.reshape(img_raw, (img_ch, img_w, img_h))
  img_reshape = tfReshape(img_tf, [1, img_ch, img_h, img_w])
  img_transpose = tfTranspose(img_reshape, perm = [2, 3, 0, 1])

  # (row, col, 1, ch/paral, paral) -> (1, row, ch/paral, col, paral)
  data_1d = utils.ConvertTensor(img_transpose, [2, 0, 3, 1, 4], paral_in)

  if args.bin:
    print('[INFO][convert_act_structure.py] Open file as binary.')
    file_mode = 'wb'
  else:
    print('[INFO][convert_act_structure.py] Open file as text.')
    file_mode = 'w'

  QuantizeFunc = utils.GenerateRoundFn(
      args.quantize_x_integer, args.quantize_x, 'mul')

  with open(dat_path, file_mode) as f:
    if args.bin:
      print('[INFO][convert_act_structure.py] '
          'Store activation as binary for on-board test.')
      data_quantize = QuantizeFunc(data_1d)
      for npiter in np.nditer(data_quantize.numpy().astype(np.int32)):
        f.write(npiter)

    else:
      print('[INFO][convert_act_structure.py] '
          'Store activation as txt for simluation.')
      data_reshape = tfReshape(data_1d, [-1, 8])
      data_quantize = QuantizeFunc(data_reshape)
      it = np.nditer(
          data_quantize.numpy().astype(np.int16), flags=['multi_index'])
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
