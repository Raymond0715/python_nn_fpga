from pathlib import Path
import argparse
import numpy as np

from tensorflow import convert_to_tensor as tfConverttoTensor
from tensorflow import transpose as tfTranspose
from tensorflow import reshape as tfReshape
from tensorflow import float32 as tfFloat32

import utils
import pdb


if __name__ == '__main__':
  # argparse
  parser = argparse.ArgumentParser(
      description = 'For fpga, test convolution ip calculate result.')
  parser.add_argument(
      '--directory', default = 'post_process_mul',
      help = 'File directory')
  parser.add_argument(
      '--paral_out', default = 8, type = int,
      help = 'Output file degree of parallelism. Depend on data width.')
  parser.add_argument(
      '--input', default = 'out_56_256_leakyrelu.dat',
      help = 'Input file name.')
  parser.add_argument(
      '--output', default = 'out_56_256_leakyrelu_process.dat',
      help = 'Output file name.')
  parser.add_argument(
      '--img_size', default = 56, type = int,
      help = 'Image size')
  parser.add_argument(
      '--img_channels', default = 256, type = int,
      help = 'Image size')
  parser.add_argument(
      '--quantize_x_integer', default = 4, type = int,
      help = 'Specify integer data width of input tensor.')
  parser.add_argument(
      '--quantize_x', default = 12, type = int,
      help = 'Specify data width of input tensor.')
  args = parser.parse_args()

  # Parameter
  dat_raw_path = Path('.') / 'dat' / args.directory / args.input
  dat_path     = Path('.') / 'dat' / args.directory / args.output

  img_raw = np.fromfile(dat_raw_path, dtype=np.float32)
  img_tf = tfConverttoTensor(img_raw, dtype=tfFloat32)
  img_reshape = tfReshape(
      img_tf, [1, args.img_channels, args.img_size, args.img_size])

  '''
  Generate simulation data for reorder module. Weight tiling is 2 and output
  parallelism degree is 64.
  '''
  # bin: (row, col, 1, ch/paral, paral) -> (1, row, ch/paral, col, paral)
  # txt: (row, col, 2, ch/paral/2, paral) -> (2, row, ch/paral/2, col, paral)
  img_reshape = tfReshape(
      img_tf, [2, -1, args.img_size, args.img_size])
  # img_reshape = tfReshape(
      # img_tf, [1, -1, args.img_size, args.img_size])
  pdb.set_trace()

  img_transpose = tfTranspose(img_reshape, perm = [2, 3, 0, 1])

  data_1d = utils.ConvertTensor(img_transpose, [2, 0, 3, 1, 4], args.paral_out)

  QuantizeFunc = utils.GenerateRoundFn(
      args.quantize_x_integer, args.quantize_x, 'mul')

  print('[INFO][convert_out_structure.py] Open file as text.')
  file_mode = 'w'

  with open(dat_path, file_mode) as f:
    print('[INFO][convert_out_structure.py] '
        'Store output as txt for simluation. '
        'ESPECIALLY for YOLO tiny layer 11.')
    data_reshape = tfReshape(data_1d, [-1, args.paral_out])
    data_quantize = QuantizeFunc(data_reshape)
    it = np.nditer(
        data_quantize.numpy().astype(np.int32), flags=['multi_index'])
    for npiter in it:
      # 13 * 13 * 4 = 676
      if it.multi_index[0] % 676 == 0 and it.multi_index[1] == 0:
      # # 26 * 26 * 4 = 2704
      # if it.multi_index[0] % 2704 == 0 and it.multi_index[1] == 0:

        print('[INFO][convert_out_structure.py] Line {}. Writing zero...'
            .format(it.multi_index[0]))
        utils.WriteZero(13*4, f)
        # utils.WriteZero(26*4, f)

      if it.multi_index[1] == 0:
        data_str = ''

      if npiter >= 0:
        pixel_str = '{:0>8x}'.format(npiter)
      else:
        pixel_str = '{:0>8x}'.format(0x100000000+npiter)

      data_str = data_str + pixel_str

      if it.multi_index[1] == args.paral_out-1 or it.finished:
        f.write(data_str + '\n')
