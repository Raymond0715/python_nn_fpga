import argparse
from pathlib import Path

import numpy as np

import utils

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description =
      '''
      Convert bias from binary to text. Text file is used for simulation and
      binary file is used for on board test.
      ''')
  parser.add_argument(
      '--package_size', default = 4, type = int,
      help = 'Input degree of parallelism. No need to change in most case.')
  parser.add_argument(
      '--directory', default = 'yolo',
      help = 'Output directory.')
  parser.add_argument(
      '--input_file', default = 'bias_208_16_shift_process.dat',
      help = 'Input file name.')
  parser.add_argument(
      '--output_file', default = 'bias_208_16_shift_process_sim.txt',
      help = 'Output file name.')
  args = parser.parse_args()

  input_path = \
      Path('.') / 'ckpt_dat' / args.directory / args.input_file
  output_path = \
      Path('.') / 'ckpt_dat' / args.directory / args.output_file

  bias_int = np.fromfile(input_path, dtype=np.int32)

  with open(str(output_path), mode = 'w') as f:
    it = np.nditer(
        np.reshape(bias_int, (-1, args.package_size)), flags=['multi_index'])
    utils.StoreFormatTxt(it, '{:0>8x}', 0x100000000, args.package_size, f)
