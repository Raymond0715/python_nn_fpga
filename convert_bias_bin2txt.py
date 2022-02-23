import argparse
from pathlib import Path

import numpy as np

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description =
      '''
      Convert bias from binary to text. Text file is used for simulation and
      binary file is used for on board test.
      ''')
  parser.add_argument(
      '--num_data', default = 256, type = int,
      help = 'Number of data.')
  parser.add_argument(
      '--package_size', default = 4, type = int,
      help = 'Input degree of parallelism. No need to change in most case.')
  parser.add_argument(
      '--directory', default = 'post_process_bias_mul',
      help = 'Output directory.')
  parser.add_argument(
      '--input_file', default = 'bias_56_256.bin',
      help = 'Input file name.')
  parser.add_argument(
      '--output_file', default = 'bias_56_256_sim.bin',
      help = 'Output file name.')
  args = parser.parse_args()

  input_path = \
      Path('.') / 'ckpt_dat' / args.directory / args.input_file
  output_path = \
      Path('.') / 'ckpt_dat' / args.directory / args.output_file

  bias_int = np.fromfile(input_path, dtype=np.int32)

  with open(str(output_path), mode = 'w') as f:
    for j in range(int(args.num_data/args.package_size)):
      for i in range(args.package_size):
        pixel = bias_int[j*args.package_size+i]
        if pixel >= 0:
          pixel_str = '{:0>8x}'.format(pixel)
        else:
          pixel_str = '{:0>8x}'.format(0x10000+pixel)
        f.write(pixel_str)

      f.write('\n')
