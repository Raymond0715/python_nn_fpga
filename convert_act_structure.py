import argparse
from pathlib import Path
import numpy as np


def Round2Int(x, integer=16, k=32):
  assert integer >= 1, integer
  fraction = k - integer
  bound = np.power(2.0, k - 1)
  n = np.power(2.0, fraction)
  min_val = -bound
  max_val = bound
  # ??? Is this round function correct
  x_round = np.around(x * n)
  clipped_value = np.clip(x_round, min_val, max_val).astype(np.int32)
  return clipped_value


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = 'For FPGA, convert activation data layout')
  parser.add_argument(
      '--paral_in', default = 8, type = int,
      help = 'Input degree of parallelism. No need to change in most case.')
  parser.add_argument(
      '--img_size', default = 56, type = int,
      help = 'Input image size.')
  parser.add_argument(
      '--img_channels', default = 256, type = int,
      help = 'Image channels.')
  parser.add_argument(
      '--quantize_x_integer', default = 4, type = int,
      help = 'Specify integer data width of input tensor.')
  parser.add_argument(
      '--quantize_x', default = 12, type = int,
      help = 'Specify data width of input tensor.')
  parser.add_argument(
      '--input', default = 'img_56_256.bin',
      help = 'Input file name.')
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
  PARAL_IN = args.paral_in
  img_w = args.img_size
  img_h = args.img_size
  img_ch = args.img_channels

  num_pixel = img_w * img_h * img_ch

  sp = 1
  sw = PARAL_IN
  sc = img_w * PARAL_IN
  sh = img_ch * img_w

  dat_raw_path = Path('.') / 'fig' / args.input
  dat_path = Path('.') / 'fig' / args.output

  img_raw = np.fromfile(dat_raw_path, dtype=np.float32)
  img_raw = np.reshape(img_raw, (img_ch, img_w, img_h))

  data_img = np.zeros(num_pixel, dtype = np.int32)

  for row in range(img_h):
    for k in range(int(img_ch / PARAL_IN)):
      for col in range(img_w):
        for p in range(PARAL_IN):
          data_img[row * sh + k * sc + col * sw + p * sp] = \
              Round2Int(img_raw[k * PARAL_IN + p, row, col],
                  args.quantize_x_integer, args.quantize_x)

  if args.bin:
    print('[INFO][convert_act_structure.py] Store output as binary.')
    with open(dat_path, 'wb') as f:
      for npiter in np.nditer(data_img):
        f.write(npiter)
  else:
    print('[INFO][convert_act_structure.py] Store output as text.')
    with open(dat_path, 'w') as f:
      for j in range(int(num_pixel/args.paral_in)):
        for i in range(args.paral_in):
          pixel = data_img[j*args.paral_in+args.paral_in-1-i]
          if pixel >= 0:
            pixel_str = '{:0>4x}'.format(pixel)
          else:
            pixel_str = '{:0>4x}'.format(0x10000+pixel)
          f.write(pixel_str)

        f.write('\n')
