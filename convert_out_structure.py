from pathlib import Path
import pdb
import argparse
import numpy as np

def Round2Int(x, integer=16, k=32):
  assert integer >= 1, integer
  fraction = k - integer
  bound = np.power(2.0, k - 1)
  n = np.power(2.0, fraction)
  min_val = -bound
  max_val = bound-1
  # x_round = np.around(x * n)
  x_round = np.floor(x * n)
  clipped_value = np.clip(x_round, min_val, max_val).astype(np.int32)
  return clipped_value


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
  PARAL_OUT = args.paral_out

  integer = args.quantize_x_integer
  width = args.quantize_x

  img_w = args.img_size
  img_h = args.img_size
  img_ch = args.img_channels
  num_pixel = img_w * img_h * img_ch

  sp = 1
  sw = PARAL_OUT
  sc = img_w * PARAL_OUT
  sh = img_ch * img_w

  dat_raw_path = \
      Path('.') / 'dat' / args.directory / args.input
  dat_path     = \
      Path('.') / 'dat' / args.directory / args.output

  # main
  img_raw = np.fromfile(dat_raw_path, dtype=np.float32)
  img_raw = np.reshape(img_raw, (img_ch, img_w, img_h))

  data_img = np.zeros(num_pixel, dtype = np.int32)

  for row in range(img_h):
    for k in range(int(img_ch / PARAL_OUT)):
      for col in range(img_w):
        for p in range(PARAL_OUT):
          data_img[row * sh + k * sc + col * sw + p * sp] = \
              Round2Int(img_raw[k * PARAL_OUT + p, row, col], integer, width)

  with open(dat_path, 'wb') as f:
    for npiter in np.nditer(data_img):
      f.write(npiter)
