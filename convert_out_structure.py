from pathlib import Path
# import pdb
import argparse
import numpy as np

# argparse
parser = argparse.ArgumentParser(
    description = 'For fpga, test convolution ip calculate result.')

# img
parser.add_argument(
    '--input', default = 'out_28_512_leakyrelu.dat',
    help = 'Input file name.')
parser.add_argument(
    '--output', default = 'out_28_512_leakyrelu_process.dat',
    help = 'Output file name.')
parser.add_argument(
    '--img_size', default = 28, type = int,
    help = 'Image size')
parser.add_argument(
    '--img_channels', default = 512, type = int,
    help = 'Image size')
args = parser.parse_args()

# Parameter
PARAL_IN = 8

img_w = args.img_size
img_h = args.img_size
img_ch = args.img_channels
num_pixel = img_w * img_h * img_ch

sp = 1
sw = PARAL_IN
sc = img_w * PARAL_IN
sh = img_ch * img_w

dat_raw_path = \
    Path('.') / 'dat' / 'post_process' / args.input
dat_path = \
    Path('.') / 'dat' / 'post_process' / args.output

# Convert
img_raw = np.fromfile(dat_raw_path, dtype=np.int32)
img_raw = np.reshape(img_raw, (img_ch, img_w, img_h))

data_img = np.zeros(num_pixel, dtype = np.int32)

for row in range(img_h):
  for k in range(int(img_ch / PARAL_IN)):
    for col in range(img_w):
      for p in range(PARAL_IN):
        data_img[row * sh + k * sc + col * sw + p * sp] = \
            img_raw[k * PARAL_IN + p, row, col]

with open(dat_path, 'wb') as f:
  for npiter in np.nditer(data_img):
    f.write(npiter)
