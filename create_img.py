import cv2
import numpy as np

from pathlib import Path
import pdb

img_w = 416
img_h = 416
img_ch = 8
num_pixel = img_w * img_h * img_ch
sc = img_h * img_w
sh = img_h
sw = 1

dat_path = Path('.') / 'fig' / 'img_2.bin'

img = np.random.randint(255, size = (img_ch, img_h, img_w), dtype = np.uint8)

# for row in range(img_h):
  # for col in range(img_w):
    # for k in range(img_ch):
      # data_img[k * sc + row * sh + col * sw] = \
          # (img[row, col, k] - 127.0) / 128.0

data_img = (img - 127.0) / 128.0

with open(dat_path, mode='wb') as f:
  for npiter in np.nditer(data_img.astype(np.float32)):
    f.write(npiter)
