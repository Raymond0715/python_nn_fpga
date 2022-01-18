from pathlib import Path
import pdb
import numpy as np

# PARAMETER
PARAL_IN = 8

# img_w = 28
# img_h = 28
# img_ch = 512
img_w = 13
img_h = 13
img_ch = 1024

def Round2Fixed(x, integer=16, k=32):
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

num_pixel = img_w * img_h * img_ch

sp = 1
sw = PARAL_IN
sc = img_w * PARAL_IN
sh = img_ch * img_w

dat_raw_path = Path('.') / 'fig' / 'img_13_1024.bin'
dat_path = Path('.') / 'fig' / 'img_13_1024_process_fix.bin'

img_raw = np.fromfile(dat_raw_path, dtype=np.float32)
img_raw = np.reshape(img_raw, (img_ch, img_w, img_h))

data_img = np.zeros(num_pixel, dtype = np.int32)

for row in range(img_h):
  for k in range(int(img_ch / PARAL_IN)):
    for col in range(img_w):
      for p in range(PARAL_IN):
        data_img[row * sh + k * sc + col * sw + p * sp] = \
            Round2Fixed(img_raw[k * PARAL_IN + p, row, col], 4, 12)
            # Round2Fixed(img_raw[k * PARAL_IN + p, row, col], 3, 8)

# pdb.set_trace()
f = open(dat_path, 'wb')
for npiter in np.nditer(data_img):
  f.write(npiter)
f.close()
