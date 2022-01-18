import cv2
import numpy as np

from pathlib import Path
import pdb

img_w = 13
img_h = 13
img_ch = 1024
num_pixel = img_w * img_h * img_ch
sc = img_h * img_w
sh = img_h
sw = 1

dat_path = Path('.') / 'fig' / 'img_13_1024.bin'

img = np.random.randint(255, size = (img_h, img_w, img_ch), dtype = np.uint8)
data_img  = np.zeros(num_pixel, dtype=np.float32)

for row in range(img_h):
  for col in range(img_w):
    for k in range(img_ch):
      data_img[k * sc + row * sh + col * sw] = \
          (img[row, col, k] - 127.0) / 128.0

f = open(dat_path, 'wb')
for npiter in np.nditer(data_img):
  f.write(npiter)
f.close()
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.imwrite('./fig/one_conv_net.jpeg', img)

# f = open('./fig/one_conv_net_img.bin', 'wb')
# for cols in range(64):
  # for rows in range(64):
    # f.write((img[cols, rows] - 127.) / 128.)
# f.close()

# img = cv2.imread('./fig/one_conv_net.jpeg')
# cv2.imshow('Image', img)
# cv2.waitKey(0)
