from pathlib import Path
import pdb

import numpy as np

import utils


# parameter
in_channels = 16
out_channels = 32
ckpt_directory = 'yolo'
ckpt_in = 'weight_208_16_shift_process.dat'
# ckpt_out = 'weight_208_16_shift_process_64paral.dat'
ckpt_out = 'weight_208_16_shift_process_64paral_sim.txt'
flag_bin = False


# main
weight_raw_path = Path('.') / 'ckpt_dat' / ckpt_directory / ckpt_in
out_weight_path = Path('.') / 'ckpt_dat' / ckpt_directory / ckpt_out

weight_raw = np.fromfile(weight_raw_path, dtype=np.int16)
weight_reshape = np.fliplr(np.reshape(weight_raw, (-1, 2)))
weight_tensor = np.reshape(weight_reshape, (1, in_channels, 3, 3, out_channels))
weight_concat = np.concatenate((weight_tensor, weight_tensor), axis=4)

if flag_bin:
  with open(str(out_weight_path), mode='wb') as f:
    weight_out = np.copy(np.fliplr(np.reshape(weight_concat, (-1, 2))))
    for npiter in np.nditer(weight_out):
      f.write(npiter)
else:
  with open(str(out_weight_path), mode='w') as f:
    weight_out = np.reshape(weight_concat, (-1, 2))
    it = np.nditer(np.reshape(weight_out, [-1, 8]), flags=['multi_index'])
    utils.StoreFormatTxt(it, '{:0>4x}', 0x10000, 8, f)
