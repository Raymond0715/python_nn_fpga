from pathlib import Path
import numpy as np
import utils


# dat_raw_path = Path('.') / 'dat' / 'yolo_tiny_pass_version' / 'out_yolo_bench1_l16_process.dat'
# dat_path = Path('.') / 'dat' / 'o1_process.bin'

# dat_raw_path = Path('.') / 'dat' / 'o1_process.bin'
# dat_path = Path('.') / 'dat' / 'o1_process_txt.txt'

dat_raw_path = Path('.') / 'dat' / 'o1.bin'
dat_path = Path('.') / 'dat' / 'o1_txt.txt'

# out_raw = np.fromfile(dat_raw_path, dtype=np.int32)
out_raw = np.fromfile(dat_raw_path, dtype=np.int16)
out_reshape = np.reshape(out_raw, (-1,8))
out_fliplr = np.fliplr(out_reshape)
out_final = np.copy(out_fliplr.astype(np.int16))

# with open(str(dat_path), mode='wb') as f:
  # for npiter in np.nditer(out_final):
    # f.write(npiter)

with open(str(dat_path), mode='w') as f:
  it = np.nditer(np.reshape(out_final, [-1, 8]), flags=['multi_index'])
  utils.StoreFormatTxt(it, '{:0>4x}', 0x10000, 8, f)
