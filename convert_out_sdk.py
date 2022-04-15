from pathlib import Path
import numpy as np
import utils


# dat_raw_path = Path('.') / 'dat' / 'yolo_tiny_pass_version' / 'out_yolo_bench1_l16_process.dat'
# dat_raw_path = Path('.') / 'dat' / 'yolo' / 'out2_yolo_bench1_l16_process.dat'
# dat_path = Path('.') / 'dat' / 'o3_pc_process.txt'

# dat_raw_path = Path('.') / 'dat' / 'out_test_process.dat'
# dat_path = Path('.') / 'dat' / 'out_test_process_txt.txt'
# dat_raw_path = Path('.') / 'dat' / 'out_l9l10_pc_process.dat'
# dat_path = Path('.') / 'dat' / 'out_l9l10_pc_process_txt.txt'

dat_raw_path = Path('.') / 'dat' / 'out_yolov2.dat'
dat_path = Path('.') / 'dat' / 'out_yolov2_txt_temp.txt'

# FPGA
out_raw = np.fromfile(dat_raw_path, dtype=np.int32)
# PC
# out_raw = np.fromfile(dat_raw_path, dtype=np.int16)

# out_reshape = np.reshape(out_raw, (-1,8))

# FPGA Last two
out_reshape = np.reshape(out_raw, (13, 16, 8, 13, 8))
out_trans = np.swapaxes(out_reshape, 0, 1)

# PC
# out_fliplr = np.fliplr(out_reshape)
# out_final = np.copy(out_fliplr.astype(np.int16))

# FPGA
# out_final = out_reshape
# FPGA Last two
out_final = np.copy(out_trans)

# Max pool
# out_2d = np.copy(out_fliplr.astype(np.int16))
# out_4d = np.reshape(out_2d, (26, 32, 26, 8))
# out_final = np.copy(out_4d[0:-1:2, :, 0:-1:2, :])

# with open(str(dat_path), mode='wb') as f:
  # for npiter in np.nditer(out_final):
    # f.write(npiter)

with open(str(dat_path), mode='w') as f:
  it = np.nditer(np.reshape(out_final, [-1, 8]), flags=['multi_index'])
  utils.StoreFormatTxt(it, '{:0>4x}', 0x10000, 8, f)
