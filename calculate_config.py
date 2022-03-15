import csv
from pathlib import Path
import numpy as np
import pdb

# Parameter
layer_num = 1
# conf_path = Path('.') / 'config' / 'yolo.csv'
conf_path = Path('.') / 'config' / 'test.csv'

config_value = np.ndarray((11, layer_num))

with open(conf_path, newline='') as csvfile:
  reader = csv.DictReader(csvfile, delimiter=',')
  for row in reader:
    config_value = \
        int(row['conf_33']) \
        + int(row['conf_act_src'])             * 2 \
        + int(row['conf_weight_src'])          * 2**2 \
        + int(row['conf_relu_mode'])           * 2**3 \
        + int(row['conf_relu_switch'])         * 2**4 \
        + int(row['conf_bias_switch'])         * 2**5 \
        + int(row['conf_sampling_switch'])     * 2**6 \
        + int(row['conf_bitintercept_switch']) * 2**7 \
        + int(row['conf_img_h'])               * 2**8 \
        + int(row['conf_output_sink'])         * 2**17 \
        + int(row['conf_init_weight'])         * 2**18 \
        + int(row['conf_finish'])              * 2**19 \
        + int(row['conf_reset'])               * 2**20 \
    # print(type(row['conf_33']))
    print(
        '[INFO][calculate_config.py] config_value {:>11} : 0x{:0>8x}'
        .format(row['name'], config_value))
