import csv
from pathlib import Path
import numpy as np

# Parameter
layer_num = 1
conf_path = Path('.') / 'config' / 'yolo.csv'

# Config
conf_33                  = 0
conf_act_src             = 1
conf_weight_src          = 1
conf_relu_mode           = 1
conf_relu_switch         = 1
conf_bias_switch         = 1
conf_sampling_switch     = 0
conf_bitintercept_switch = 1
conf_img_h               = 56
conf_finish              = 1
conf_reset               = 0

with open(conf_path, newline='') as csvfile:
  reader = csv.DictReader(csvfile, delimiter=',')
  for row in reader:
    print(type(row['conf_33']))

config_value = np.ndarray((11, layer_num))

for i in range(layer_num):
  config_value[0, i] = \
      conf_33 + 2 * conf_act_src + 2**2 * conf_weight_src \
      + 2**3 * conf_relu_mode + 2**4 * conf_relu_switch \
      + 2**5 * conf_bias_switch + 2**6 * conf_sampling_switch \
      + 2**7 * conf_bitintercept_switch + 2**8 * conf_img_h \
      + 2**17 * conf_finish + 2**18 * conf_reset
  print('[INFO][calculate_config.py] config_value', i, ":", hex(int(config_value[0, i])))
