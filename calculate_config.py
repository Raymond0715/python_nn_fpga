import csv
from pathlib import Path
import numpy as np
import pdb

def GenerateCCode(f, i, channel_in, channel_out, img_h,
    config_value, out_in_ch, act_addr, weight_addr, weight_len, bias_addr,
    ddr_addr):
  i = i + 1
  f.write('\t/*** YOLO layer {}:\n'.format(i))
  if row['conf_sampling_switch']:
    f.write('\t * - Convolution: With downsampling\n')
  else:
    f.write('\t * - Convolution\n')
  f.write('\t * - Activation:  {} * {} * {}\n'.format(channel_in, img_h, img_h))
  f.write('\t * - Weight:      {} * {} * 3 * 3\n'.format(channel_in, channel_out))
  f.write('\t */\n')

  f.write('{\n')

  f.write('\t// Layer {}: Parameter\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG0_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(config_value))
  f.write('\t// Layer {}: Parameter 2: {{ output_channels, input_channels }}\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG1_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(out_in_ch))
  f.write('\t// Layer {}: act write address\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG2_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(act_addr))
  f.write('\t// Layer {}: act read address\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG3_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(act_addr))
  f.write('\t// Layer {}: weight write address\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG4_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(weight_addr))
  f.write('\t// Layer {}: weight read address\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG5_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(weight_addr))
  f.write('\t// Layer {}: weight write length: Number of elements.\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG6_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(weight_len))
  f.write('\t// Layer {}: bias write address\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG7_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(bias_addr))
  f.write('\t// Layer {}: bias read address\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG8_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(bias_addr))
  f.write('\t// Layer {}: bias write length: Number of elements.\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG9_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(channel_out))
  f.write('\t// Layer {}: ddr write address\n'.format(i))
  f.write('\tCONFIG_SYSTOLIC_CONV_mWriteReg(\n')
  f.write('\t\t\tXPAR_CONFIG_SYSTOLIC_CONV_0_S00_AXI_BASEADDR, CONFIG_SYSTOLIC_CONV_S00_AXI_SLV_REG10_OFFSET,\n')
  f.write('\t\t\t0x{:0>8x});\n'.format(ddr_addr))

  f.write('}\n\n')


# Parameter
channel_in = 8

# conf_path = Path('.') / 'config' / 'yolo_v3_tiny.csv'
# weight_addr = 0
# act_addr = 0x2000000
# bias_addr = 0x1000000

conf_path = Path('.') / 'config' / 'yolo_v2.csv'
out_path = Path('.') / 'config' / 'yolo_v2_template.cpp'
act_addr = 0x5000000
weight_addr = 0
bias_addr = 0x4000000


print('[INFO][calculate_config.py] Initial weight: 0x000c000e')
print('[INFO][calculate_config.py] Reset: 0x00100000')

with open(conf_path, newline='') as csvfile, \
    open(out_path, 'w') as outfile:
  reader = csv.DictReader(csvfile, delimiter=',')
  for i, row in enumerate(reader):
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

    channel_out = int(row['conf_channel_out'])
    ddr_addr = \
        int(row['conf_img_h']) * int(row['conf_img_h']) * channel_in * 2 \
        + act_addr

    if not bool(int(row['conf_33'])):
      weight_len = 9 * channel_in * channel_out
    else:
      weight_len = channel_in * channel_out

    out_in_ch = channel_out * 16**3 + channel_in

    print(
        '[INFO][calculate_config.py][{}]\n'
        'config_value: 0x{:0>8x}\n'
        'out_in_ch:    0x{:0>8x}\n'
        'act_addr:     0x{:0>8x}\n'
        'weight_addr:  0x{:0>8x}\n'
        'weight_len:   0x{:0>8x}\n'
        'bias_addr:    0x{:0>8x}\n'
        'bias_len:     0x{:0>8x}\n'
        'ddr_addr:     0x{:0>8x}'
        .format(row['name'], config_value, out_in_ch, act_addr, weight_addr,
          weight_len, bias_addr, channel_out, ddr_addr))
    GenerateCCode(outfile, i, channel_in, channel_out, int(row['conf_img_h']),
        config_value, out_in_ch, act_addr, weight_addr, weight_len, bias_addr,
        ddr_addr)

    # Update parameter
    channel_in = channel_out
    weight_addr = weight_addr + weight_len * 2
    bias_addr = bias_addr + channel_out * 4
    act_addr = ddr_addr
