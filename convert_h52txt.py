import pdb
# import tensorflow as tf

from pathlib import Path
import argparse
import numpy as np

# from models.alexnet_fpga import model
# from models.alexnet_merge_bn import model_merge_bn
from test_conv import OneConvNet
from utils import (Store4D, Store1D, StoreWeight, Store4DBin, Store1DBin,
    StoreWeightBin)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description = 'Store model as .dat.')
  parser.add_argument(
      '--merge_bn', default = False, action = 'store_true',
      help = 'Choose if merge BN layer. Default set to False.')
  parser.add_argument(
      '--bin', default = False, action = 'store_true',
      help = 'Store as bin if set true.')
  parser.add_argument(
      '--input_file', default = 'weight_56_512.h5',
      help = 'Input file name.')
  parser.add_argument(
      '--output_file', default = 'weight_56_512.bin',
      help = 'Output file name.')
  args = parser.parse_args()


  ckpt_path = Path('.') / 'ckpt' / args.input_file
  output_path = Path('.') / 'ckpt_dat' / args.output_file

  # input_tensor_shape = (None, 224, 224, 3)
  # if args.merge_bn:
    # print('[INFO][main.py] Model merge BN layer.')
    # model_merge_bn.build(input_tensor_shape)
    # model_merge_bn.load_weights(str(ckpt_path))
    # model = model_merge_bn
  # else:
    # print('[INFO][main.py] Model doesn\'t merge BN layer.')
    # model.build(input_tensor_shape)
    # model.load_weights(str(ckpt_path))

  input_tensor_shape = (None, 56, 56, 512)
  model = OneConvNet('ste', 8, 8)
  model.build(input_tensor_shape)
  model.load_weights(str(ckpt_path))

  pdb.set_trace()

  if args.bin:
    print('[INFO][main.py] Store as binary.')
    f = open(str(output_path), 'wb')
    for i, weight in enumerate(model.weights):
      StoreWeightBin(weight, f)
    f.close()
  else:
    print('[INFO][main.py] Store as text.')
    f = open(str(output_path), 'w')
    for i, weight in enumerate(model.weights):
      StoreWeight(weight, f)
    f.write('\n')
    f.close()
