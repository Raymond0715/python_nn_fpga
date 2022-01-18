import argparse

from pathlib import Path
import numpy as np

# from models.alexnet_fpga import model
# from models.alexnet_merge_bn import model_merge_bn
from test_conv import OneConvNet
from utils import StoreWeight
import pdb

def RoundPower2(x, k=4):
  bound = np.power(2.0, k - 1)
  min_val = np.power(2.0, -bound + 1.0)
  s = np.sign(x)
  x = np.clip(np.absolute(x), min_val, 1.0)
  p = -1 * np.around(np.log(x) / np.log(2.))
  if s == -1:
    p = 0x8 + p.astype(np.int8)
  return p.astype(np.int8)
  # return 0x2


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


# Multiplication convolution
# def Store4DBinConvert(weight, f):
  # weight_h = weight.shape[0]
  # weight_w = weight.shape[1]
  # weight_c = weight.shape[2]
  # weight_f = weight.shape[3]
  # weight_p = 64

  # num_pixel = weight_f * weight_c * weight_h * weight_w
  # step_f = weight_c * weight_h * weight_w * weight_p       # Step for filter
  # step_c = weight_h * weight_w * weight_p                  # Step for col
  # step_h = weight_w * weight_p                             # Step for channel
  # step_w = weight_p                                        # Step for row

  # weight_np = weight.numpy()
  # data_weight = np.zeros(num_pixel, dtype=np.float32)
  # for b in range(int(weight_f / weight_p)):
    # for k in range(weight_c):
      # for row in range(weight_h):
        # for col in range(weight_w):
          # for p in range(weight_p):
            # # index = b * step_f + k * step_c + row * step_h + col * step_w
            # index = b * step_f + k * step_c + row * step_h + col * step_w + p
            # data_weight[index] = weight_np[row, col, k, b * weight_p + p]

  # for npiter in np.nditer(data_weight):
    # # f.write(str(npiter) + ' ')
    # f.write(npiter)


# Shift convolution
def Store4DBinConvert(weight, f):
  weight_h = weight.shape[0]
  weight_w = weight.shape[1]
  weight_c = weight.shape[2]
  weight_f = weight.shape[3]
  weight_p = 64
  # weight_p = 1

  num_pixel = weight_f * weight_c * weight_h * weight_w
  step_f = weight_c * weight_h * weight_w * weight_p       # Step for filter
  step_c = weight_h * weight_w * weight_p                  # Step for col
  step_h = weight_w * weight_p                             # Step for channel
  step_w = weight_p                                        # Step for row

  weight_np = weight.numpy()
  data_weight = np.zeros(int(num_pixel), dtype=np.int32)
  # data_weight = np.zeros(int(num_pixel), dtype=np.float32)
  for b in range(int(weight_f / weight_p)):
    for k in range(weight_c):
      for row in range(weight_h):
        for col in range(weight_w):
          for p in range(weight_p):
            index = int(
                b * step_f + k * step_c + row * step_h + col * step_w + p)
            # data_weight[index] = \
                # RoundPower2(weight_np[row, col, k, b * weight_p + p])
            # data_weight[index] = weight_np[row, col, k, b * weight_p + p]
            data_weight[index] = \
                Round2Fixed(weight_np[row, col, k, b * weight_p + p], 4, 12)
            # data_weight[index] = 1

  for npiter in np.nditer(data_weight):
    # f.write(str(npiter) + ' ')
    f.write(npiter)


def Store1DBinConvert(weight, f):
  weight_c = weight.shape[0]

  weight_np = weight.numpy()
  data_weight = np.zeros(int(weight_c), dtype=np.int32)
  # data_weight = np.zeros(int(num_pixel), dtype=np.float32)
  for b in range(int(weight_c)):
    # data_weight[b] = Round2Fixed(weight_np[b], 4, 12)
    data_weight[b] = Round2Fixed(weight_np[b], 8, 24)

  for npiter in np.nditer(data_weight):
    f.write(npiter)


def StoreWeightBinConvert(weight, f):
  if len(weight.shape) == 4:
    print('[INFO][utils.py] Store 4D tensor {}'.format(weight.name))
    Store4DBinConvert(weight, f)
  elif len(weight.shape) == 1:
    print('[INFO][utils.py] Store 1D tensor {}'.format(weight.name))
    Store1DBinConvert(weight, f)
  else:
    print('[INFO][utils.py] Wrong weight shape!!! '
        'Variable name is {}'.format(weight.name))

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
      '--input_file', default = 'weight_13_1024_256.h5',
      help = 'Input file name.')
  parser.add_argument(
      '--output_file', default = 'weight_13_1024_256_process_fix.bin',
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

  input_tensor_shape = (None, 13, 13, 1024)
  model = OneConvNet('ste', 12, 12)
  model.build(input_tensor_shape)
  model.load_weights(str(ckpt_path))
  pdb.set_trace()

  if args.bin:
    print('[INFO][main.py] Store as binary.')
    with open(str(output_path), mode = 'wb') as f:
      for i, weight in enumerate(model.weights):
        # StoreWeightBin(weight, f)
        StoreWeightBinConvert(weight, f)
  else:
    print('[INFO][main.py] Store as text.')
    with open(str(output_path), mode = 'wb') as f:
      for i, weight in enumerate(model.weights):
        StoreWeight(weight, f)
      f.write('\n')
