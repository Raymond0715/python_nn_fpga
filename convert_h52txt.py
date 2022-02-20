import argparse

from pathlib import Path
import numpy as np

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
  data_weight = np.zeros(int(num_pixel), dtype=np.int16)
  for b in range(int(weight_f / weight_p)):
    for k in range(weight_c):
      for row in range(weight_h):
        for col in range(weight_w):
          for p in range(weight_p):
            index = int(
                b * step_f + k * step_c + row * step_h + col * step_w + p)
            ######################
            ### Shift          ###
            ######################
            data_weight[index] = \
                RoundPower2(weight_np[row, col, k, b * weight_p + p])
            ######################
            ### Multiplication ###
            ######################
            # data_weight[index] = \
                # Round2Fixed(weight_np[row, col, k, b * weight_p + p], 4, 12)

  for i in range(int(num_pixel / 2)):
    temp = data_weight[2 * i]
    data_weight[2 * i] = data_weight[2 * i + 1]
    data_weight[2 * i + 1] = temp

  for npiter in np.nditer(data_weight):
    f.write(npiter)


def Store1DBinConvert(weight, f):
  weight_c = weight.shape[0]

  weight_np = weight.numpy()
  data_weight = np.zeros(int(weight_c), dtype=np.int32)
  for b in range(int(weight_c)):
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
      description = 'Store model as .dat. Default running command is'
      '`python convert_h52txt.py`')
  parser.add_argument(
      '--img_w', default = 56, type = int,
      help = 'Image width and height.')
  parser.add_argument(
      '--img_ch', default = 256, type = int,
      help = 'Number of image channels.')
  parser.add_argument(
      '--input_file', default = 'weight_56_256.h5',
      help = 'Input file name.')
  parser.add_argument(
      '--output_file', default = 'weight_56_256_shift_process_16bit.bin',
      help = 'Output file name.')
  args = parser.parse_args()


  ckpt_path = Path('.') / 'ckpt' / args.input_file
  output_path = Path('.') / 'ckpt_dat' / args.output_file

  input_tensor_shape = (None, args.img_w, args.img_w, args.img_ch)
  # Don't change. Quantization will be executed in store function.
  model = OneConvNet('ste', 32, 32)
  model.build(input_tensor_shape)
  model.load_weights(str(ckpt_path))

  with open(str(output_path), mode = 'wb') as f:
    for i, weight in enumerate(model.weights):
      StoreWeightBinConvert(weight, f)
