import argparse
from pathlib import Path
import pdb
import numpy as np

from tensorflow import Variable
from tensorflow import convert_to_tensor as tfConverttoTensor
from tensorflow import transpose as tfTranspose
from tensorflow import float32 as tfFloat32
from tensorflow.keras import Model
from tensorflow.keras.activations import relu as tfReLu
# from tensorflow.keras.layers import Dense as Dense

from nn_utils import QConv2D

class OneConvNet(Model):
  def __init__(
      self,
      quantize,
      quantize_w,
      quantize_x):

    super().__init__(name = '')
    self.quantize = quantize
    self.quantize_w = quantize_w
    self.quantize_x = quantize_x
    self.alpha = Variable(0., trainable = False)

    self.conv1 = QConv2D(
        # 64, 3, 1,
        256, 1, 1,
        quantize = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = 0,
        use_bias = False,
        alpha = self.alpha)
    # self.dense1 = Dense(4096, use_bias = False)

  def call(self, input_tensor):
    x = input_tensor

    # x = Flatten()(x)
    # x = MaxPooling2D(pool_size = 2, strides = 1, padding="same")(x)
    x = self.conv1(x)
    # x = self.dense1(x)

    return x


# A better solution is original `Round2Fixed()` and `Fix2int`
def Round2Int(x, integer=16, k=32):
  assert integer >= 1, integer
  fraction = k - integer
  bound = np.power(2.0, k - 1)
  n = np.power(2.0, fraction)
  min_val = -bound
  max_val = bound
  x_round = np.around(x * n)
  clipped_value = np.clip(x_round, min_val, max_val).astype(np.int32)
  return clipped_value

# A better solution is original `Round2Fixed()` and `Fix2int`
def Round2Fixed(x, integer=16, k=32):
  assert integer >= 1, integer
  fraction = k - integer
  bound = np.power(2.0, integer - 1)
  n = np.power(2.0, fraction)
  min_val = -bound
  max_val = bound
  x_round = np.around(x * n) / n
  clipped_value = np.clip(x_round, min_val, max_val)
  return clipped_value

def Store4DTensor(tensor, output_path, integer, width):
  tensor_h = tensor.shape[1]
  tensor_w = tensor.shape[2]
  tensor_c = tensor.shape[3]
  tensor_b = tensor.shape[0]

  step_b = tensor_h * tensor_w * tensor_c
  step_c = tensor_h * tensor_w
  step_h = tensor_h
  step_w = 1

  num_pixel   = tensor_h * tensor_w * tensor_c * tensor_b
  tensor_np   = tensor.numpy()
  data_tensor = np.zeros(num_pixel, dtype=np.int32)

  for b in range(tensor_b):
    for row in range(tensor_h):
      for col in range(tensor_w):
        for k in range(tensor_c):
          index = b * step_b + k * step_c + row * step_h + col * step_w
          data_tensor[index] = \
                  Round2Int(tensor_np[b, row, col, k], integer, width)

  with open(str(output_path), 'wb') as f:
    for npiter in np.nditer(data_tensor):
      f.write(npiter)

def Store2DTensor(tensor, output_path):
  tensor_len  = tensor.shape[1]
  tensor_np   = tensor.numpy()

  with open(str(output_path), 'wb') as f:
    for i in range(tensor_len):
      f.write(tensor_np[0,i])

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = 'For fpga, test convolution ip calculate result.')

  # img
  parser.add_argument(
      '--img_dat', default = 'img_13_1024.bin',
      help = 'Image file name.')
  parser.add_argument(
      '--img_size', default = 13, type = int,
      help = 'Input image size.')
  parser.add_argument(
      '--img_channels', default = 1024, type = int,
      help = 'Image channels.')

  # ckpt
  parser.add_argument(
      '--ckpt', default = 'weight_13_1024_256.h5',
      help = 'Checkpoint name. Set as None if don\'t load existing model.')
  parser.add_argument(
      '--ckpt_filter_num', default = 256, type = int,
      help = 'Number of output channels of filter.')
  parser.add_argument(
      '--ckpt_bias', default = 'None',
      help = 'Checkpoint file for bias.')
  parser.add_argument(
      '--ckpt_store', default = 'None',
      help = 'Output ckpt name. Set as None if don\'t store model.')
  parser.add_argument(
      '--ckpt_bias_store', default = 'bias_13_1024.bin',
      help = 'Output ckpt bias name. Set as None if don\'t store model.')

  # quantize
  parser.add_argument(
      '--quantize', default = 'ste',
      help = 'Choose quantization mode.')
  parser.add_argument(
      '--quantize_w_integer', default = 4, type = int,
      help = 'Specify integer data width of weight.')
  parser.add_argument(
      '--quantize_w', default = 12, type = int,
      help = 'Specify data width of weight.')
  parser.add_argument(
      '--quantize_x_integer', default = 4, type = int,
      help = 'Specify integer data width of input tensor.')
  parser.add_argument(
      '--quantize_x', default = 12, type = int,
      help = 'Specify data width of input tensor.')

  # output
  parser.add_argument(
      '--output_conv', default = 'out_13_1024_conv.dat',
      help = 'Output file for convolution name.')
  parser.add_argument(
      '--output_bias', default = 'out_13_1024_bias.dat',
      help = 'Output file for port process name.')
  parser.add_argument(
      '--output_relu', default = 'out_13_1024_leakyrelu.dat',
      help = 'Output file for port process name.')
  args = parser.parse_args()

  # Define parameters.
  # img
  image_bin_path = Path('.') / 'fig' / args.img_dat
  # ckpt
  ckpt_path = Path('.') / 'ckpt' / args.ckpt
  ckpt_store_path = Path('.') / 'ckpt' / 'post_process' / args.ckpt_store
  ckpt_bias_path = Path('.') / 'ckpt_dat' / 'post_process' /args.ckpt_bias
  ckpt_bias_store_path = Path('.') / 'ckpt_dat' / 'post_process' / args.ckpt_bias_store
  # output
  output_conv_path = Path('.') / 'dat' / 'post_process' / args.output_conv
  output_relu_path = Path('.') / 'dat' / 'post_process' / args.output_relu
  output_bias_path = Path('.') / 'dat' / 'post_process' / args.output_bias

  # Read image .bin
  img_size = args.img_size
  img_channels = args.img_channels
  img = np.fromfile(image_bin_path, dtype = np.float32)
  img = np.reshape(img, (1, img_channels, img_size, img_size))
  img = tfConverttoTensor(img, dtype=tfFloat32)
  img = tfTranspose(img, perm = [0, 2, 3, 1])

  # Read bias .bin
  if args.ckpt_bias != 'None':
    print('[INFO][test_postprocess.py] Read bias:', ckpt_bias_path)
    bias_quantize = np.fromfile(ckpt_bias_path, dtype = np.int32) \
            / np.power(2,
                    args.quantize_x + args.quantize_w
                    - args.quantize_x_integer - args.quantize_w_integer)
    bias_int = Round2Int(
        bias_quantize,
        args.quantize_x_integer + args.quantize_w_integer,
        args.quantize_x + args.quantize_w)
    with open(str(ckpt_bias_store_path), 'wb') as f:
      for bias_data in np.nditer(bias_int):
        f.write(bias_data)
  else:
    print('[INFO][test_postprocess.py] Write bias:', ckpt_bias_store_path)
    bias_quantize = Round2Fixed(
            np.random.rand(args.ckpt_filter_num).astype(np.float32) - 0.5,
            args.quantize_x_integer + args.quantize_w_integer,
            args.quantize_x + args.quantize_w)
    bias_int = Round2Int(
        bias_quantize,
        args.quantize_x_integer + args.quantize_w_integer,
        args.quantize_x + args.quantize_w)
    with open(str(ckpt_bias_store_path), 'wb') as f:
      for bias_data in np.nditer(bias_int):
        f.write(bias_data)

  # Instantiate model.
  print('[INFO][test_postprocess.py] Instantiate model.')
  input_tensor_shape = (None, img_size, img_size, img_channels)
  model = OneConvNet(args.quantize, args.quantize_w, args.quantize_x)
  model.build(input_tensor_shape)

  # Calculate convolution
  if args.ckpt != 'None':
    print('[INFO][test_postprocess.py] Load model:', ckpt_path)
    model.load_weights(str(ckpt_path))
    pdb.set_trace()

  output_conv = model(img)

  if args.ckpt_store != 'None':
    print('[INFO][inference4fpga.py] Store output ckpt:', args.ckpt_store)
    model.save_weights(str(ckpt_store_path))

  print('[INFO][test_postprocess.py] Store output of convolution.')
  if len(output_conv.numpy().shape) == 4:
    Store4DTensor(
        output_conv,
        output_conv_path,
        args.quantize_x_integer + args.quantize_w_integer,
        args.quantize_x + args.quantize_w)
  else:
    print('[ERROR][test_postprocess.py] Fail to store output_conv.')

  # Calculate bias
  output_conv_h = output_conv.shape[1]
  output_conv_w = output_conv.shape[2]
  output_conv_c = output_conv.shape[3]

  output_bias = np.zeros(
      (1, output_conv_h, output_conv_w, output_conv_c),
      dtype=np.float32)
  output_conv_quantize = Round2Fixed(
      output_conv,
      args.quantize_x_integer + args.quantize_w_integer,
      args.quantize_x + args.quantize_w)

  for row in range(output_conv_h):
    for col in range(output_conv_w):
      for k in range(output_conv_c):
        output_bias[0, row, col, k] = \
            output_conv_quantize[0, row, col, k] + bias_quantize[k]

  print('[INFO][test_postprocess.py] Store output of bias.')
  if len(output_bias.shape) == 4:
    Store4DTensor(
        tfConverttoTensor(output_bias),
        output_bias_path,
        args.quantize_x_integer + args.quantize_w_integer,
        args.quantize_x + args.quantize_w)
  else:
    print('[ERROR][test_postprocess.py] Fail to store output_bias.')

  # Calculate ReLu
  print('[INFO][test_postprocess.py] Store output of ReLu.')
  output_relu = tfReLu(tfConverttoTensor(output_bias), alpha = 0.015625)
  if len(output_relu.numpy().shape) == 4:
    Store4DTensor(
        output_relu,
        output_relu_path,
        args.quantize_x_integer + args.quantize_w_integer,
        args.quantize_x + args.quantize_w)
  else:
    print('[ERROR][test_postprocess.py] Fail to store output_relu.')
