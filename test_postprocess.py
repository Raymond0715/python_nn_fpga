import argparse
from pathlib import Path
import numpy as np

from tensorflow import Variable
from tensorflow import convert_to_tensor as tfConverttoTensor
from tensorflow import transpose as tfTranspose
from tensorflow import reshape as tfReshape
from tensorflow import reverse as tfReverse
from tensorflow import float32 as tfFloat32
from tensorflow.keras import Model
from tensorflow.keras.activations import relu as tfReLu
# from tensorflow.keras.layers import Dense as Dense

from nn_utils import QConv2D
import utils

class OneConvNet(Model):
  def __init__(
      self,
      quantize,
      quantize_w_int,
      quantize_w,
      quantize_x_int,
      quantize_x):

    super().__init__(name = '')
    self.quantize = quantize
    self.quantize_w_int = quantize_w_int
    self.quantize_w = quantize_w
    self.quantize_x_int = quantize_x_int
    self.quantize_x = quantize_x
    self.alpha = Variable(0., trainable = False)

    self.conv1 = QConv2D(
        256, 3, 1,
        # 4096, 1, 1,
        quantize = self.quantize,
        quantize_w_int = self.quantize_w_int,
        quantize_w = self.quantize_w,
        quantize_x_int = self.quantize_x_int,
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
  # x_round = np.around(x * n)
  x_round = np.trunc(x * n)
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
  # Activation tensor.
  # (batch, row, col, channels) -> (row, col, batch, channels)
  img_transpose = tfTranspose(tensor, perm = [1, 2, 0, 3])

  # (row, col, 1, ch, 1) -> (1, 1, ch, row, col)
  data_1d = utils.ConvertTensor(img_transpose, [2, 4, 3, 0, 1], 1)

  data_tensor = Round2Fixed(data_1d, integer, width)

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
      '--img_dat', default = 'img_56_256.bin',
      # '--img_dat', default = 'img_56_256_layer2.bin',
      help = 'Image file name.')
  parser.add_argument(
      '--img_size', default = 56, type = int,
      help = 'Input image size.')
  parser.add_argument(
      '--img_channels', default = 256, type = int,
      help = 'Image channels.')

  # ckpt
  parser.add_argument(
      '--ckpt', default = 'weight_56_256.h5',
      help = 'Checkpoint name. Set as None if don\'t load existing model.')
  parser.add_argument(
      '--ckpt_filter_num', default = 256, type = int,
      help = 'Number of output channels of filter.')
  parser.add_argument(
      '--ckpt_bias', default = 'bias_56_256.dat',
      help = 'Checkpoint file for bias. Set as None to create new bias data.')
  parser.add_argument(
      '--ckpt_store', default = 'None',
      help = 'Output ckpt name. Set as None if don\'t store model.')
  parser.add_argument(
      '--ckpt_bias_store', default = 'None',
      help = 'Output ckpt bias name. Set as None if don\'t store model.')
  parser.add_argument(
      '--ckpt_directory', default = 'post_process_bias_mul',
      help = 'Output directory.')

  # quantize
  parser.add_argument(
      '--quantize', default = 'mul',
      help = 'Choose quantization mode. '
      '`shfit` for shift and `mul` for multiply')
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
  parser.add_argument(
      '--quantize_o_integer', default = 8, type = int,
      help = 'Specify integer data width of input tensor.')
  parser.add_argument(
      '--quantize_o', default = 24, type = int,
      help = 'Specify data width of input tensor.')

  # output
  parser.add_argument(
      '--output_directory', default = 'post_process_mul',
      help = 'Output directory.')
  parser.add_argument(
      '--output_conv', default = 'out_56_256_conv.dat',
      help = 'Output file for convolution name.')
  parser.add_argument(
      '--output_bias', default = 'out_56_256_bias.dat',
      help = 'Output file for port process name.')
  parser.add_argument(
      '--output_relu', default = 'out_56_256_leakyrelu.dat',
      help = 'Output file for port process name.')
  args = parser.parse_args()

  # Define parameters.
  # img
  image_bin_path = Path('.') / 'fig' / args.img_dat
  # ckpt
  ckpt_path = \
      Path('.') / 'ckpt' / args.ckpt
  ckpt_store_path = \
      Path('.') / 'ckpt' / args.ckpt_store
  ckpt_bias_path = \
      Path('.') / 'ckpt_dat' / args.ckpt_directory /args.ckpt_bias
  ckpt_bias_store_path = \
      Path('.') / 'ckpt_dat' / args.ckpt_directory / args.ckpt_bias_store
  # output
  output_conv_path = \
      Path('.') / 'dat' / args.output_directory / args.output_conv
  output_relu_path = \
      Path('.') / 'dat' / args.output_directory / args.output_relu
  output_bias_path = \
      Path('.') / 'dat' / args.output_directory / args.output_bias

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
    bias_quantize = \
        np.fromfile(ckpt_bias_path, dtype = np.int32) \
        / np.power(2, args.quantize_o - args.quantize_o_integer)
  else:
    print('[INFO][test_postprocess.py] Write bias:', ckpt_bias_store_path)
    bias_quantize = Round2Fixed(
        np.random.rand(args.ckpt_filter_num).astype(np.float32) - 0.5,
        args.quantize_o_integer, args.quantize_o)
    bias_int = Round2Int(
        bias_quantize, args.quantize_o_integer, args.quantize_o)
    with open(str(ckpt_bias_store_path), 'wb') as f:
      for bias_data in np.nditer(bias_int):
        f.write(bias_data)

  # Instantiate model.
  print('[INFO][test_postprocess.py] Instantiate model.')
  input_tensor_shape = (None, img_size, img_size, img_channels)
  model = OneConvNet(
      args.quantize, args.quantize_w_integer, args.quantize_w,
      args.quantize_x_integer, args.quantize_x)
  model.build(input_tensor_shape)

  # Load weight data.
  if args.ckpt != 'None':
    print('[INFO][test_postprocess.py] Load model:', ckpt_path)
    model.load_weights(str(ckpt_path))

  # Calculate convolution
  output_conv = model(img)

  if args.ckpt_store != 'None':
    print('[INFO][inference4fpga.py] Store output ckpt:', args.ckpt_store)
    model.save_weights(str(ckpt_store_path))

  print('[INFO][test_postprocess.py] Store output of convolution.')
  if len(output_conv.numpy().shape) == 4:
    Store4DTensor(
        output_conv, output_conv_path, args.quantize_o_integer, args.quantize_o)
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
      output_conv, args.quantize_o_integer, args.quantize_o)

  output_bias = (output_conv_quantize + bias_quantize).astype(np.float32)

  print('[INFO][test_postprocess.py] Store output of bias.')
  if len(output_bias.shape) == 4:
    Store4DTensor(
        tfConverttoTensor(output_bias),
        output_bias_path, args.quantize_o_integer, args.quantize_o)
  else:
    print('[ERROR][test_postprocess.py] Fail to store output_bias.')

  # Calculate ReLu
  print('[INFO][test_postprocess.py] Store output of ReLu.')
  output_relu = tfReLu(tfConverttoTensor(output_bias), alpha = 0.015625)

  if len(output_relu.numpy().shape) == 4:
    Store4DTensor(
        output_relu, output_relu_path, args.quantize_o_integer, args.quantize_o)
  else:
    print('[ERROR][test_postprocess.py] Fail to store output_relu.')
