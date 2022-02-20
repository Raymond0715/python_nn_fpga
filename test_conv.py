import argparse
from pathlib import Path
import pdb
import numpy as np

from tensorflow import Variable
from tensorflow import convert_to_tensor as tfConverttoTensor
from tensorflow import transpose as tfTranspose
from tensorflow import float32 as tffloat32
from tensorflow.keras import Model
# from tensorflow.keras.layers import MaxPooling2D as MaxPooling2D
# from tensorflow.keras.layers import Dense as Dense
# from tensorflow.keras.layers import Flatten as Flatten
# from tensorflow.io import read_file as tfReadFile
# from tensorflow.image import decode_jpeg as tfDecodeJpeg
# from tensorflow.image import resize as tfResize
# from tensorflow.math import divide as tfDivide
# from tensorflow.math import add as tfAdd

from nn_utils import QConv2D

class OneConvNet(Model):
  def __init__(
      self,
      quantize,
      quantize_w,
      quantize_x):

    # super(OneConvNet, self).__init__(name = '')
    super().__init__(name = '')
    self.quantize = quantize
    self.quantize_w = quantize_w
    self.quantize_x = quantize_x
    self.alpha = Variable(0., trainable = False)

    self.conv1 = QConv2D(
        256, 3, 1,
        # 4096, 1, 1,
        quantize = self.quantize,
        quantize_w = self.quantize_w,
        quantize_x = self.quantize_x,
        weight_decay = 0,
        # use_bias = True,
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

def Store4DTensor(tensor, output_path):
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
  # data_tensor = np.zeros(num_pixel, dtype=np.float32)
  data_tensor = np.zeros(num_pixel, dtype=np.int32)

  for b in range(tensor_b):
    for row in range(tensor_h):
      for col in range(tensor_w):
        for k in range(tensor_c):
          index = b * step_b + k * step_c + row * step_h + col * step_w
          data_tensor[index] = \
                  Round2Fixed(tensor_np[b, row, col, k], 4, 16)
                  # Round2Fixed(tensor_np[b, row, col, k], 4, 12)
                  # Round2Fixed(tensor_np[b, row, col, k], 8, 24)

  with open(str(output_path), 'wb') as f:
    # for i, npiter in enumerate(np.nditer(data_tensor)):
    for npiter in np.nditer(data_tensor):
      f.write(npiter)
      # f.write(str(npiter) + ' ')
      # f.write(str(npiter) + ', ')
      # if (i + 1) % 9 == 0:
        # f.write('\n')

    # if num_pixel % 9 != 0:
      # f.write('\n')


def Store2DTensor(tensor, output_path):
  tensor_len  = tensor.shape[1]
  tensor_np   = tensor.numpy()

  with open(str(output_path), 'wb') as f:
    for i in range(tensor_len):
      # f.write(str(tensor_np[0,i]) + ', ')
      f.write(tensor_np[0,i])
      # if (i + 1) % 9 == 0:
        # f.write('\n')

    # if tensor_len % 9 != 0:
      # f.write('\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = 'For fpga, test convolution ip calculate result.')
  parser.add_argument(
      '--quantize', default = 'ste',
      help = 'Choose quantization mode.')
  parser.add_argument(
      '--quantize_w', default = 12, type = int,
      help = 'Specify data width of weight.')
  parser.add_argument(
      '--quantize_x', default = 12, type = int,
      help = 'Specify data width of input tensor.')
  parser.add_argument(
      '--image', default = 'one_conv_net.jpeg',
      help = 'Image name.')
  parser.add_argument(
      '--ckpt', default = 'weight_56_256.h5',
      help = 'Checkpoint name. Default is alexnet_random.h5. Set as None if'
      ' don\'t load existing model.')
  parser.add_argument(
      '--dat', default = 'out_56_256.dat',
      help = 'Output file name.')
  parser.add_argument(
      '--output_ckpt', default = 'None',
      help = 'Output ckpt name. Set as None if don\'t store model.')
  parser.add_argument(
      '--img_size', default = 56, type = int,
      help = 'Input image size.')
  parser.add_argument(
      '--img_channels', default = 256, type = int,
      help = 'Input image channels.')
  args = parser.parse_args()

  # Define parameters.
  image_path = Path('.') / 'fig' / args.image
  image_bin_path = Path('.') / 'fig' / 'img_56_256.bin'
  ckpt_path = Path('.') / 'ckpt' / args.ckpt
  output_path = Path('.') / 'dat' / args.dat
  output_ckpt = Path('.') / 'ckpt' / args.output_ckpt
  img_size = args.img_size
  img_channels = args.img_channels

  # # Read image .jpg
  # img = tfReadFile(str(image_path))
  # img = tfDecodeJpeg(img, channels = img_channels)
  # img = tfResize(img, [img_size, img_size])
  # img = tfDivide(tfAdd(tfCast(img, tffloat32), -127), 128)
  # img = tfReshape(img, [1, img_size, img_size, img_channels])

  # Read image .bin
  img = np.fromfile(image_bin_path, dtype = np.float32)
  img = np.reshape(img, (1, img_channels, img_size, img_size))
  img = tfConverttoTensor(img, dtype=tffloat32)
  img = tfTranspose(img, perm = [0, 2, 3, 1])

  # Instantiate model.
  model = OneConvNet(args.quantize, args.quantize_w, args.quantize_x)
  input_tensor_shape = (None, img_size, img_size, img_channels)
  model.build(input_tensor_shape)
  if args.ckpt != 'None':
    model.load_weights(str(ckpt_path))

  # pdb.set_trace()
  output = model(img)
  pdb.set_trace()
  if len(output.numpy().shape) == 4:
    Store4DTensor(output, output_path)
  else:
    Store2DTensor(output, output_path)

  # Store output ckpt
  if args.output_ckpt != 'None':
    print('[INFO][test_conv.py] Store output ckpt:', args.output_ckpt)
    model.save_weights(str(output_ckpt))
