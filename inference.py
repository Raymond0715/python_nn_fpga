from pathlib import Path
import argparse
import pdb
import numpy as np

# import tensorflow as tf
from tensorflow import convert_to_tensor as tfConverttoTensor
from tensorflow import transpose as tfTranspose
from tensorflow import float32 as tfFloat32
from models.yolo_tiny import GenerateModel

import nn_utils
import utils


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'CNN inference for fpga.'
      'Including merge bn layer. Store output at the end.')

  # input
  parser.add_argument(
      '--img', default = 'ILSVRC2012_val_00050000.JPEG',
      help = 'Image name. Default is ILSVRC2012_val_00050000.JPEG')
  parser.add_argument(
      '--img_size', default = 416, type = int,
      help = 'Input image size.')
  parser.add_argument(
      '--img_channels', default = 3, type = int,
      help = 'Image channels.')

  # ckpt
  parser.add_argument(
      '--ckpt', default = 'yolo_tiny.h5',
      help = 'Checkpoint name. Set as None if there is no existing ckpt' \
          'randomly.')
  parser.add_argument(
      '--ckpt_store', default = 'None',
      help = 'Output ckpt name. Set as None if don\'t store model.')
  parser.add_argument(
      '--merge_bn', action = 'store_true',
      help = 'Choose if merge bn layer. Default is False.')

  # quantize
  parser.add_argument(
      '--quantize', default = 'shift',
      help = 'Choose quantization mode.')
  parser.add_argument(
      '--quantize_w_integer', default = 4, type = int,
      help = 'Specify integer data width of weight.')
  parser.add_argument(
      '--quantize_w', default = 4, type = int,
      help = 'Specify data width of weight.')
  parser.add_argument(
      '--quantize_x_integer', default = 3, type = int,
      help = 'Specify integer data width of input tensor.')
  parser.add_argument(
      '--quantize_x', default = 8, type = int,
      help = 'Specify data width of input tensor.')
  parser.add_argument(
      '--quantize_o_integer', default = 7, type = int,
      help = 'Specify integer data width of input tensor.')
  parser.add_argument(
      '--quantize_o', default = 16, type = int,
      help = 'Specify data width of input tensor.')

  parser.add_argument(
      '--output_directory', default = 'yolo',
      help = 'Output directory.')
  parser.add_argument(
      '--output', default = 'output.dat',
      help = 'Output file name. Default is output_merge.dat')
  parser.set_defaults(merge_bn=False)
  args = parser.parse_args()

  # Path
  image_path = Path('.') / 'fig' / args.img
  ckpt_path = Path('.') / 'ckpt' / args.ckpt
  ckpt_store_path = Path('.') / 'ckpt' / args.ckpt_store
  output_path = Path('.') / 'dat' / args.output_directory / args.output

  # # Read Image
  # img = tf.io.read_file(str(image_path))
  # img = tf.image.decode_jpeg(img, channels = args.img_channels)
  # img = tf.image.resize(img, [args.img_size, args.img_size])
  # img = tf.math.divide(tf.math.add(tf.cast(img, tf.float32), -127), 128)
  # img = tf.reshape(img, [1, args.img_size, args.img_size, args.img_channels])

  # Read image .bin
  img_size = args.img_size
  img_channels = args.img_channels
  img = np.fromfile(image_path, dtype = np.float32)
  img = np.reshape(img, (1, img_channels, img_size, img_size))
  img = tfConverttoTensor(img, dtype=tfFloat32)
  img = tfTranspose(img, perm = [0, 2, 3, 1])

  # Build and load model
  input_tensor_shape = (None, args.img_size, args.img_size, args.img_channels)
  model = GenerateModel(
      args.quantize, args.quantize_w_integer, args.quantize_w,
      args.quantize_x_integer, args.quantize_x)
  model.build(input_tensor_shape)
  if args.ckpt != 'None':
    print('[INFO][inference.py] Load model:', ckpt_path)
    model.load_weights(str(ckpt_path))

  if args.merge_bn:
    print('[INFO][inference.py] Model merge BN layer.')
    # model_merge_bn.build(input_tensor_shape)
    temp = model_merge_bn(img)
    utils.MergeBN(model_merge_bn, model)
    output = model_merge_bn(img)
  else:
    print('[INFO][inference.py] Model doesn\'t merge BN layer.')
    output = model(img)

  # Store output
  if len(output.numpy().shape) == 4:
    utils.Store4DTensor(
        output, output_path, args.quantize_o_integer, args.quantize_o)
  else:
    utils.Store2DTensor(
        output, output_path, args.quantize_o_integer, args.quantize_o)

  # Store output ckpt
  if args.ckpt_store != 'None':
    if args.merge_bn:
      print('[INFO][inference.py] Store merge bn output ckpt:',
          args.ckpt_store)
      model_merge_bn.save_weights(str(ckpt_store_path))
    else:
      print('[INFO][inference.py] Store not merge bn output ckpt:',
          args.ckpt_store)
      model.save_weights(str(ckpt_store_path))
