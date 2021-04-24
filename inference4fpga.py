from pathlib import Path
import argparse
import pdb
import numpy as np

import tensorflow as tf
from models.alexnet_fpga import model
from models.alexnet_merge_bn import model_merge_bn

import nn_utils
from utils import MergeBN, Store4DTensor, Store2DTensor


parser = argparse.ArgumentParser(description = 'CNN inference for fpga.'
    'Including merge bn layer. Store output at the end.')
parser.add_argument(
    '--merge_bn', default = False, action = 'store_true',
    help = 'Choose if merge bn layer. Default is False.')
parser.add_argument(
    '--image', default = 'ILSVRC2012_val_00050000.JPEG',
    help = 'Image name. Default is ILSVRC2012_val_00050000.JPEG')
parser.add_argument(
    '--ckpt', default = 'alexnet_random_2.h5',
    help = 'Checkpoint name. Default is alexnet_random.h5')
parser.add_argument(
    '--dat', default = 'output_merge.dat',
    help = 'Output file name. Default is output_merge.dat')
parser.add_argument(
    '--output_ckpt', default = 'None',
    help = 'Output ckpt name. Default is None.')
args = parser.parse_args()

# Path
# image_path = './fig/ILSVRC2012_val_00050000.JPEG'
image_path  = Path('.') / 'fig' / args.image
ckpt_path   = Path('.') / 'ckpt' / args.ckpt
output_path = Path('.') / 'dat' / args.dat
output_ckpt = Path('.') / 'ckpt' / args.output_ckpt

# Read Image
img = tf.io.read_file(str(image_path))
img = tf.image.decode_jpeg(img, channels = 3)
img = tf.image.resize(img, [224, 224])
img = tf.math.divide(tf.math.add(tf.cast(img, tf.float32), -127), 128)
img = tf.reshape(img, [1, 224, 224, 3])

# Build and load model
input_tensor_shape = (None, 224, 224, 3)
model.build(input_tensor_shape)
model.load_weights(str(ckpt_path))
if args.merge_bn:
  print('[INFO][inference4fpga.py] Model merge BN layer.')
  # model_merge_bn.build(input_tensor_shape)
  temp = model_merge_bn(img)
  MergeBN(model_merge_bn, model)
  pdb.set_trace()
  output = model_merge_bn(img)
else:
  print('[INFO][inference4fpga.py] Model doesn\'t merge BN layer.')
  output = model(img)

# Store output
if len(output.numpy().shape) == 4:
  Store4DTensor(output, output_path)
else:
  Store2DTensor(output, output_path)

# Store output ckpt
if args.output_ckpt != 'None':
  print('[INFO][inference4fpga.py] Store output ckpt:', args.output_ckpt)
  model_merge_bn.save_weights(str(output_ckpt))
