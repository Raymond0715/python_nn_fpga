import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers

from main import args
from models.vgg_utils import VGGBlock
import pdb

class VGG13(tf.keras.Model):
    def __init__(
            self,
            weight_decay,
            class_num,
            quantilize   = 'full', 
            quantilize_w = 32,
            quantilize_x = 32,
            num_epochs   = 250):

        super(VGG16, self).__init__(name = '')
        self.weight_decay = weight_decay
        self.quantilize   = quantilize
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x
        self.alpha        = tf.Variable(0., trainable = False)
        self.num_epochs   = num_epochs

        self.dense1 = Dense(
                512,
                use_bias = False,
                kernel_regularizer = regularizers.l2(self.weight_decay))
        self.bn1 = BatchNormalization()
        self.dense2 = Dense(
                class_num,
                kernel_regularizer = regularizers.l2(self.weight_decay))
        self.block1 = VGGBlock(
                2, 64,
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                first        = True,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block2 = VGGBlock(
                2, 128,
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block3 = VGGBlock(
                2, 256,
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block4 = VGGBlock(
                2, 512,
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block5 = VGGBlock(
                2, 512,
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)

    def call(self, input_tensor):
        x = input_tensor

        x = self.block1(x)
        x = MaxPooling2D(pool_size = 2, strides = 2)(x)
        x = self.block2(x)
        x = MaxPooling2D(pool_size = 2, strides = 2)(x)
        x = self.block3(x)
        x = MaxPooling2D(pool_size = 2, strides = 2)(x)
        x = self.block4(x)
        x = MaxPooling2D(pool_size = 2, strides = 2)(x)
        x = self.block5(x)
        x = MaxPooling2D(pool_size = 2, strides = 2)(x)

        x = Flatten()(x)
        x = self.dense1(x)
        x = self.bn1(x)
        if self.quantilize != 'full' and self.quantilize_x == 1:
            x = tf.clip_by_value(x, -1, 1)
        else:
            x = Activation('relu')(x)

        x = self.dense2(x)

        return Activation('softmax')(x)

class_num    = args.class_num
quantilize   = args.quantilize
quantilize_w = args.quantilize_w
quantilize_x = args.quantilize_x
weight_decay = args.weight_decay
num_epochs   = args.num_epochs

model = VGG13(
        weight_decay, class_num, quantilize = quantilize,
        quantilize_w = quantilize_w, quantilize_x = quantilize_x, 
        num_epochs = num_epochs)
