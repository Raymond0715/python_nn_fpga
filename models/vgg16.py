import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers

from nn_utils import QConv2D
from quantization import QuantilizeFnSTE, QuantilizeFnNG 
from main import args

class VGGUnit(tf.keras.layers.Layer):
    def __init__(
            self,
            outputs_depth,
            quantilize   = 'full',
            quantilize_w = 32,
            quantilize_x = 32,
            weight_decay = 0.0005,
            alpha        = 0 ):

        super(VGGUnit, self).__init__()
        self.quantilize   = quantilize
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x
        self.alpha        = alpha
        
        self.conv = QConv2D(
                outputs_depth, 3, 
                quantilize   = self.quantilize,
                quantilize_w = self.quantilize_w, 
                quantilize_x = self.quantilize_x,
                alpha        = self.alpha)
        self.bn = BatchNormalization()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        if self.quantilize != 'full' and self.quantilize_x == 1:
            # print('[DEBUG][models/vgg16.py] VGG init quantilize')
            x = tf.clip_by_value(x, -1, 1)
        else:
            # print('[DEBUG][models/vgg16.py] VGG init full')
            x = Activation('relu')(x)

        return x

class VGGBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            num_units,
            outputs_depth,
            quantilize   = 'full',
            quantilize_w = 32,
            quantilize_x = 32,
            first        = False,
            weight_decay = 0.0005,
            alpha        = 0):

        super(VGGBlock, self).__init__()
        self.num_units    = num_units
        self.quantilize   = quantilize
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x
        self.first        = first
        self.alpha        = alpha

        self.units = []
        if self.first:
            # print('[DEBUG][models/vgg16.py] Block first')
            self.units.append(
                    VGGUnit( 
                        outputs_depth, 
                        quantilize   = 'full', 
                        weight_decay = weight_decay,
                        alpha        = self.alpha))
        else:
            self.units.append(
                    VGGUnit( 
                        outputs_depth, 
                        quantilize   = self.quantilize, 
                        quantilize_w = self.quantilize_w,
                        quantilize_x = self.quantilize_x,
                        weight_decay = weight_decay,
                        alpha        = self.alpha))

        for i in range(1, self.num_units):
            self.units.append(
                    VGGUnit( 
                        outputs_depth, 
                        quantilize   = self.quantilize, 
                        quantilize_w = self.quantilize_w,
                        quantilize_x = self.quantilize_x,
                        weight_decay = weight_decay,
                        alpha        = self.alpha))

    def call(self, input_tensor):
        x = input_tensor
        for i in range(self.num_units):
            x = self.units[i](x)

        return x

class VGG16(tf.keras.Model):
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
        self.alpha = 0
        self.num_epochs = num_epochs

        self.dense1 = Dense(
                512,
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
                3, 256,
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block4 = VGGBlock(
                3, 512,
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block5 = VGGBlock(
                3, 512,
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
            # print('[DEBUG][models/vgg16.py] VGG call quantilize')
            x = tf.clip_by_value(x, -1, 1)
        else:
            # print('[DEBUG][models/vgg16.py] VGG call full')
            x = Activation('relu')(x)

        x = self.dense2(x)

        return Activation('softmax')(x)

class_num    = args.class_num
quantilize   = args.quantilize
quantilize_w = args.quantilize_w
quantilize_x = args.quantilize_x
weight_decay = args.weight_decay
num_epochs   = args.num_epochs

model = VGG16(
        weight_decay, class_num, quantilize = quantilize,
        quantilize_w = quantilize_w, quantilize_x = quantilize_x, 
        num_epochs = num_epochs)
