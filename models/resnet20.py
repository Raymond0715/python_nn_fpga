import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers

from nn_utils import QConv2D
from quantization import QuantilizeFnSTE, QuantilizeFnNG
from main import args

'''
For 1 bit model, activation function use 'tf.clip_by_value', others use 'ReLU'
'''

class ResnetUnitL2(tf.keras.layers.Layer):
    def __init__(
            self, 
            outputs_depth, 
            strides = 1, 
            first = False, 
            quantilize   = 'full',
            quantilize_w = 32,
            quantilize_x = 32,
            weight_decay = 0.0005,
            alpha        = 0):

        super(ResnetUnitL2, self).__init__()
        self.strides = strides
        self.first = first
        self.quantilize   = quantilize
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x
        self.alpha        = alpha

        self.conv2a = QConv2D(
                outputs_depth, 3, strides, 
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = weight_decay, 
                use_bias     = False,
                alpha        = self.alpha)
            
        self.bn2a = BatchNormalization()
        
        self.conv2b = QConv2D(
                outputs_depth, 3, 
                quantilize   = self.quantilize,
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = weight_decay, 
                use_bias     = False,
                alpha        = self.alpha)

        self.bn2b = BatchNormalization()

        if self.first:
            self.conv_shortcut = QConv2D(
                    outputs_depth, 1, strides, 
                    # quantilize   = self.quantilize, 
                    quantilize   = 'full', 
                    quantilize_w = self.quantilize_w,
                    quantilize_x = self.quantilize_x,
                    weight_decay = weight_decay, 
                    use_bias     = False,
                    alpha        = self.alpha)
            self.bn_shortcut = BatchNormalization()

    def call(self, input_tensor):
        x = input_tensor
        if self.first:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = x

        x = self.conv2a(x)
        x = self.bn2a(x)
        if self.quantilize != 'full' and self.quantilize_x == 1:
            # print('[DEBUG][models/resnet20.py] resnet unit 1 call quantilize')
            x = tf.clip_by_value(x, -1, 1)
        else:
            # print('[DEBUG][models/resnet20.py] resnet unit 1 call full')
            x = Activation('relu')(x)

        x = self.conv2b(x)
        x = self.bn2b(x)

        x += shortcut
        if self.quantilize != 'full' and self.quantilize_x == 1:
            # print('[DEBUG][models/resnet20.py] resnet unit 2 call quantilize')
            x = tf.clip_by_value(x, -1, 1)
        else:
            # print('[DEBUG][models/resnet20.py] resnet unit 2 call full')
            x = Activation('relu')(x)
        return x


class ResnetBlockL2(tf.keras.layers.Layer):
    def __init__(
            self,
            num_units,
            outputs_depth,
            strides,
            quantilize   = 'full',
            quantilize_w = 32,
            quantilize_x = 32,
            weight_decay = 0.0005,
            alpha        = 0):

        super(ResnetBlockL2, self).__init__()
        self.num_units    = num_units
        self.quantilize   = quantilize
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x
        self.alpha        = alpha

        self.units = []
        self.units.append(
                ResnetUnitL2(
                    outputs_depth, strides = strides, first = True, 
                    quantilize   = quantilize, 
                    quantilize_w = self.quantilize_w,
                    quantilize_x = self.quantilize_x,
                    weight_decay = weight_decay,
                    alpha        = self.alpha))
        for i in range(1, self.num_units):
            self.units.append(
                    ResnetUnitL2( 
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


class Resnet20(tf.keras.Model):
    def __init__(
            self,
            weight_decay,
            class_num,
            quantilize   = 'full', 
            quantilize_w = 32,
            quantilize_x = 32,
            num_epochs   = 250):

        super(Resnet20, self).__init__(name = '')
        self.weight_decay = weight_decay
        self.quantilize   = quantilize
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x
        self.alpha = 0
        self.num_epochs = num_epochs

        self.conv_first = QConv2D(
                16, 3, quantilize = 'full', weight_decay = self.weight_decay, 
                use_bias = False, alpha = self.alpha)
        self.bn_first = BatchNormalization()
        self.dense = Dense(
                class_num, 
                kernel_regularizer = regularizers.l2(self.weight_decay))
        self.block1 = ResnetBlockL2(
                3, 16, 1, 
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block2 = ResnetBlockL2(
                3, 32, 2, 
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block3 = ResnetBlockL2(
                3, 64, 2, 
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)

    def call(self, input_tensor):
        x = input_tensor
        x = self.conv_first(x)
        x = self.bn_first(x)
        if self.quantilize != 'full' and self.quantilize_x == 1:
            # print('[DEBUG][models/resnet20.py] resnet call quantilize')
            x = tf.clip_by_value(x, -1, 1)
        else:
            # print('[DEBUG[models/resnet20.py] resnet call full')
            x = Activation('relu')(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = MaxPooling2D(pool_size = 8)(x)
        x = Flatten()(x)
        x = self.dense(x)

        return Activation('softmax')(x)

# weight_decay = 0.0005
# quantization = False

# weight_decay = 0.0005
# quantilize   = True
# quantilize_w = 1
# quantilize_x = 1
# class_num = 10

class_num    = args.class_num
quantilize   = args.quantilize
quantilize_w = args.quantilize_w
quantilize_x = args.quantilize_x
weight_decay = args.weight_decay
num_epochs   = args.num_epochs

model = Resnet20(
        weight_decay, class_num, quantilize = quantilize,
        quantilize_w = quantilize_w, quantilize_x = quantilize_x,
        num_epochs = num_epochs)
