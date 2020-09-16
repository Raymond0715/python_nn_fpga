import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization

from nn_utils import QConv2D

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
            alpha        = None):

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
                    quantilize   = self.quantilize, 
                    # quantilize   = 'full', 
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
            alpha        = None):

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
                    quantilize   = self.quantilize, 
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
