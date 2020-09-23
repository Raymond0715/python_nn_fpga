from nn_utils import QConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation


# class VGGUnit(tf.keras.Model):
class VGGUnit(tf.keras.layers.Layer):
    def __init__(
            self,
            outputs_depth,
            quantilize   = 'full',
            quantilize_w = 32,
            quantilize_x = 32,
            weight_decay = 0.0005,
            alpha        = None ):

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
                weight_decay = weight_decay,
                use_bias     = False,
                alpha        = self.alpha)
        self.bn = BatchNormalization()

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        if self.quantilize == 'full' and self.quantilize_x == 1:
            x = tf.clip_by_value(x, -1, 1)
        else:
            x = Activation('relu')(x)

        return x


# class VGGBlock(tf.keras.Model):
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
            alpha        = None):

        super(VGGBlock, self).__init__()
        self.num_units    = num_units
        self.quantilize   = quantilize
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x
        self.first        = first
        self.alpha        = alpha

        self.units = []
        if self.first:
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
