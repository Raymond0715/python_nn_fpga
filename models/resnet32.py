import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers

from models.resnet_utils import ResnetBlockL2
from nn_utils import QConv2D
from main import args

class Resnet32(tf.keras.Model):
    def __init__(
            self,
            weight_decay,
            class_num,
            quantilize   = 'full', 
            quantilize_w = 32,
            quantilize_x = 32,
            num_epochs   = 250):

        super(Resnet32, self).__init__(name = '')
        self.weight_decay = weight_decay
        self.quantilize   = quantilize
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x
        self.alpha = tf.Variable(0., trainable = False)
        self.num_epochs = num_epochs

        self.conv_first = QConv2D(
                16, 3, quantilize = 'full', weight_decay = self.weight_decay, 
                use_bias = False, alpha = self.alpha)
        self.bn_first = BatchNormalization()
        self.dense = Dense(
                class_num, 
                kernel_regularizer = regularizers.l2(self.weight_decay))
        self.block1 = ResnetBlockL2(
                5, 16, 1, 
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block2 = ResnetBlockL2(
                5, 32, 2, 
                quantilize   = self.quantilize, 
                quantilize_w = self.quantilize_w,
                quantilize_x = self.quantilize_x,
                weight_decay = self.weight_decay,
                alpha        = self.alpha)
        self.block3 = ResnetBlockL2(
                5, 64, 2, 
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


class_num    = args.class_num
quantilize   = args.quantilize
quantilize_w = args.quantilize_w
quantilize_x = args.quantilize_x
weight_decay = args.weight_decay
num_epochs   = args.num_epochs

model = Resnet32(
        weight_decay, class_num, quantilize = quantilize,
        quantilize_w = quantilize_w, quantilize_x = quantilize_x,
        num_epochs = num_epochs)
