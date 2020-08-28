import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers

from nn_utils import QConv2D
from quantization import QuantilizeFnSTE, QuantilizeFnNG 
from main import args

class AlexNet(tf.keras.Model):
    def __init__(
            self,
            weight_decay,
            quantilize   = 'full',
            quantilize_w = 32,
            quantilize_x = 32,
            num_epochs   = 250):
        
        super(AlexNet, self).__init__(name = '')
        self.weight_decay = weight_decay
        self.quantilize   = quantilize 
        self.quantilize_w = quantilize_w
        self.quantilize_x = quantilize_x 
        self.alpha        = 0
        self.num_epochs   = num_epochs

        self.conv1 = QConv2D(64, 11, 4, padding = 'VALID', bias = False)
