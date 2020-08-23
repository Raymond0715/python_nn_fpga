import tensorflow as tf
from tensorflow.keras import regularizers
from quantization import QuantilizeFnSTE, QuantilizeFnNG, tangent

class QConv2D(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_depth,
            kernel_size,
            strides = [1, 1],
            padding = 'SAME', 
            quantilize = 'full',
            quantilize_w = 32,
            quantilize_x = 32,
            weight_decay = 0.0005,
            use_bias = True,
            name = None,
            alpha = 0):

        super(QConv2D, self).__init__()
        self.kernel_depth = kernel_depth
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.quantilize = quantilize
        self.weight_decay = weight_decay
        self.use_bias = use_bias
        if self.quantilize == 'ste':
            # print('[DEBUG][nn_utils.py] init QConv2D ste')
            self.QuantilizeWeight, self.QuantilizeActivation = \
                    QuantilizeFnSTE(quantilize_w, quantilize_x)
        elif self.quantilize == 'ng':
            # print('[DEBUG][nn_utils.py] init QConv2D ng')
            self.QuantilizeWeight, self.QuantilizeActivation = \
                    QuantilizeFnNG(quantilize_w, quantilize_x)
            self.alpha = alpha # For nature gradient quantilization
        else:
            # print('[DEBUG][nn_utils.py] init QConv2D full')
            pass

    def build(self, input_shape):
        self.filters = self.add_weight(
                shape = [ 
                    self.kernel_size,
                    self.kernel_size, 
                    int(input_shape[-1]),                   
                    self.kernel_depth],
                initializer = tf.keras.initializers.GlorotUniform(),
                regularizer = regularizers.l2(self.weight_decay))

        if self.use_bias:
            self.bias = self.add_weight(
                    shape = self.kernel_depth,
                    initializer = tf.keras.initializers.Zeros())
        
    def call(self, input_tensor):
        if self.quantilize == 'ste':
            # print('[DEBUG][nn_utils.py] QConv2D call ste')
            filters = tf.clip_by_value(self.filters, -1, 1)
            filters = self.QuantilizeWeight(filters)
            input_tensor = self.QuantilizeActivation(input_tensor)
        elif self.quantilize == 'ng':
            # print('[DEBUG][nn_utils.py] QConv2D call ng')
            quantize_filters = self.QuantilizeWeight(self.filters)
            filters = tangent(self.filters, quantize_filters, self.alpha)
            input_tensor_quantilize = self.QuantilizeActivation(input_tensor)
            input_tensor = tangent(
                    input_tensor, input_tensor_quantilize, self.alpha)
        else:
            # print('[DEBUG][nn_utils.py] QConv2D call full')
            filters = self.filters

        output = tf.nn.conv2d(
                input_tensor, filters, self.strides, self.padding)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        return output
