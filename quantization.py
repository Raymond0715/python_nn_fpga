import tensorflow as tf
import pdb

def QuantilizeFn(Wbit, Abit):

    def Round2Fixed(x, integer=16, k=32):
        assert integer >= 1, integer
        fraction = k - integer
        bound = tf.math.pow(2.0, integer - 1)
        n = tf.math.pow(2.0, fraction)
        min_val = -bound
        max_val = bound
        round = tf.round(x * n) / n
        clipped_value = tf.clip_by_value(round, min_val, max_val)
        return clipped_value

    def RoundPower2(x, k=4):
        bound = tf.math.pow(2.0, k - 1)
        min_val = tf.math.pow(2.0, -bound + 1.0)
        s = tf.sign(x)
        x = tf.clip_by_value(tf.math.abs(x), min_val, 1.0)
        p = tf.round(tf.math.log(x) / tf.math.log(2.))
        return s * tf.math.pow(2.0, p)

    def CeilPower2(x):
        p = tf.math.ceil(tf.math.log(x) / tf.math.log(2.))
        return tf.math.pow(2.0, p)

    @tf.custom_gradient
    def QuantilizeWeight(w):
        if Wbit == 1:   # BNN
            # mean = tf.reduce_mean(tf.abs(w))
            # E = tf.stop_gradient(mean)
            # output = tf.sign(w / E) * E
            output = tf.sign(w)
        elif Wbit == 32:
            output = w
        else:   # QNN
            max = tf.reduce_max(tf.abs(w))
            w = w / max
            output =  max * Round2Fixed(w, 1, Wbit)

        def Grad(dy):
            return dy

        return output, Grad

    @tf.custom_gradient
    def QuantilizeActivation(x):
        if Abit == 1:   # BNN
            # mean = tf.reduce_mean(tf.abs(x))
            # E = tf.stop_gradient(mean)
            # output = tf.sign(x / E) * E
            output = tf.sign(x)
        elif Abit == 32:
            output = x
        else:   # QNN
            max = tf.reduce_max(tf.abs(x))
            x = x / max
            output = max * Round2Fixed(x, 1, Abit)

        def Grad(dy):
            return dy

        return output, Grad

    return QuantilizeWeight, QuantilizeActivation


def tangent(x, x_quantilize, alpha):
    # print('[DEBUG][quantization.py] alpha:', alpha)
    # return x - (x - x_quantilize) * alpha.value()
    filters = \
            x_quantilize - alpha * tf.math.tanh(x - x_quantilize)
    return filters
