import tensorflow as tf

def QuantilizeFnSTE(Wbit, Abit):
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
            mean = tf.reduce_mean(tf.abs(w))
            E = tf.stop_gradient(mean)
            output = tf.sign(w / E) * E
            # output = tf.sign(w)
        elif Wbit == 32:
            output = w
        else:   # QNN
            max = CeilPower2(tf.reduce_max(tf.abs(w)))
            w = w / max
            output =  max * RoundPower2(w, Wbit)
        # output = w

        def Grad(dy):
            return dy

        return output, Grad

    @tf.custom_gradient
    def QuantilizeActivation(x):
        if Abit == 1:   # BNN
            mean = tf.reduce_mean(tf.abs(x))
            E = tf.stop_gradient(mean)
            output = tf.sign(x / E) * E
            # output = tf.sign(x)
        elif Abit == 32:
            output = x
        else:   # QNN
            max = CeilPower2(tf.reduce_max(tf.abs(x)))
            x = x / max
            output = max * RoundPower2(x, Abit)
        # output = x

        def Grad(dy):
            return dy

        return output, Grad

    return QuantilizeWeight, QuantilizeActivation


def QuantilizeFnNG(Wbit, Abit):
    def QuantilizeWeight(w):
        if Wbit == 1:   # BNN
            mean = tf.reduce_mean(tf.abs(w))
            E = tf.stop_gradient(mean)
            output = tf.sign(w / E) * E
        elif Wbit == 32:
            output = w
        else:   # QNN
            max = CeilPower2(tf.reduce_max(tf.abs(w)))
            w = w / max
            output =  max * RoundPower2(w, Wbit)

        return output

    def QuantilizeActivation(x):
        if Abit == 1:   # BNN
            mean = tf.reduce_mean(tf.abs(x))
            E = tf.stop_gradient(mean)
            output = tf.sign(x / E) * E
        elif Abit == 32:
            output = x
        else:   # QNN
            max = CeilPower2(tf.reduce_max(tf.abs(x)))
            x = x / max
            output = max * RoundPower2(x, Abit)

        return output

    return QuantilizeWeight, QuantilizeActivation


def tangent(x, x_quantilize, alpha):
    return x - (x - x_quantilize) * alpha
