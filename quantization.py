import tensorflow as tf


@tf.custom_gradient
def QSign(x):
  output = tf.sign(x)

  def Grad(dy):
    return dy

  return output, Grad


@tf.custom_gradient
def QRound(x):
  output = tf.round(x)

  def Grad(dy):
    return dy

  return output, Grad


def Round2Fixed(x, integer=16, k=32):
  assert integer >= 1, integer
  fraction = k - integer
  bound = tf.math.pow(2.0, integer - 1)
  n = tf.math.pow(2.0, fraction)
  min_val = -bound
  max_val = bound-1
  # ??? Is this round function correct
  x_round = QRound(x * n) / n
  clipped_value = tf.clip_by_value(x_round, min_val, max_val)
  return clipped_value


# def RoundPower2(x, k=4):
  # bound = tf.math.pow(2.0, k - 1)
  # min_val = tf.math.pow(2.0, -bound + 1.0)
  # s = tf.sign(x)
  # # x = tf.clip_by_value(tf.math.abs(x), min_val, 1.0)
  # x = tf.clip_by_value(tf.math.abs(x * 8), min_val, 1.0)
  # p = QRound(tf.math.log(x) / tf.math.log(2.))
  # return s * tf.math.pow(2.0, p)


def RoundPower2Exp(x, k=4):
  bound = tf.math.pow(2.0, k - 1)
  min_val = tf.math.pow(2.0, -bound + 1.0)
  s = tf.sign(x)

  # Temporary. `*8` during inference and don't change during convert.
  # In fact, it should be `/8` during convert and don't change during inference.
  x = tf.clip_by_value(tf.math.abs(x), min_val, 1.0)
  # x = tf.clip_by_value(tf.math.abs(x * 8), min_val, 1.0)

  p = QRound(tf.math.log(x) / tf.math.log(2.))
  return s, p


def RoundPower2(x, k=4):
  s, p = RoundPower2Exp(x, k)
  return s * tf.math.pow(2.0, p)


def Round2Int(x, integer=16, k=32):
  assert integer >= 1, integer
  fraction = k - integer
  bound = tf.math.power(2.0, k - 1)
  n = tf.math.power(2.0, fraction)
  min_val = -bound
  max_val = bound-1
  x_round = tf.math.floor(x * n)
  clipped_value = tf.clip_by_value(x_round, min_val, max_val).astype(np.int32)
  return clipped_value


def QuantizeFn(w_int, a_int, w_bit, a_bit, flag_shift=True):
  # def CeilPower2(x):
    # p = tf.math.ceil(tf.math.log(x) / tf.math.log(2.))
    # return tf.math.pow(2.0, p)

  def QuantizeWeight(w):
    if w_bit == 1:   # BNN
      # mean = tf.reduce_mean(tf.abs(w))
      # E = tf.stop_gradient(mean)
      # output = QSign(w / E) * E
      output = QSign(w)
    elif w_bit == 32:
      output = w
    else:   # QNN
      # max = tf.reduce_max(tf.abs(w))
      # w = w / max
      # output =  max * Round2Fixed(w, w_int, w_bit)
      if flag_shift:
        output = RoundPower2(w, w_bit)
      else:
        output = Round2Fixed(w, w_int, w_bit)

    return output

  def QuantizeActivation(x):
    if a_bit == 1:   # BNN
      # mean = tf.reduce_mean(tf.abs(x))
      # E = tf.stop_gradient(mean)
      # output = QSign(x / E) * E
      output = QSign(x)
    elif a_bit == 32:
      output = x
    else:   # QNN
      output = Round2Fixed(x, a_int, a_bit)

    return output

  return QuantizeWeight, QuantizeActivation


def tangent(x, x_quantilize, alpha):
  # print('[DEBUG][quantization.py] alpha:', alpha)
  # return x - (x - x_quantilize) * alpha.value()
  filters = \
      x_quantilize - alpha * tf.math.tanh(x - x_quantilize)
  return filters
