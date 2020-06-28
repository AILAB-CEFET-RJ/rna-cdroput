import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def select_dropout(dropout_opt):
    print(f"Selected Dropout: {dropout_opt}")
    if dropout_opt == 'none':
        return None
    if dropout_opt == 'DropoutErrorBased':
        return DropoutErrorBased()
    if dropout_opt == 'ErrorBasedDropoutZero':
        return ErrorBasedDropoutZero()


def select_scaler(scaler_opt):
    print(f"Selected Scaler: {scaler_opt}")
    if scaler_opt == 'none':
        return None
    if scaler_opt == 'StandardScaler':
        return StandardScaler()
    if scaler_opt == 'MinMaxScaler':
        return MinMaxScaler()


def bit_mask(keep_prob):
    return tf.keras.backend.random_binomial(
        keep_prob.shape, p=keep_prob
    )


def gen_prob_bits(noise):
    m = tf.nn.softmax(noise)
    keep_prob = 1 - m
    bits = bit_mask(keep_prob)

    return bits, keep_prob


def dropout(x, noise, noise_shape=None, seed=None, name=None):
    bits, keep_prob = gen_prob_bits(noise)

    scale = 1 / keep_prob
    ret = x * scale * bits

    return ret


def split_input(inputs):
    errs = inputs[:, 5:]
    inputs = inputs[:, :5]

    return inputs, errs


class DropoutErrorBased(Dropout):
    def __init__(self, **kwargs):
        super(DropoutErrorBased, self).__init__(1.0, dynamic=True, **kwargs)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        inputs, errs = split_input(inputs)

        def dropped_inputs():
            return dropout(inputs, errs, seed=self.seed)

        output = tf_utils.smart_cond(
            training, dropped_inputs, lambda: array_ops.identity(inputs)
        )

        return output


class ErrorBasedDropoutZero(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ErrorBasedDropoutZero, self).__init__(**kwargs)
        print('successfuly constructed')

    def call(self, inputs, training=None):
        NUM_BANDS = 5
        errs = inputs[:,NUM_BANDS:]
        ugriz = inputs[:,:NUM_BANDS]
        #print('call.ugriz:', ugriz)
        #print('call.errs:', errs)

        def droppedout_ugriz(ugriz, errs):
          #print('errs:', errs)
          l2_norm_errs = tf.linalg.normalize(errs)
          #print('l2_norm_errs:', l2_norm_errs[0])
          ones = tf.ones(shape=(1,NUM_BANDS),dtype=tf.dtypes.float32)
          #print('ones:', ones)
          keep_probs = tf.math.subtract(ones, l2_norm_errs[0])
          rnd_unif = tf.random.uniform(shape=(1,NUM_BANDS), dtype=tf.dtypes.float32)
          mask = tf.math.greater(keep_probs, rnd_unif)
          casted_mask = tf.cast(mask, dtype=tf.dtypes.float32)
          masked_input = tf.math.multiply(ugriz, casted_mask)
          return masked_input

        if training:
          output = droppedout_ugriz(ugriz, errs)
        else:
          output = ugriz

        return output