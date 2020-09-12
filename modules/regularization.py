import tensorflow as tf
from tensorflow.keras.layers import Layer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def select_dropout(dropout_opt):
    print(f"Selected Dropout: {dropout_opt}")
    if dropout_opt == 'none' or dropout_opt is None:
        return None
    if dropout_opt == 'ErrorBasedDropoutZero':
        return ErrorBasedDropoutZero()
    if dropout_opt == 'ErrorBasedDropoutIR':
        return ErrorBasedDropoutIR()
    if dropout_opt == 'ErrorBasedDropoutDT':
        return ErrorBasedDropoutDT()
    if dropout_opt == 'ErrorBasedInvertedDropout':
        return ErrorBasedInvertedDropout()


def select_scaler(scaler_opt):
    print(f"Selected Scaler: {scaler_opt}")
    if scaler_opt == 'none' or scaler_opt is None:
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


def dropout(x, noise):
    bits, keep_prob = gen_prob_bits(noise)

    scale = 1 / keep_prob
    ret = x * scale * bits

    return ret


def split_input(inputs):
    errs = inputs[:, 5:]
    inputs = inputs[:, :5]

    return inputs, errs


class ErrorBasedDropoutZero(Layer):
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


class ErrorBasedDropoutIR(Layer):
    def __init__(self, **kwargs):
        super(ErrorBasedDropoutIR, self).__init__(**kwargs)
        print('ErrorBasedDropoutIR')

    def call(self, inputs, training=None):
        NUM_BANDS = 5
        ugriz = inputs[:,:NUM_BANDS]
        errs = inputs[:,NUM_BANDS:2*NUM_BANDS]
        expErrs = inputs[:,2*NUM_BANDS:]

        def droppedout_ugriz(ugriz, errs):
          ones = tf.ones(shape=(1,NUM_BANDS),dtype=tf.dtypes.float32)
          sfmax = tf.nn.softmax(tf.math.divide(tf.math.subtract(errs, expErrs), errs))
          keep_probs = tf.math.subtract(ones, sfmax[0])
          print('keep_probs:', keep_probs)
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

class ErrorBasedDropoutDT(ErrorBasedDropoutIR):
    def __init__(self, **kwargs):
        super(ErrorBasedDropoutDT, self).__init__(**kwargs)
        print('ErrorBasedDropoutDT')


class ErrorBasedInvertedDropout(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ErrorBasedInvertedDropout, self).__init__(**kwargs)
        print('ErrorBasedInvertedDropout')

    def call(self, inputs, training=None):
        NUM_BANDS = 5
        ugriz = inputs[:, :NUM_BANDS]
        errs = inputs[:, NUM_BANDS:2 * NUM_BANDS]
        expErrs = inputs[:, 2 * NUM_BANDS:]
        #print('call.ugriz:', ugriz)
        #print('call.errs:', errs)
        #print('call.expErrs:', expErrs)

        def droppedout_ugriz(ugriz, errs):
            ones = tf.ones(shape=(1, NUM_BANDS), dtype=tf.dtypes.float32)
            sfmax = tf.nn.softmax(tf.math.divide(tf.math.subtract(errs, expErrs), errs))
            keep_probs = tf.math.subtract(ones, sfmax[0])
            #print('keep_probs:', keep_probs)
            rnd_unif = tf.random.uniform(shape=(1, NUM_BANDS), dtype=tf.dtypes.float32)
            mask = tf.math.greater(keep_probs, rnd_unif)
            casted_mask = tf.cast(mask, dtype=tf.dtypes.float32)

            masked_input = tf.math.multiply(ugriz, casted_mask)
            #print('before scaling: ', masked_input)

            keep_probs_mean = tf.math.reduce_mean(keep_probs)
            #print('keep_probs_mean: ', keep_probs_mean)

            masked_input = tf.math.divide(masked_input, keep_probs_mean)
            #print('after scaling: ', masked_input)

            return masked_input

        if training:
            output = droppedout_ugriz(ugriz, errs)
        else:
            output = ugriz

        return output
