from keras.models import Model
import keras.backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import concatenate

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def leaky_relu(x):
    """Leaky Rectified Linear activation.
    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """
    # return K.relu(x, alpha=0.1, max_value=None)
    # requires less memory than keras implementation
    alpha = 0.1
    zero = _to_tensor(0.0, x.dtype.base_dtype)
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x = alpha * tf.minimum(x, zero) + tf.maximum(x, zero)
    return x

def bn_acti_conv(
    x, filters, kernel_size=1, stride=1, padding="same", activation="relu"
):
    x = BatchNormalization(scale=True)(x)
    x = Activation(activation)(x)
    x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    return x


def dense_block(x, n, growth_rate, width=4, activation="relu"):
    input_shape = K.int_shape(x)
    c = input_shape[3]
    for i in range(n):
        x1 = x
        x2 = bn_acti_conv(x, growth_rate * width, 1, 1, activation=activation)
        x2 = bn_acti_conv(x2, growth_rate, 3, 1, activation=activation)
        x = concatenate([x1, x2], axis=3)
        c += growth_rate
    return x

def getDenseModel(input_size=(96, 160, 3), activation="relu"):
    """ Description of the function !
    """
    img_in = Input(shape=input_size, name='img_in')

    if activation == "leaky_relu":
        activation = leaky_relu

    growth_rate = 6
    compression = .5

    # convolution part of the model
    # Dense Block 1
    x = MaxPooling2D(pool_size=2, strides=2, padding="same")(img_in)
    x = dense_block(x, 2, growth_rate, 4, activation)
    x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
    x = bn_acti_conv(
        x, int(K.int_shape(x)[3] * compression), 1, 1, activation=activation
    )
    # Dense Block 2

    x = Flatten(name='flattened')(x)

    # fully connected part of the model
    x = Dense(50, use_bias=False, name="fdense_2")(x)
    x = BatchNormalization(name="fnorm_2")(x)
    x = Activation(activation, name="factivation_2")(x)
    x = Dropout(.1, name="fdrop_2")(x)

    out_speed = Dense(2, activation='softmax', name="fc_speed")(x)
    out_dir = Dense(5, activation='softmax', name="fc_direction")(x)

    model = Model(inputs=[img_in], outputs=[out_speed, out_dir])
    return model

if __name__ == "__main__":
    model = getDenseModel()
    model.summary()
