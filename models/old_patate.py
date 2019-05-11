from keras.models import Model
from keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout,
    Convolution2D,
    Activation,
    BatchNormalization,
)

def getOldModel(input_size=(96, 160, 3)):
    """ Description of the function !
    """
    img_in = Input(shape=input_size, name='img_in')

    # convolution part of the model
    x = Convolution2D(4, (5,5), strides=(2,2), use_bias=False)(img_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Convolution2D(8, (5,5), strides=(2,2), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Convolution2D(16, (5,5), strides=(2,2), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Flatten(name='flattened')(x)

    # fully connected part of the model
    x = Dense(100, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(.2)(x)
    x = Dense(100, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(.2)(x)
    x = Dense(100, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(.2)(x)

    out_speed = Dense(2, activation='softmax', name="speed")(x)
    out_dir = Dense(5, activation='softmax', name="direction")(x)

    model = Model(inputs=[img_in], outputs=[out_speed, out_dir])
    return model

if __name__ == "__main__":
    model = getOldModel()
    model.summary()
