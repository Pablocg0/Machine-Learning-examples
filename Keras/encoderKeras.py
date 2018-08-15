from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping


def encoderTrain(x_train, data_test):

    encoding_dim = 26

    input_img = Input(shape=(42588,))

    encoded = Dense(encoding_dim, activation = 'tanh')(input_img)
    decoded = Dense(42588, activation= 'sigmoid')(encoded)

    autoencoder =  Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy')
    stop_callback = EarlyStopping(monitor='val_loss',min_delta=.0001,patience=150, mode='auto')
    autoencoder.fit(x_train, x_train, epochs =1000, batch_size=256, validation_split=0.05, callbacks=[stop_callback])

    encoded_img = encoder.predict(data_test)
    print(encoded_img.shape)
    decoded_img = decoder.predict(encoded_img)
    return decoded_img
