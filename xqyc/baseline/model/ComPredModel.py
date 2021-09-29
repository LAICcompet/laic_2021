import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, ReLU, Dropout, MaxPool1D, Conv1D, Reshape, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers, losses, metrics, optimizers


class ComPredModel:
    @staticmethod
    def cnn_regression(config):
        inputs_left = Input(shape=(config["max_length"], ))
        x = inputs_left
        x = Embedding(config["vocab_size"], config["embedding_dim"], input_length=config["max_length"])(x)
        filters_left = 32
        for i in range(3):
            x = Conv1D(filters=filters_left, kernel_size=2, activation=ReLU(), padding='same')(x)
            x = MaxPool1D()(x)
            filters_left *= 2
        y = Flatten()(x)
        y = Dropout(config['dropout'])(y)
        outputs = Dense(1)(y)
        model = Model([inputs_left], outputs)
        model.summary()
        model.compile(loss='mae', optimizer=optimizers.Adam(), metrics=['mae'])
        return model
