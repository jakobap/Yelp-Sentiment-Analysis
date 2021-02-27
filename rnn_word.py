from main import ModelFramework
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


class RNNword(ModelFramework):
    """
    Top Validation Performance [loss, accuracy]: [0.3811458349227905, 0.846666693687439]
    """
    def __init__(self, data_file, epochs, batch_size, dropout):
        super().__init__(data_file, epochs, batch_size)
        self.c_filt_size = 2
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self._model()

    def _model(self):
        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.Bidirectional(tf.keras.layers.GRU(1024, return_sequences=True))(embedded_sequences)
        # x = layers.Bidirectional(tf.keras.layers.GRU(124))(x)
        # x = layers.Dense(64)(x)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)

        # int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        # embedded_sequences = self.embedding_layer(int_sequences_input)
        # x = layers.Bidirectional(tf.keras.layers.GRU(10, return_sequences=True))(embedded_sequences)
        # x = layers.Bidirectional(tf.keras.layers.GRU(5))(x)
        # x = layers.Dropout(self.dropout)(x)
        # preds = layers.Dense(1, activation="sigmoid")(x)
        # self.model = keras.Model(int_sequences_input, preds)


if __name__ == '__main__':
    rnn_word = RNNword(data_file="data/yelp_labelled.txt", epochs=10, batch_size=64, dropout=.5)
    print(rnn_word.model.summary())
    rnn_word.fit()

    print('hello worlds')