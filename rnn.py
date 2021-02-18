from main import ModelFramework
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


class RNN(ModelFramework):
    """
    Best Validatino Accuracy so far: 80.5
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
        x = layers.Bidirectional(tf.keras.layers.LSTM(64, activation='relu'))(embedded_sequences)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(embedded_sequences)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(len(self.class_names), activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)


if __name__ == '__main__':
    rnn = RNN(data_file="data/yelp_labelled.txt", epochs=25, batch_size=64, dropout=.7)
    print(rnn.model.summary())
    rnn.fit()

    print('hello worlds')