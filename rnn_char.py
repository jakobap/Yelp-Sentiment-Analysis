from main import ModelFramework, CharFramework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

import numpy as np


class RNNChar(CharFramework):
    """
    Top Validation Performance [loss, accuracy]: [0.6910178065299988, 0.6600000262260437]
    """
    def __init__(self, data_file, epochs, batch_size, dropout):
        super().__init__(data_file, epochs, batch_size, dropout)
        self.epochs = epochs  # Model Training Epochs
        self.batch_size = batch_size  # Training Batch Size
        self.dropout = dropout  # Dropout Probability for dropout layers
        self.model_name = 'RNNChar'  # Model Name for saving purpose

        self._model()  # Calling Model Architecture

    def _model(self):
        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.LSTM(32, return_sequences=True)(embedded_sequences)
        x = layers.LSTM(4)(x)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)


if __name__ == '__main__':
    cnn_char = RNNChar(data_file="data/yelp_labelled.txt", epochs=10, batch_size=16, dropout=.2)
    print(cnn_char.model.summary())
    cnn_char.fit()

    print('hello worlds')