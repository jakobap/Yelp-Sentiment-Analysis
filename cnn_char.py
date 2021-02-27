from main import ModelFramework, CharFramework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

import numpy as np


class CNNChar(CharFramework):
    """
    Top Validation Performance [loss, accuracy]: [0.4801945388317108, 0.8133333325386047]
    """
    def __init__(self, data_file, epochs, batch_size, dropout):
        super().__init__(data_file, epochs, batch_size, dropout)
        self.epochs = epochs  # Model Training Epochs
        self.batch_size = batch_size  # Training Batch Size
        self.dropout = dropout  # Dropout Probability for dropout layers
        self.model_name = 'CNNChar'  # Model Name for saving purpose

        self._model()  # Calling Model Architecture

    def _model(self):
        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.Conv1D(filters=32, kernel_size=4, strides=1, activation="relu")(embedded_sequences)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)
        print('##### Model Architecture Check #####')


if __name__ == '__main__':
    cnn_char = CNNChar(data_file="data/yelp_labelled.txt", epochs=15, batch_size=64, dropout=.3)
    print(cnn_char.model.summary())
    cnn_char.fit()

    print('hello worlds')