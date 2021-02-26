from main import ModelFramework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

import numpy as np


class CNN(ModelFramework):
    """
    Top Score:
    Score: [0.35780662298202515, 0.8733333349227905]
    """
    def __init__(self, data_file, epochs, batch_size, dropout):
        super().__init__(data_file, epochs, batch_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self._model()

    def _model(self):
        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.Conv1D(filters=32, kernel_size=4, strides=1, activation="relu")(embedded_sequences)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(8, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)
        print('##### Model Architecture Check #####')

        # int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        # embedded_sequences = self.embedding_layer(int_sequences_input)
        # x = layers.Conv1D(filters=2056, kernel_size=4, strides=1, activation="relu")(embedded_sequences)
        # x = layers.GlobalMaxPooling1D()(x)
        # x = layers.Dropout(self.dropout)(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(64, activation="relu")(x)
        # x = layers.Dropout(self.dropout)(x)
        # preds = layers.Dense(1, activation="sigmoid")(x)
        # self.model = keras.Model(int_sequences_input, preds)
        # print('##### Model Architecture Check #####')


class CNNChar(ModelFramework):
    def __init__(self, data_file, epochs, batch_size, dropout):
        super().__init__(data_file, epochs, batch_size, dropout)
        self.c_filt_size = 2
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self._model()

    def _embedding_init(self):
        tk = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                       lower=True, split=' ', char_level=True, oov_token='UNK')
        tk.fit_on_texts(self.Xtr)

        # Convert string to index
        train_sequences = tk.texts_to_sequences(self.Xtr)
        test_texts = tk.texts_to_sequences(self.Xte)

        # Padding
        train_data = pad_sequences(train_sequences, padding='pre', maxlen=256)
        test_data = pad_sequences(test_texts, padding='pre', maxlen=256)

        # Convert to numpy array
        self.Xtr = np.array(train_data, dtype='float32')
        self.Xte = np.array(test_data, dtype='float32')

        self.embedding_layer = Embedding(input_dim=len(tk.word_index)+1, output_dim=self.embedding_dim, input_length=256)
        print('##### Vocab & Embedding Layer Check #####')

    def _train_test_emb(self):
        self.ytr = tf.one_hot(self.ytr, len(self.class_names), dtype='float32').numpy()
        self.yte = tf.one_hot(self.yte, len(self.class_names), dtype='float32').numpy()
        print('##### Train Test Embedding Check #####')

    def _model(self):
        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.Conv1D(256, self.c_filt_size, strides=1, activation="relu")(embedded_sequences)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(128, self.c_filt_size, strides=1, activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.GlobalMaxPooling1D()(x)
        preds = layers.Dense(len(self.class_names), activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)
        print('##### Model Architecture Check #####')


if __name__ == '__main__':
    cnn = CNN(data_file="data/yelp_labelled.txt", epochs=50, batch_size=64, dropout=.2)
    print(cnn.model.summary())
    cnn.fit()

    # cnn_char = CNNChar(data_file="data/yelp_labelled.txt", epochs=25, batch_size=5, dropout=.3)
    # print(cnn_char.model.summary())
    # cnn_char.fit()

    print('hello worlds')