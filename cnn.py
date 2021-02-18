from main import ModelFramework
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer

import numpy as np


class CNN(ModelFramework):
    """
    Best Validation Accuracy so far: 86.5
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
        x = layers.Conv1D(64, self.c_filt_size, strides=1, activation="relu")(embedded_sequences)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.GlobalMaxPooling1D()(x)
        preds = layers.Dense(len(self.class_names), activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)


class CNNChar(CNN):
    def __init__(self, data_file, epochs, batch_size, dropout):
        super().__init__(data_file, epochs, batch_size, dropout)
        self.c_filt_size = 2
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

        self._model()

    def _embedding_init(self):
        num_tokens = len(self.word_index) + 2
        hits = 0
        misses = 0

        tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        tk.fit_on_texts(train_texts)


        #
        # # Prepare embedding matrix
        # self.embedding_matrix = np.zeros((num_tokens, self.embedding_dim))
        # for word, i in self.word_index.items():
        #     embedding_vector = embeddings_index.get(word)
        #     if embedding_vector is not None:
        #         # Words not found in embedding index will be all-zeros.
        #         # This includes the representation for "padding" and "OOV"
        #         self.embedding_matrix[i] = embedding_vector
        #         hits += 1
        #     else:
        #         misses += 1
        # print("Converted %d words (%d misses)" % (hits, misses))
        # self.embedding_layer = Embedding(num_tokens, self.embedding_dim,
        #                                  embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
        #                                  trainable=False)


if __name__ == '__main__':
    # cnn = CNN(data_file="data/yelp_labelled.txt", epochs=25, batch_size=1, dropout=.6)
    # print(cnn.model.summary())
    # cnn.fit()

    cnn_char = CNNChar(data_file="data/yelp_labelled.txt", epochs=25, batch_size=1, dropout=.6)
    print(cnn_char.model.summary())
    cnn_char.fit()

    print('hello worlds')