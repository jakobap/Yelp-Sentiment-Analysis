from main import ModelFramework
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import numpy as np


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
        # int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        # embedded_sequences = self.embedding_layer(int_sequences_input)
        # x = layers.SimpleRNN(2056, activation='relu')(embedded_sequences)
        # # x = layers.LSTM(16, return_sequences=True, activation='relu')(embedded_sequences)
        # # x = layers.Bidirectional(tf.keras.layers.LSTM(8, activation='sigmoid'))(x)
        # x = layers.Dense(1024, activation='sigmoid')(x)
        # # x = layers.GlobalMaxPooling2D()(x)
        # x = layers.Dropout(self.dropout)(x)
        # preds = layers.Dense(1, activation="sigmoid")(x)
        # self.model = keras.Model(int_sequences_input, preds)

        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.Bidirectional(tf.keras.layers.GRU(10, return_sequences=True))(embedded_sequences)
        x = layers.Bidirectional(tf.keras.layers.GRU(5))(x)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)


class RNNChar(ModelFramework):
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
        x = layers.LSTM(16, return_sequences=True)(embedded_sequences)
        x = layers.LSTM(8)(x)
        x = layers.Dense(8)(x)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)


if __name__ == '__main__':
    rnn = RNN(data_file="data/yelp_labelled.txt", epochs=10, batch_size=64, dropout=.5)
    print(rnn.model.summary())
    rnn.fit()

    # cnn_char = RNNChar(data_file="data/yelp_labelled.txt", epochs=25, batch_size=10, dropout=.5)
    # print(cnn_char.model.summary())
    # cnn_char.fit()

    print('hello worlds')