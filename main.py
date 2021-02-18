import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
import os

from glove import embeddings_index


class ModelFramework:
    def __init__(self, data_file, epochs, batch_size):
        self.data_file = data_file
        self.SEED = 69
        self.class_names = ['positive', 'negative']
        self.model = None
        self.embedding_dim = 300
        self._xy_extraction()
        self._train_test_split()
        self._text_vectorization()
        self._embedding_init()
        self._train_test_emb()

    def _xy_extraction(self):
        """Transforming Input Text Into np array with col0=X and col1=1"""
        with open(self.data_file, "r") as f:
            xy_tuples = [(str(line[:-4]), int(line[-2])) for line in f]
        self.X = [row[0] for row in xy_tuples]
        self.y = [row[1] for row in xy_tuples]

    def _train_test_split(self):
        """Split In Train And Test Set"""
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(self.X, self.y, test_size=0.2,
                                                                  random_state=self.SEED, shuffle=True)

        print('hahahaa')

    def _text_vectorization(self):
        self.vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=self.embedding_dim)
        text_ds = tf.data.Dataset.from_tensor_slices(self.Xtr).batch(128)
        self.vectorizer.adapt(text_ds)
        voc = self.vectorizer.get_vocabulary()
        self.word_index = dict(zip(voc, range(len(voc))))  # matching vocabulary to index

    def _embedding_init(self):
        num_tokens = len(self.word_index) + 2
        hits = 0
        misses = 0
        # Prepare embedding matrix
        self.embedding_matrix = np.zeros((num_tokens, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                self.embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))
        self.embedding_layer = Embedding(num_tokens, self.embedding_dim,
                                         embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
                                         trainable=False)


    def _train_test_emb(self):
        self.Xtr = self.vectorizer(np.array([[s] for s in self.Xtr])).numpy()
        self.Xte = self.vectorizer(np.array([[s] for s in self.Xte])).numpy()
        self.ytr = tf.one_hot(self.ytr, len(self.class_names), dtype='float32').numpy()
        self.yte = tf.one_hot(self.yte, len(self.class_names), dtype='float32').numpy()

    def fit(self):
        self.model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")
        self.model.fit(self.Xtr, self.ytr, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(self.Xte, self.yte))


if __name__ == '__main__':
    model = ModelFramework(data_file="data/yelp_labelled.txt")
    print('hello world')
