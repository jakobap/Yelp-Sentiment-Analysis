import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split

from glove import pretrained_embedd


class ModelFramework:
    def __init__(self, data_file, epochs, batch_size):
        self.data_file = data_file  # Raw Text Data to classify
        self.SEED = 1312  # Random Seed
        self.class_names = ['positive', 'negative']  # Class names to classify for
        self.model = None  # Model Variable
        self.embedding_dim = 300  # Target dimension of word vector embedding

        self._xy_extraction()  # Calling data extraction
        self._train_test_split()  # Calling train test split
        self._text_vectorization()  # Calling Text vectorization
        self._embedding_init()  # Calling embedding initialization
        self._train_test_emb()  # Calling Embedding application on train and test set

    def _xy_extraction(self):
        """
        Transforming Input Text Into np array with col0=X and col1=1
        """
        with open(self.data_file, "r") as f:
            xy_tuples = [(str(line[:-4]), int(line[-2])) for line in f]
        self.X = [row[0] for row in xy_tuples]
        self.y = [row[1] for row in xy_tuples]

    def _train_test_split(self):
        """
        Split In Train And Test Set & Check for balanced Classes
        """
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(self.X, self.y, test_size=0.15,
                                                                  random_state=self.SEED, shuffle=True)
        print(f'Train Set:{np.mean(self.ytr)}')
        print(f'Train Set:{np.mean(self.yte)}')
        print('##### Train Test Split Check #####')

    def _text_vectorization(self):
        """
        Function Creates the Vocabulary we need to represent with our embedding.
        1st Creation of vocabulary from training data including standardization of text.
        2nd Fitting the Vocabulary into a dictionary with its respective index
        """
        self.vectorizer = TextVectorization(max_tokens=2000, output_mode='int',
                                            output_sequence_length=max([len(i) for i in self.Xtr]))
        text_ds = tf.data.Dataset.from_tensor_slices(self.Xtr).batch(128)
        self.vectorizer.adapt(text_ds)
        voc = self.vectorizer.get_vocabulary()
        self.word_index = dict(zip(voc, range(len(voc))))  # matching vocabulary to index in dict
        print('##### Vocabulary Check #####')

    def _embedding_init(self):
        """
        Creating embedding matrix that maps vocabulary to pre-trained matrix
        1st Create 0 Matrix Dummy
        2nd Iterate all words in vocab and index pre trained representation
        3rd Add to matrix if in pre trained embedding and skip if not
        4th Create Standard Embedding Layer to include in keras model
        """
        num_tokens = len(self.word_index) + 2
        hits = 0
        misses = 0

        # self.embedding_matrix = np.zeros((len(pretrained_embedd)+2, self.embedding_dim))  # set up zero matrix
        # print('#####')
        # # count = 0
        #
        # for count, word in enumerate(pretrained_embedd):
        #     self.embedding_matrix[count] = pretrained_embedd[word]
        #
        # self.embedding_layer = Embedding(input_dim=len(pretrained_embedd)+2, output_dim=self.embedding_dim,
        #                                  embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
        #                                  trainable=False)
        # print('##### Embedding Layer Check #####')

        self.embedding_matrix = np.zeros((num_tokens, self.embedding_dim))  # set up zero matrix
        for word, i in self.word_index.items():  # iterate all words in vocabulary
            embedding_vector = pretrained_embedd.get(word)  # import glove pre-trained vectorization
            if embedding_vector is not None:  # if word is found in pre-trained embeddings
                self.embedding_matrix[i] = embedding_vector  # add to embedding matrix
                hits += 1
            else:  # skip if not found
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))
        self.embedding_layer = Embedding(input_dim=num_tokens, output_dim=self.embedding_dim,
                                         embeddings_initializer=keras.initializers.Constant(self.embedding_matrix),
                                         trainable=False)
        print('##### Embedding Layer Check #####')

    def _train_test_emb(self):
        """
        Bringing Input Text In the appropriate vector format
        1st Vectorizer assignes vocabulary index to words
        2nd bring labels into one hot or binary vector
        """
        self.Xtr = self.vectorizer(np.array([[s] for s in self.Xtr])).numpy()
        self.Xte = self.vectorizer(np.array([[s] for s in self.Xte])).numpy()

        ### Binary Encoding for final layer with one node
        self.ytr = np.array(self.ytr).astype('int64')
        self.yte = np.array(self.yte).astype('int64')

        ### One-Hot Encoding for final layer with 2 cells
        # self.ytr = tf.one_hot(self.ytr, len(self.class_names), dtype='float32').numpy()
        # self.yte = tf.one_hot(self.yte, len(self.class_names), dtype='float32').numpy()
        print('##### Train Test Embedding Check #####')

    def fit(self):
        """
        Compile, fit and evaluate model.
        1st Compile model with optimization parameters
        2nd Fit model by passing data and training parameters
        3rd Validate model with test set and compute model performance
        """
        self.model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")
        self.model.fit(self.Xtr, self.ytr, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(self.Xte, self.yte))
        score = self.model.evaluate(self.Xte, self.yte, batch_size=self.batch_size)
        print(self.model.metrics_names)
        print(f'Score: {score}')
        self.model.save(f'./models/{self.model_name}acc_{score[1]}')


if __name__ == '__main__':
    model = ModelFramework(data_file="data/yelp_labelled.txt")
    print('hello world')
