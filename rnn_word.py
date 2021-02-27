from main import ModelFramework
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


class RNNword(ModelFramework):
    """
    Top Validation Performance [loss, accuracy]: [0.3951157033443451, 0.8600000143051147]
    """
    def __init__(self, data_file, epochs, batch_size, dropout):
        super().__init__(data_file, epochs, batch_size)
        self.epochs = epochs  # Model Training Epochs
        self.batch_size = batch_size  # Training Batch Size
        self.dropout = dropout  # Dropout Probability for dropout layers
        self.model_name = 'RNNWord'  # Model Name for saving purpose

        self._model()  # Calling Model Architecture

    def _model(self):
        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.Bidirectional(tf.keras.layers.GRU(10, return_sequences=True))(embedded_sequences)
        x = layers.Bidirectional(tf.keras.layers.GRU(5))(x)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)


if __name__ == '__main__':
    rnn_word = RNNword(data_file="data/yelp_labelled.txt", epochs=7, batch_size=32, dropout=.2)
    print(rnn_word.model.summary())
    rnn_word.fit()

    print('hello worlds')