from main import ModelFramework
from tensorflow import keras
from tensorflow.keras import layers


class CNNWord(ModelFramework):
    """
    Top Validation Performance [loss, accuracy]: [0.4190414249897003, 0.8733333349227905]
    """
    def __init__(self, data_file, epochs, batch_size, dropout):
        super().__init__(data_file, epochs, batch_size)
        self.epochs = epochs  # Model Training Epochs
        self.batch_size = batch_size  # Training Batch Size
        self.dropout = dropout  # Dropout Probability for dropout layers
        self.model_name = 'CNNWord'  # Model Name for saving purpose

        self._model()  # Calling Model Architecture

    def _model(self):
        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.Conv1D(filters=2056, kernel_size=4, strides=1, activation="relu")(embedded_sequences)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        preds = layers.Dense(1, activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)
        print('##### Model Architecture Check #####')


if __name__ == '__main__':
    cnn = CNNWord(data_file="data/yelp_labelled.txt", epochs=5, batch_size=16, dropout=.3)  # epo:8 batch:16 drop:.3
    print(cnn.model.summary())
    cnn.fit()

    print('hello worlds')