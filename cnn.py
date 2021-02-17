from main import ModelFramework
from tensorflow import keras
from tensorflow.keras import layers


class CNN(ModelFramework):
    def __init__(self, data_file, epochs, batch_size):
        super().__init__(data_file, epochs, batch_size)
        self.c_filt_size = 2

        self._model()

    def _model(self):
        int_sequences_input = keras.Input(shape=(None,), dtype='int64')
        embedded_sequences = self.embedding_layer(int_sequences_input)
        x = layers.Conv1D(128, self.c_filt_size, activation="relu")(embedded_sequences)
        x = layers.Dropout(0.5)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, self.c_filt_size, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        preds = layers.Dense(len(self.class_names), activation="sigmoid")(x)
        self.model = keras.Model(int_sequences_input, preds)


if __name__ == '__main__':
    cnn = CNN(data_file="data/yelp_labelled.txt", epochs=15, batch_size=8)
    print(cnn.model.summary())
    cnn.fit()


    print('hello worlds')