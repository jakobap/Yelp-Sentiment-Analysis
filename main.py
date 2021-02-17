import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


class ModelFramework:
    def __init__(self, data_file):
        self.data_file = data_file
        self.X = None
        self.y = None
        self.Xtr = None
        self.ytr = None
        self.Xte = None
        self.yte = None
        self.SEED = 69

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

    def pre_processing(self):
        """Run Preprocessing"""
        self._xy_extraction()
        self._train_test_split()


if __name__ == '__main__':
    model = ModelFramework(data_file="yelp_labelled.txt")
    model.pre_processing()
    print('hello world')
