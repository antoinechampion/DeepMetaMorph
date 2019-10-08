from __future__ import print_function
import keras as K
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
import numpy as np
import random
import sys
import io
import pandas as pd

class DeepMetaMorphTrainer:
    def load_data(self, input_data_path, output_data_path):
        X = pd.read_csv(input_data_path).astype("int32")
        Y = pd.read_csv(output_data_path).astype("int32")

        self.m = X["Sequence"].max()
        self.T_x = X["InstructionIndex"].max()
        self.n = 736

        X = X.pop("StateDiff")
        Y = to_categorical(Y.pop("OneHotInstruction"))

        self.T_y = 736
        
        X = tf.reshape(X, [self.m, self.T_x, self.n])

        self.__dataset = tf.data.Dataset.from_tensor_slices((X.values, Y.values))

    def build_model(self):
        # CuDNNLSTM
        self.__model = K.Sequential([
            K.Input(shape=(self.T_x, self.n)),
            K.LSTM(256),
            K.Dense(self.T_y, activation='softmax')
        ])

        self.__model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    def train(self):        
        model.summary()
        self.__model.fit(self.__dataset,
              batch_size=512,
              epochs=60)

if __name__ == '__main__':
    trainer = DeepMetaMorphTrainer()
    trainer.load_data("data/input.csv", "data/output.csv")
    trainer.build_model()
    trainer.train()