from __future__ import print_function
import keras as K
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
import numpy as np
import random
import sys
import io
import pandas as pd
import tensorflow as tf

class DeepMetaMorphTrainer:
    def load_data(self, input_data_path, output_data_path):
        X = pd.read_csv(input_data_path).astype("int32")
        Y = pd.read_csv(output_data_path).astype("int32")

        self.m = X["Sequence"].max()
        self.T_x = X["InstructionIndex"].max()
        self.n_x = 736
        self.T_y = Y["InstructionIndex"].max()
        # We add 1 for the sequence beginning token
        self.n_y = Y["OneHotInstruction"].max() + 1

        X = X.pop("StateDiff")
        X = tf.reshape(X, [self.m, self.T_x, self.n_x])

        Y = to_categorical(Y.pop("OneHotInstruction")+1)
        Y = tf.reshape(Y, [self.m, self.T_x, self.n_y])

        X_decoder = np.zeros((self.m, self.T_y, self.n_y), dtype = "int32")
        for i in range(self.m):
            X_decoder[i,0,:] = np.zeros((self.n_y))
            X_decoder[i,0,0] = 1
            for j in range(1, self.T_y):
                X_decoder[i,j,:] = Y[i,j-1,:]

        self.__X_encoder = X
        self.__Y = Y
        self.__X_decoder = X_decoder

        #self.__dataset = tf.data.Dataset.from_tensor_slices((X.values, Y.values))

    def build_model(self, lstm_size = 256, batch_size = 512, epochs = 60):
        # CuDNNLSTM
        print("Building encoder-decoder")
        inputs = K.Inputs(shape=(None, self.n_x))
        encoder_inputs = K.Mask(-1)(inputs)
        if tf.test.is_built_with_cuda():
            encoder_lstm = K.CuDNNLSTM(lstm_size, return_state=True)
        else:
            encoder_lstm = K.LSTM(lstm_size, return_state=True)
        _, a, c = encoder_lstm(encoder_inputs)
        encoder_lstm_outputs = [a, c]

        decoder_inputs = K.Input(shape=(None, self.n_y))
        if tf.test.is_built_with_cuda():
            decoder_lstm = K.CuDNNLSTM(lstm_size, return_sequences=True, return_state=True)
        else:
            decoder_lstm = K.LSTM(lstm_size, return_sequences=True, return_state=True)
        decoder_dense = K.Dense(self.n_y, activation="softmax")
        decoder_lstm_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_lstm_outputs)
        decoder_outputs = decoder_dense(decoder_lstm_outputs)

        self.__model = K.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.__model.compile(optimizer="adam", 
                             loss = "categorical_crossentropy",
                            metrics=['accuracy'])   
        self.__model.summary()   

        print("Building inference model")
        print("--> Encoder")

        self.__encoder_model = K.Model(encoder_inputs, encoder_lstm_outputs)
        self.__encoder_model.summary()

        print("--> Decoder")

        decoder_input_a = Input(shape=(lstm_size,))
        decoder_input_c = Input(shape=(lstm_size,))
        decoder_lstm_inputs = [decoder_input_a, decoder_input_c]
        decoder_outputs, a, c = decoder_lstm(decoder_inputs, initial_state = decoder_lstm_inputs)
        decoder_lstm_outputs = [a, c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.__decoder_model = Model([decoder_inputs] + decoder_lstm_inputs,
                                     [decoder_outputs] + decoder_lstm_outputs)
        self.__decoder_model.summary()

        print(">>>>> DONE <<<<<")
        
    def train(self):          
        self.__model.fit([self.__X, self.__X_decoder], self.__Y,
              batch_size=batch_size,
              epochs=epochs, 
              validation_split=0.1)
        print("Training complete. Saving...")
        self.__model.save("models/deepmm-model.h5")
        self.__encoder_model.save("models/deepmm-inference-encoder.h5")
        self.__decoder_model.save("models/deepmm-inference-decoder.h5")
        print(">>>>> DONE <<<<<")



if __name__ == '__main__':
    trainer = DeepMetaMorphTrainer()
    trainer.load_data("data/input.csv", "data/output.csv")
    trainer.build_model()
    trainer.train()