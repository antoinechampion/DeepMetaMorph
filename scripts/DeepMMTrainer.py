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
from BatchGenerator import BatchGenerator

class DeepMMTrainer:
    def __init__(self, data_folder_path="data_analysis", batch_size=32, validation_frac=0.1):
        self.batch_size = batch_size
        self.validation_frac = validation_frac
        self.data = BatchGenerator(data_folder_path, self.batch_size, self.validation_frac)
        self.model_compiled = False

    def create_model(self, batch_size=32, lstm_hidden_size=256, print_summary=True):
        lstm_size = lstm_hidden_size
        print("Building encoder-decoder")
        encoder_inputs = K.Input(shape=(None, self.data.n_x))
        mask_inputs = K.layers.Masking(-1)(encoder_inputs)
        if False and tf.test.is_built_with_cuda():
            encoder_lstm = K.layers.CuDNNLSTM(lstm_size, return_state=True)
        else:
            encoder_lstm = K.layers.LSTM(lstm_size, return_state=True)
        _, a, c = encoder_lstm(mask_inputs)
        encoder_lstm_outputs = [a, c]

        decoder_inputs = K.Input(shape=(None, self.data.n_y))
        if False and tf.test.is_built_with_cuda():
            decoder_lstm = K.layers.CuDNNLSTM(lstm_size, return_sequences=True, return_state=True)
        else:
            decoder_lstm = K.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
        decoder_dense = K.layers.Dense(self.data.n_y, activation="softmax")
        decoder_lstm_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_lstm_outputs)
        decoder_outputs = decoder_dense(decoder_lstm_outputs)

        self._model = K.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self._model.compile(optimizer="adam", 
                                loss = "categorical_crossentropy",
                            metrics=['accuracy'])   
        if print_summary:
            self._model.summary()   

        print("Building inference model")
        print("--> Encoder")

        self._encoder_model = K.Model(encoder_inputs, encoder_lstm_outputs)
        if print_summary:
            self._encoder_model.summary()

        print("--> Decoder")

        decoder_input_a = K.Input(shape=(lstm_size,))
        decoder_input_c = K.Input(shape=(lstm_size,))
        decoder_lstm_inputs = [decoder_input_a, decoder_input_c]
        decoder_outputs, a, c = decoder_lstm(decoder_inputs, initial_state = decoder_lstm_inputs)
        decoder_lstm_outputs = [a, c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self._decoder_model = K.Model([decoder_inputs] + decoder_lstm_inputs,
                                        [decoder_outputs] + decoder_lstm_outputs)

        if print_summary:
            self._decoder_model.summary()

        print(">>>>> DONE <<<<<")
        self.model_compiled = True

    def train_model(self, epochs=30, log_tensorboard = True, save_models = True):
        if not self.model_compiled:
            self.create_model(print_summary=False)

        if (log_tensorboard):
            callbacks = [K.callbacks.TensorBoard(log_dir='tf_logs')]
        else:
            callbacks = []

        print("Training the model...")
        steps_train = ((1-self.validation_frac)*self.data.m) // (self.data.batch_size * self.data.T_x) - 1
        steps_val = (self.validation_frac*self.data.m) // (self.data.batch_size * self.data.T_x) - 1
        self._model.fit_generator(self.data.generator_train(),
                steps_per_epoch = steps_train,
                epochs=epochs, 
                validation_data=self.data.generator_validation(),
                validation_steps=steps_val,
                callbacks = callbacks)

        print("Training complete. Saving...")

        if (save_models):
            self._model.save("models/deepmm-model.h5")
            self._encoder_model.save("models/deepmm-inference-encoder.h5")
            self._decoder_model.save("models/deepmm-inference-decoder.h5")
            pd.DataFrame(self.data._dict).to_csv("models/dict.csv")

        print(">>>>> DONE <<<<<")
