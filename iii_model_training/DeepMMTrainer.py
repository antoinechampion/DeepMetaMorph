from __future__ import print_function
import tensorflow.keras as K
from tensorflow.keras import backend as KB
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import random
import sys
import io
import tensorflow as tf
from BatchGenerator import BatchGenerator

class DeepMMTrainer:
    def __init__(self, batch_size=32, validation_frac=0.05, test_frac=0.05):
        self.batch_size = batch_size
        self.validation_frac = validation_frac
        self.test_frac = test_frac
        self.data = BatchGenerator(self.batch_size, self.validation_frac, self.test_frac)
        self.model_compiled = False
        self._model = None
        self._encoder_model = None
        self._decoder_model = None

        # Bugfix in TF 2.0 CUDA memory allocation
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
        self._model.compile(optimizer = K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
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

    def infer(self, X):
        if X.shape[0] > 1:
            return self.infer_many(X)
        if self._encoder_model is None or self._decoder_model is None:
            self._encoder_model = K.models.load_model("models/deepmm-inference-encoder.h5", compile=False)
            self._decoder_model = K.models.load_model("models/deepmm-inference-decoder.h5", compile=False)
        states_value = self._encoder_model.predict(X)
        target_seq = np.zeros((1, 1, self.data.n_y))
        target_seq[0, 0] = self.data.encode_inst("<GO>")
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            y_pred, a, c = self._decoder_model.predict(
                [target_seq] + states_value)

            y_pred_clip = np.zeros((self.data.n_y,))
            categorical_pred = np.argmax(y_pred[0, 0, :])
            y_pred_clip[categorical_pred] = 1

            inst = self.data.decode_inst(y_pred_clip)
            decoded_sentence.append(inst)

            if (inst == "<END>" or
            len(decoded_sentence) > self.data.T_y):
                stop_condition = True

            target_seq = np.zeros((1, 1, self.data.n_y))
            target_seq[0, 0] = y_pred_clip

            states_value = [a, c]

        return decoded_sentence

    def infer_many(self, X):
        all_seq = []
        for seq_idx in range(X.shape[0]):
            all_seq.append(self.infer(X[seq_idx:seq_idx+1,:,:]))
        return all_seq
    
    def print_metrics(self):
        end_token = np.argmax(self.data.encode_inst("<END>"))
        total_correct = 0
        total_processed = 0
        print("Evaluating performance on test data...")
        for X_test, Y_test in self.data.generator_test():
            Y_pred = self.infer_many(X_test[0])
            for i in range(self.data.batch_size):
                amax_preds = np.zeros(self.data.T_y,)
                Y_pred[i] = Y_pred[i] + ["<END>" for _ in range(self.data.T_y-len(Y_pred[i]))]
                for j in range(self.data.T_y):
                    amax_preds[j] = np.argmax(self.data.encode_inst(Y_pred[i][j]))
                
                amax_correct = np.argmax(Y_test[i], axis=1).flatten()
                seq_end = max(np.where(amax_preds==end_token)[0][0],
                                np.where(amax_correct==end_token)[0][0])
                correct_preds = amax_preds[:seq_end+1] == amax_correct[:seq_end+1]
                total_correct += correct_preds.sum()
                total_processed += np.size(correct_preds)

        print("Model accuracy on test data: %2f" % (total_correct/total_processed))

    def test_model(self):
        self.create_model(print_summary=False)
        self._model.load_weights("models/deepmm-model.h5")
        self._encoder_model.load_weights("models/deepmm-inference-encoder.h5")
        self._decoder_model.load_weights("models/deepmm-inference-decoder.h5")
        self.print_metrics()

    def train_model(self, epochs=30, log_tensorboard = True, save_models = True, load_existing = False):
        if load_existing:
            self.create_model(print_summary=False)
            self._model.load_weights("models/deepmm-model.h5")
            self._encoder_model.load_weights("models/deepmm-inference-encoder.h5")
            self._decoder_model.load_weights("models/deepmm-inference-decoder.h5")
        elif not self.model_compiled:
            self.create_model(print_summary=False)

        if (log_tensorboard):
            callbacks = [K.callbacks.TensorBoard(log_dir='tf_logs')]
        else:
            callbacks = []

        print("Training the model...")
        steps_train = self.data.m_train // self.batch_size
        steps_val = self.data.m_validation // self.batch_size
        self._model.fit(x = self.data.generator_train(),
                steps_per_epoch = steps_train,
                epochs = epochs, 
                validation_data = self.data.generator_validation(),
                validation_steps = steps_val,
                callbacks = callbacks)

        print("Training complete. Saving...")

        if (save_models):
            self._model.save("models/deepmm-model.h5")
            self._encoder_model.save("models/deepmm-inference-encoder.h5")
            self._decoder_model.save("models/deepmm-inference-decoder.h5")

        print("Statistics")
        self.print_metrics()

        print(">>>>> DONE <<<<<")
