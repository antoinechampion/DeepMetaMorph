import numpy as np
import pandas as pd
import random as rd

class BatchGenerator():
    def __init__(self, data_folder_path, batch_size, validation_frac=0.05, test_frac=0.05):
        self._data_folder_path = data_folder_path
        self._input_data_path = data_folder_path + "/input.csv"
        self._output_data_path = data_folder_path + "/output.csv"
        self._dimensions_data_path = data_folder_path + "/dimensions.csv"
        self.batch_size = int(batch_size)

        dimensions = pd.read_csv(self._dimensions_data_path).astype("int32")
        self.m = dimensions["m"][0]
        self.T_x = dimensions["T_x"][0]
        self.T_y = dimensions["T_y"][0]
        self.n_x = dimensions["n_x"][0]
        self.n_y = dimensions["n_y"][0]

        self.create_dictionary()

        self._rd_seed = rd.random()
        self._validation_frac = validation_frac
        self._test_frac = test_frac

    def create_dictionary(self):
        df = pd.read_csv(self._output_data_path)
        vals = sorted(df["InstAndArgs"].unique())
        idm = np.identity(len(vals))
        self._dict = {}
        for i in range(len(vals)):
            self._dict[vals[i]] = idm[i]
        self.n_y = len(vals)

    def prepare_batch(self, df_input, df_output):
        X = next(df_input).astype("float32").to_numpy()
        X = np.reshape(X, (self.batch_size, self.T_x, self.n_x))
        Y_cat = next(df_output).to_numpy()
        Y = np.zeros((self.batch_size, self.T_y, self.n_y))
        for i in range(self.batch_size):
            for j in range(self.T_y):
                Y[i,j,:] = self._dict[Y_cat[i*j, 0]]

        X_decoder = np.zeros((self.batch_size, self.T_y, self.n_y))
        for i in range(self.batch_size):
            X_decoder[i,0,0] = 1
            for j in range(1, self.T_y):
                X_decoder[i,j,:] = Y[i,j-1,:]
        return [X, X_decoder], Y

    def generator_train(self):
        max_iter = self.m // self.batch_size
        start = 0
        end = int(max_iter*(1 - self._validation_frac - self._test_frac))

        while (True):
            input_df = pd.read_csv(self._input_data_path, chunksize = int(self.batch_size * self.T_x))
            output_df = pd.read_csv(self._output_data_path, chunksize = int(self.batch_size * self.T_y))

            for _ in range(start, end):
                yield self.prepare_batch(input_df, output_df)
    
    def generator_validation(self):
        max_iter = self.m // self.batch_size
        start = int(max_iter*(1 - self._validation_frac - self._test_frac)) + 1
        end = int(max_iter*(1 - self._test_frac))

        while (True):
            input_df = pd.read_csv(self._input_data_path, chunksize = int(self.batch_size * self.T_x))
            output_df = pd.read_csv(self._output_data_path, chunksize = int(self.batch_size * self.T_y))

            for _ in range(start, end):
                yield self.prepare_batch(input_df, output_df)

    def get_test_data(self):
        max_iter = self.m // self.batch_size
        start = int(max_iter*(1 - self._test_frac)) + 1
        end = max_iter

        X = []
        X_decoder = []
        Y = []

        input_df = pd.read_csv(self._input_data_path, chunksize = int(self.batch_size * self.T_x))
        output_df = pd.read_csv(self._output_data_path, chunksize = int(self.batch_size * self.T_y))

        for _ in range(start, end):
            [X_i, X_decoder_i], Y_i = self.prepare_batch(input_df, output_df)
            X.append(X_i)
            X_decoder.append(X_decoder_i)
            Y.append(Y_i)

        print(np.vstack(Y_i).shape)
        
        return [np.hstack(X), np.hstack(X_decoder)], np.hstack(Y)