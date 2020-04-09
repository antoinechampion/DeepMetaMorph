import numpy as np
import pandas as pd

class BatchGenerator():
    def __init__(self, data_folder_path, batch_size):
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

        self._pd_input = pd.read_csv(self._input_data_path, chunksize = int(self.batch_size * self.T_x))
        self._pd_output = pd.read_csv(self._output_data_path, chunksize = int(self.batch_size * self.T_y))

        self.validation_data = self.prepare_batch()

    def prepare_batch(self):
        X = next(self._pd_input).astype("float32").to_numpy()
        X = np.reshape(X, (self.batch_size, self.T_x, self.n_x))
        Y = next(self._pd_output).astype("float32").to_numpy()
        Y = np.reshape(Y, (self.batch_size, self.T_y, self.n_y))

        X_decoder = np.zeros((self.batch_size, self.T_y, self.n_y))
        for i in range(self.batch_size):
            X_decoder[i,0,0] = 1
            for j in range(1, self.T_y):
                X_decoder[i,j,:] = Y[i,j-1,:]
        return [X, X_decoder], Y

    def generator(self):
        max_iter = self.m // (self.T_x * self.batch_size)
        while (True):
            self._pd_input = pd.read_csv(self._input_data_path, chunksize = int(self.batch_size * self.T_x))
            self._pd_output = pd.read_csv(self._output_data_path, chunksize = int(self.batch_size * self.T_y))

            # we skip the first batch, which is used for validation
            next(self._pd_input)
            next(self._pd_output)

            for _ in range(max_iter):
                X = next(self._pd_input).astype("float32").to_numpy()
                X = np.reshape(X, (self.batch_size, self.T_x, self.n_x))
                Y = next(self._pd_output).astype("float32").to_numpy()
                Y = np.reshape(Y, (self.batch_size, self.T_y, self.n_y))

                X_decoder = np.zeros((self.batch_size, self.T_y, self.n_y))
                for i in range(self.batch_size):
                    X_decoder[i,0,0] = 1
                    for j in range(1, self.T_y):
                        X_decoder[i,j,:] = Y[i,j-1,:]
                yield self.prepare_batch()