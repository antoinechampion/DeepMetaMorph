import numpy as np
import random as rd
import redis

class BatchGenerator():
    def __init__(self, batch_size, validation_frac=0.05, test_frac=0.05):
        self.redis_prefix = "deepmm_"
        self.batch_size = int(batch_size)
        self.r = redis.Redis(decode_responses=True, 
            host='localhost', 
            port=6379, db=0)

        self.redis_clean()
        self.redis_load_dimensions()
        self.redis_create_dictionary()
        self.redis_split_train_val_test(validation_frac, test_frac)

    def redis_load_dimensions(self):
        self.m = int(self.r.scard(self.redis_prefix + "available_data"))
        self.T_x = int(self.r.get(self.redis_prefix + "T_x"))
        self.T_y = int(self.r.get(self.redis_prefix + "T_y"))
        self.n_x = int(self.r.get(self.redis_prefix + "n_x"))
        self.n_y = int(self.r.get(self.redis_prefix + "n_y"))

    def encode_inst(self, inst):
        return self.dict[inst]

    def decode_inst(self, one_hot):
        try:
            return self.reverse_dict[np.argmax(one_hot)]
        except KeyError as e:
            raise KeyError("Key error: " + str(one_hot)) from e

    def redis_create_dictionary(self):
        vals = self.r.zrangebyscore(
            self.redis_prefix + "dict",
            "-inf", "+inf")
        idm = np.identity(len(vals))
        self.dict = {}
        self.reverse_dict = {}
        for i in range(len(vals)):
            self.dict[vals[i]] = idm[i]
            self.reverse_dict[i] = vals[i]
        self.n_y = len(vals)

    def redis_split_train_val_test(self, validation_frac, test_frac):
        train_frac = 1 - validation_frac - test_frac
        validation = self.r.spop(self.redis_prefix + "available_data", 
            int(self.m * validation_frac))
        self.r.sadd(self.redis_prefix + "available_validation_data", 
            *validation)
        test = self.r.spop(self.redis_prefix + "available_data", 
            int(self.m * test_frac))
        self.r.sadd(self.redis_prefix + "available_test_data", 
            *test)
        self.r.rename(self.redis_prefix + "available_data", 
            self.redis_prefix + "available_train_data")

        self.m_train = int(self.r.scard(self.redis_prefix + "available_train_data"))
        self.m_validation = int(self.r.scard(self.redis_prefix + "available_validation_data"))
        self.m_test = int(self.r.scard(self.redis_prefix + "available_test_data"))

    def redis_reset_dataset(self, dataset):
        used_key = self.redis_prefix + "used_" + dataset + "_data"
        available_key = self.redis_prefix + "available_" + dataset + "_data"
        self.r.sunionstore(available_key, available_key, used_key)
        self.r.delete(used_key)

    def redis_clean(self):
        keys = [self.redis_prefix + "available_data", 
            self.redis_prefix + "available_train_data",
            self.redis_prefix + "used_train_data",
            self.redis_prefix + "available_validation_data",
            self.redis_prefix + "used_validation_data",
            self.redis_prefix + "available_test_data",
            self.redis_prefix + "used_test_data"]
        self.r.sunionstore(self.redis_prefix + "available_data", *keys)
        for k in keys[1:]:
            self.r.delete(k)

    def prepare_batch(self, dataset):
        if (dataset not in ["train", "validation", "test"]):
            raise ValueError("`dataset` should be train, validation or test")
        else:
            redis_key_available = self.redis_prefix + "available_" + dataset + "_data"
            redis_key_used = self.redis_prefix + "used_" + dataset + "_data"
        vals = self.r.spop(redis_key_available, self.batch_size)

        XY_str = [[vv.split(";") for vv in v.split("-")] for v in vals]

        X = np.zeros((self.batch_size, self.T_x, self.n_x), dtype=np.float32)
        X_decoder = np.zeros((self.batch_size, self.T_y, self.n_y), dtype=np.float32)
        Y = np.zeros((self.batch_size, self.T_y, self.n_y), dtype=np.float32)
        for seq_idx in range(len(XY_str)):
            for x_idx in range(len(XY_str[seq_idx][0])):
                X[seq_idx,x_idx] = np.array(list(XY_str[seq_idx][0][x_idx]), dtype=np.float32)
            for y_idx in range(len(XY_str[seq_idx][1])):
                X_decoder[seq_idx,y_idx] = self.encode_inst(XY_str[seq_idx][1][y_idx])
                
        self.r.sadd(redis_key_used, *vals)
        Y = np.roll(X_decoder, -1, 1)
        one_hot_end = self.encode_inst("<END>")
        for i in range(self.batch_size):
            Y[i,-1,:] = one_hot_end
        return [X, X_decoder], Y

    def generator_train(self):
        while (True):
            for i in range(self.m_train // self.batch_size):
                yield self.prepare_batch("train")               
            self.redis_reset_dataset("train")

    def generator_validation(self):
        while (True):
            for i in range(self.m_validation // self.batch_size):
                yield self.prepare_batch("validation")                
            self.redis_reset_dataset("validation")   

    def generator_test(self):
        for i in range(self.m_test // self.batch_size):
            yield self.prepare_batch("test")    
        self.redis_reset_dataset("test")