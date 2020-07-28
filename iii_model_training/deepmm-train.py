from DeepMMTrainer import DeepMMTrainer
from BatchGenerator import BatchGenerator

import numpy as np

if __name__ == "__main__":   
    trainer = DeepMMTrainer(batch_size=512)
    trainer.train_model(epochs=10)