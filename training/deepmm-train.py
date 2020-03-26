from DeepMMTrainer import DeepMMTrainer
from BatchGenerator import BatchGenerator

if __name__ == "__main__":   
    trainer = DeepMMTrainer(batch_size=64)
    trainer.train_model(epochs=70)