from DeepMMTrainer import DeepMMTrainer
from BatchGenerator import BatchGenerator

if __name__ == "__main__":   
    trainer = DeepMMTrainer()
    trainer.train_model(save_models = True, epochs=64)