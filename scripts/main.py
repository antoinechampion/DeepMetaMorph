from DeepMMTrainer import DeepMMTrainer

if __name__ == "__main__":   
    trainer = DeepMMTrainer()
    trainer.train_model(save_models = False, epochs=3)