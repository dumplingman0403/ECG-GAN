import numpy as np
import pickle
from GAN_model import *


if __name__ == "__main__":
    
    
    X_train = pickle.load(open("X_train_aa.pkl", "rb"))

    EPOCHS = 1000
    LATENT_SIZE = 100
    SAVE_INTRIVAL = 100
    BATCH_SIZE = 128
    INPUT_SHAPE = (216, 1)
    RANDOM_SINE = False
    SCALE = 2

    dcgan = DCGAN(INPUT_SHAPE, LATENT_SIZE, RANDOM_SINE, scale=SCALE) 
    X_train = dcgan.specify_range(X_train, -2, 2)/2
    X_train = X_train.reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    dcgan.train(EPOCHS, X_train, BATCH_SIZE, SAVE_INTRIVAL)
    print("Complete!!!")
    pass
    
