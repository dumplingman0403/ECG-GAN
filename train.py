import numpy as np
import pickle

from numpy.lib.npyio import save
from GAN_model import *


if __name__ == "__main__":
    
    
    X_train = pickle.load(open("X_train_aa.pkl", "rb"))

    EPOCHS = 10000
    LATENT_SIZE = 100
    SAVE_INTRIVAL = 100
    SAVE_MODEL_INTERVAL = 1000
    BATCH_SIZE = 128
    INPUT_SHAPE = (216, 1)  # --> AA dataset
    #INPUT_SHAPE = (180, 1)  # --> AF dataset
    RANDOM_SINE = False
    SCALE = 2 
    MINIBATCH = True # use minibatch discrimination to avoid mode collapse
    SAVE_MODEL = True
    SAVE_REPORT = True
    GEN_VERSION = 1  # use generator from in_progress
    dcgan = DCGAN(INPUT_SHAPE, LATENT_SIZE, random_sine=RANDOM_SINE, scale=SCALE, minibatch=MINIBATCH, gen_version=GEN_VERSION) 
    X_train = dcgan.specify_range(X_train, -2, 2)/2 # limit the signal range [-2, 2], scale by divid 2 
    X_train = X_train.reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    dcgan.train(EPOCHS, X_train, BATCH_SIZE, SAVE_INTRIVAL, save=SAVE_MODEL, save_model_interval=SAVE_MODEL_INTERVAL, 
                save_report=SAVE_REPORT)
    print("Complete!!!")
    
