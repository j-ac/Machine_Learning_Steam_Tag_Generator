# Helper functions for data processing
import numpy as np
import random
import pandas as pb

# Conversion of training/testing set into numpy ndarray
def npFloatArray(trainSet):
    train_x_list = np.zeros(trainSet.shape[0])
    for i in trainSet:
        #  np.stack(i, axis = 0)
        train_x_list.append(np.array(i.astype(np.float32)))
        
        

    return train_x_list