import numpy as np
import random
import pandas as pb

def npFloatArray(trainSet):
    train_x_list = np.zeros(trainSet.shape[0])
    for i in trainSet:
        #  np.stack(i, axis = 0)
        train_x_list.append(np.array(i.astype(np.float32)))
        
        

    return train_x_list