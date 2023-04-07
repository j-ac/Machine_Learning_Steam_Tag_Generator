# Helper functions for data processing
import numpy as np
from sklearn.model_selection import train_test_split

# Conversion of training/testing set into numpy ndarray
def npFloatArray(trainSet):

    train_x_list = np.zeros((1, ))
    for i in trainSet:
        #  np.stack(i, axis = 0)
        train_x_list = np.append(trainSet, np.array(i.astype(np.float32)))

    return train_x_list

# Load Function for loading dataset to model
def load_data(dataset, train_split):
    # Initialize x and y datasets using tags and desc data respecetively
    dataset_X = dataset.iloc[:, 1:].values
    dataset_y = dataset.iloc[:,:1].values

    # Compute max index within each dataset
    X_max_index = 0
    y_max_index = 0

    for array in dataset_X:
            
        npArray = np.fromstring(array[0][1:-1], 
                                dtype=int, 
                                sep=',')

        if(npArray.size != 0):
            for value in np.nditer(npArray):
                if int(value) > X_max_index: X_max_index = int(value)

    for array in dataset_y:
            
        npArray = np.fromstring(array[0][1:-1], 
                                dtype=int, 
                                sep=',')

        if(npArray.size != 0):
            for value in np.nditer(npArray):
                if int(value) > y_max_index: 
                    y_max_index = int(value)

    # Print Max index for each dataset to check
    print("Max Y Index: " + str(y_max_index))
    print("Max X Index: " + str(X_max_index))

    # 2D arrays
    all_X = np.zeros([dataset_X.shape[0], X_max_index + 1], dtype = int)
    all_y = np.zeros([dataset_y.shape[0], y_max_index + 1], dtype = int)

    # Replace 0's with 1's in corresponding indexes
    for i in range(0, dataset_X.shape[0]):
        npArray = np.fromstring(dataset_X[i][0][1:-1], 
                                dtype=int, 
                                sep=',')

        if(npArray.size != 0):
            for value in np.nditer(npArray):
                all_X[i][int(value)] = 1

    for i in range(0, dataset_y.shape[0]):
        npArray = np.fromstring(dataset_y[i][0][1:-1], 
                                dtype=int, 
                                sep=',')

        if(npArray.size != 0):
            for value in np.nditer(npArray):
                all_y[i][int(value)] = 1

    X_train, X_test, y_train, y_test = train_test_split(all_X, 
                                                        all_y, 
                                                        test_size = train_split, 
                                                        random_state=0)
    
    return X_train, X_test, y_train, y_test

# Print function for summary
def myPrint(s):
    with open('Training_Summary.txt', 'a') as f:
        print(s, file = f)
