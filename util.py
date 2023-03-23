import numpy as np
import random
import pandas as pb


# def load_data(train_set_percentage):
#     dataset = pb.read_csv('min_75_max5k_as_ids.csv', names = ["tags", "desc"])
#     # x_train = pb.read_csv('min_75_max5k_as_ids.csv')
#     # x_train = x_train.to_numpy()
#     # x_train = np.float32(x_train)

#     # load features and labels based on those present in the CSV files

#     ########## TEST VARIABLES

    

#     #########################

#     features = dataset.copy()

#     # convert features into numpy array
#     features = np.array(features)

#     # shape and extract total num
#     totalNum = np.shape(np.array(dataset["desc"]))[0]
#     trainNum = int(totalNum*train_set_percentage)
#     testNum = totalNum - trainNum

#     descs = dataset["desc"][:]
#     labels = dataset["tags"][:]

#     trainSetX = []
#     trainSetY = []

#     testSetX = []
#     testSetY = []

#     training_samples = []
#     random.seed(0)

#     for i in range(trainNum):
#             found = False
#             while not found:
#                 sample_index = random.randint(1, totalNum - 1)
#                 if sample_index not in training_samples:
#                     training_samples.append(sample_index)
#                     found = True
#                     if type(descs[sample_index]) == str and type(labels[sample_index]) == str:
#                         tmp = list(descs[sample_index].split(", "))
#                         trainSetX.append(np.array(tmp, float))
#                         trainSetY.append(np.array(labels[sample_index].split(", ")))
        
#     test_index = 0
#     for i in range(totalNum):
#         if i not in training_samples:
#             if type(descs[i]) == str and type(labels[i]) == str:
#                 testSetX.append(np.array(list(descs[i].split(", "))))
#                 testSetY.append(np.array(list(labels[i].split(", "))))
#                 test_index += 1


#     # print(descs.shape)
#     # print(trainSetX.shape)
#     # print(trainSetY.shape)
#     # print(labels.shape)
#     # print(np.shape(np.array(dataset["desc"][:])))


#     # print(np.array(list(descs[2].split(", "))))
#     # print(list(descs[1].split(", ")))
    
#     return trainSetX, trainSetY, testSetX, testSetY


def npFloatArray(trainSet):
    train_x_list = []

    for i in trainSet:
        #  np.stack(i, axis = 0)
        train_x_list.append(i.astype(np.float32))
        trainSet_NP = np.array(train_x_list)

    return trainSet_NP


