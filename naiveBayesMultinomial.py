import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dataset
dataset = pd.read_csv("min_75_max5k_as_ids.csv")

# Preprocessing
dataset_X = dataset.iloc[:, 1:].values
print(dataset_X.shape)
dataset_y = dataset.iloc[:,:1].values
print(dataset_y.shape)

# Calculate max index for X (words) and y (labels)
X_max_index = 0
y_max_index = 0

for array in dataset_X:
        
    npArray = np.fromstring(array[0][1:-1], dtype=int, sep=',')

    if(npArray.size != 0):
        for value in np.nditer(npArray):
            if int(value) > X_max_index: X_max_index = int(value)

print("X max index:", X_max_index)

for array in dataset_y:
        
    npArray = np.fromstring(array[0][1:-1], dtype=int, sep=',')

    if(npArray.size != 0):
        for value in np.nditer(npArray):
            if int(value) > y_max_index: y_max_index = int(value)

print("y max index:", y_max_index)

# Create Empty 2D arrays with dimensions of total samples vs max X and max Y

all_X = np.zeros([dataset_X.shape[0], X_max_index + 1], dtype=int)
print(all_X.shape)
all_y = np.zeros([dataset_y.shape[0], y_max_index + 1], dtype=int)
print(all_y.shape)

# Replace 0's with 1's in corresponding indexes
# TODO, Fix: IndexError: index 3276 is out of bounds for axis 0 with size 3276

for i in range(0, dataset_X.shape[0]):
    npArray = np.fromstring(dataset_X[i][0][1:-1], dtype=int, sep=',')

    if(npArray.size != 0):
        for value in np.nditer(npArray):
            all_X[i][int(value)] = 1

for i in range(0, dataset_y.shape[0]):
    npArray = np.fromstring(dataset_y[i][0][1:-1], dtype=int, sep=',')

    if(npArray.size != 0):
        for value in np.nditer(npArray):
            all_y[i][int(value)] = 1

# Output to csv to check

pd.DataFrame(all_X).to_csv("allX.csv", header=None, index=None)
pd.DataFrame(all_y).to_csv("allY.csv", header=None, index=None)

# # Dataset
# emails = pd.read_csv('emails.csv')
# emails.drop(columns="Email No.", inplace=True)

# X = emails.iloc[:, :-1].values
# y = emails.iloc[:, -1].values

# # Preprocessing: Reducing counts more than 1, to 1
# normalized_X = np.where(X > 1, 1, X)
# pd.DataFrame(normalized_X).to_csv("emailsMNBPreProcessing.csv", header=None, index=None)

# X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.2, random_state=0)

# # Training
# mnb = MultinomialNB()
# mnb.fit(X_train, y_train)
# MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

# # Prediction
# y_prediction = mnb.predict(X_test)

# # Evaluation
# mnb_accuracy = accuracy_score(y_test, y_prediction)
# mnb_precision = precision_score(y_test, y_prediction)
# mnb_recall = recall_score(y_test, y_prediction)
# mnb_f1 = f1_score(y_test, y_prediction)

# results = np.asarray([y_test, y_prediction])
# pd.DataFrame(results).to_csv("emailsMNBresults.csv", header=None, index=None)

# print("Results: ", {
#     "Accuracy": mnb_accuracy,
#     "Precision": mnb_precision,
#     "Recall": mnb_recall,
#     "F1 Score": mnb_f1
# })