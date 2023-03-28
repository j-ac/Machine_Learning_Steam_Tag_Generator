import numpy as np
import pandas as pd
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

# Dataset
dataset = pd.read_csv("min75_max5k_as_idsNOEMPTIES.csv")

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

# Checking Outputs
# for i in range(0, 10):
#     print(all_X[i])

# for i in range(0, 10):
#     print(all_y[i])

# Multilabel Classification Loop
all_tests= []
all_results = []

for i in range(0, all_y.shape[1]):

    # Isolating Single Label
    all_y_single_label = []

    for j in range(0, all_y.shape[0]):
        all_y_single_label.append(all_y[j][i])

    # Splitting Dataset
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y_single_label, test_size=0.2, random_state=0)

    # Training
    model = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
    model.fit(X_train, y_train)

    # Prediction
    y_prediction = list(model.predict(X_test))

    # Single Label Evaluation
    model_accuracy = accuracy_score(y_test, y_prediction)
    model_precision = precision_score(y_test, y_prediction)
    model_recall = recall_score(y_test, y_prediction)
    model_f1 = f1_score(y_test, y_prediction)

    all_tests.append(y_test)
    all_results.append(y_prediction)

    print("Results: ", {
        "Accuracy": model_accuracy,
        "Precision": model_precision,
        "Recall": model_recall,
        "F1 Score": model_f1
    })

# Overall Evaluation
print("Hamming Loss: ", hamming_loss(all_tests, all_results))

print(len(all_tests), len(all_tests[0]))
print(len(all_results), len(all_results[0]))

true_tags = [[0] * len(all_tests) for i in range(len(all_tests[0]))]
predicted_tags = [[0] * len(all_results) for i in range(len(all_results[0]))]

print(len(true_tags), len(true_tags[0]))
print(len(predicted_tags), len(predicted_tags[0]))

for i in range(len(all_tests)):
    for j in range(len(all_tests[0])):
        true_tags[j][i] = all_tests[i][j]
        predicted_tags[j][i] = all_results[i][j]

# Output Results to File
# with open('true_tags.csv', mode='w', newline='') as open_file:
#     file_writer = csv.writer(open_file)

#     for i in range(len(true_tags)):
#         file_writer.writerow(true_tags[i])

# with open('predicted_tags.csv', mode='w', newline='') as open_file:
#     file_writer = csv.writer(open_file)

#     for i in range(len(predicted_tags)):
#             file_writer.writerow(predicted_tags[i])

pd.DataFrame(true_tags).to_csv('true_tags.csv', index=False, header=False)
pd.DataFrame(predicted_tags).to_csv('predicted_tags.csv', index=False, header=False)

# pd.DataFrame([all_tests, all_results]).to_csv("results.csv", index=False)