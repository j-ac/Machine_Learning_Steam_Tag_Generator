import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

# Constants
TRAINING_INPUT_FILE = "min75_max5k_as_ids.csv"

TRUE_TAGS_OUTPUT_FILE = "true_tags.csv"
PREDICTED_TAGS_OUTPUT_FILE = "predicted_tags.csv"
RANDOM_GAME_OUTPUT_FILE = "random_game_prediction.csv"

TRAINING_SPLIT = 0.2
ALPHA = 0.01

# Functions
def calculateMaxXYIndex(dataset_X, dataset_y):
    
    X_max_index = 0
    y_max_index = 0

    for array in dataset_X:
            
        npArray = np.fromstring(array[0][1:-1], dtype=int, sep=',')

        if(npArray.size != 0):
            for value in np.nditer(npArray):
                if int(value) > X_max_index: X_max_index = int(value)

    for array in dataset_y:
            
        npArray = np.fromstring(array[0][1:-1], dtype=int, sep=',')

        if(npArray.size != 0):
            for value in np.nditer(npArray):
                if int(value) > y_max_index: y_max_index = int(value)

    return X_max_index, y_max_index

def preprocessDataset(dataset):

    dataset_X = dataset.iloc[:, 1:].values
    dataset_y = dataset.iloc[:,:1].values

    X_max_index, y_max_index = calculateMaxXYIndex(dataset_X, dataset_y)

    # Create Empty 2D arrays with dimensions of total samples vs max X and max Y
    all_X = np.zeros([dataset_X.shape[0], X_max_index + 1], dtype=int)
    all_y = np.zeros([dataset_y.shape[0], y_max_index + 1], dtype=int)

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

    return all_X, all_y

def runNaiveBayes(all_X, all_y):

    all_tests= []
    all_results = []

    random_game_index = np.random.randint(0, len(all_X))
    random_game_description = all_X[random_game_index]
    random_game_results = []

    for i in range(0, all_y.shape[1]):

        # Isolating Single Label
        all_y_single_label = []

        for j in range(0, all_y.shape[0]):
            all_y_single_label.append(all_y[j][i])

        # Splitting Dataset
        X_train, X_test, y_train, y_test = train_test_split(all_X, all_y_single_label, test_size=TRAINING_SPLIT, random_state=0)

        # Training
        model = MultinomialNB(alpha=ALPHA, class_prior=None, fit_prior=True)
        model.fit(X_train, y_train)

        # Prediction
        y_prediction = list(model.predict(X_test))
        random_game_prediction = list(model.predict([random_game_description]))

        # Single Label Evaluation
        model_accuracy = accuracy_score(y_test, y_prediction)
        model_precision = precision_score(y_test, y_prediction)
        model_recall = recall_score(y_test, y_prediction)
        model_f1 = f1_score(y_test, y_prediction)

        all_tests.append(y_test)
        all_results.append(y_prediction)
        random_game_results.append(random_game_prediction)

        print("Results for Label " + str(i) + ": ", {
            "Accuracy": model_accuracy,
            "Precision": model_precision,
            "Recall": model_recall,
            "F1 Score": model_f1
        })

    print("Random Game Index: ", random_game_index)
    print("Random Game Results: ")
    print(random_game_results)

    return all_tests, all_results, random_game_results

def evaluateResults(all_tests, all_results):
    print("Hamming Loss: ", hamming_loss(all_tests, all_results))

def outputResults(all_tests, all_results, random_game_results):

    # Reordering Results for gettags.py
    true_tags = [[0] * len(all_tests) for i in range(len(all_tests[0]))]
    predicted_tags = [[0] * len(all_results) for i in range(len(all_results[0]))]
    random_game_tags = [[0] * len(random_game_results) for i in range(len(random_game_results[0]))]

    for i in range(len(all_tests)):
        for j in range(len(all_tests[0])):
            true_tags[j][i] = all_tests[i][j]
            predicted_tags[j][i] = all_results[i][j]
    
    for i in range(len(random_game_results)):
        for j in range(len(random_game_results[0])):
            random_game_tags[j][i] = random_game_results[i][j]

    # Output Results to File
    pd.DataFrame(true_tags).to_csv(TRUE_TAGS_OUTPUT_FILE, index=False, header=False)
    pd.DataFrame(predicted_tags).to_csv(PREDICTED_TAGS_OUTPUT_FILE, index=False, header=False)
    pd.DataFrame(random_game_tags).to_csv(RANDOM_GAME_OUTPUT_FILE, index=False, header=False)

# Main Program
dataset = pd.read_csv(TRAINING_INPUT_FILE)

all_X, all_y = preprocessDataset(dataset)

all_tests, all_results, random_game_results = runNaiveBayes(all_X, all_y)

evaluateResults(all_tests, all_results)

outputResults(all_tests, all_results, random_game_results)