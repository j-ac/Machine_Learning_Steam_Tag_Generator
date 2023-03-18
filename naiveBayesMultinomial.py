import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dataset
emails = pd.read_csv('emails.csv')
emails.drop(columns="Email No.", inplace=True)

X = emails.iloc[:, :-1].values
y = emails.iloc[:, -1].values

# Preprocessing: Reducing counts more than 1, to 1
X = np.where(X > 1, 1, X)
pd.DataFrame(X).to_csv("emailsMNBPreProcessing.csv", header=None, index=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

# Prediction
y_prediction = mnb.predict(X_test)

# Evaluation
mnb_accuracy = accuracy_score(y_test, y_prediction)
mnb_precision = precision_score(y_test, y_prediction)
mnb_recall = recall_score(y_test, y_prediction)
mnb_f1 = f1_score(y_test, y_prediction)

results = np.asarray([y_test, y_prediction])
pd.DataFrame(results).to_csv("emailsMNBresults.csv", header=None, index=None)

print("Results: ", {
    "Accuracy": mnb_accuracy,
    "Precision": mnb_precision,
    "Recall": mnb_recall,
    "F1 Score": mnb_f1
})