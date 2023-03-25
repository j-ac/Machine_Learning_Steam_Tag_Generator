import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

import csv

# input CSV file
input_file = "training.csv"

# output CSV file
output_file = "output.csv"

# open input and output files
with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    # create csv reader and writer objects
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # iterate over each row
    for row in reader:
        # remove brackets from two columns
        row[0] = row[0].replace("[", "").replace("]", "")
        row[1] = row[1].replace("[", "").replace("]", "")

        # write modified row to output file
        writer.writerow(row)

# print success message
print("Data saved to", output_file)

import csv

# Open the CSV file
with open('output.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

# Find the highest number of integers in each column
max_ints_col1 = 0
max_ints_col2 = 0

for row in rows[1:]:
    col1_ints = len(row[0].split(','))
    col2_ints = len(row[1].split(','))

    if col1_ints > max_ints_col1:
        max_ints_col1 = col1_ints

    if col2_ints > max_ints_col2:
        max_ints_col2 = col2_ints

print(max_ints_col2)
# Fill the rows with fewer integers with zeros at the end
for row in rows[1:]:
    col1_ints = len(row[0].split(','))
    col2_ints = len(row[1].split(','))

    if col1_ints < max_ints_col1:
        row[0] += ',0' * (max_ints_col1 - col1_ints)

    if col2_ints < max_ints_col2:
        row[1] += ',0' * (max_ints_col2 - col2_ints)

# Write the updated rows to a new CSV file
with open('updated_filename.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

filename = 'updated_filename.csv'

# Load the training data from the CSV file
train_data = pd.read_csv('updated_filename.csv', header=None, names=['tags', 'desc'])

print(train_data['tags'][1:])
# print(np.fromstring(train_data['tags'][1], dtype=int, sep=','))

# Convert the data into arrays of integers
train_tags = np.array([np.fromstring(x, dtype=int, sep=',') for x in train_data['tags'][1:]])
# print(train_tags.shape)
train_desc = np.array([np.fromstring(x, dtype=int, sep=',') for x in train_data['desc'][1:]])
print(train_desc)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=10000, activation='softmax'))

# Compile the model with categorical cross-entropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', run_eagerly=True)

# Train the model with the training data
print(train_desc.shape)
print(train_tags.shape)
model.fit(train_desc, train_tags, epochs=10, batch_size=32)

# Generate game tags based on game descriptions
def generate_game_tags(desc):
    desc = np.array([np.fromstring(desc.strip('[]'), dtype=int, sep=',')])
    tags = model.predict(desc)
    return [np.argmax(tag) for tag in tags]

# Example usage
desc = '[1,2,3,4,5,6]'
tags = generate_game_tags(desc)


