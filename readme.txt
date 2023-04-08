
For Naive Bayes Model:
Dataset -> prepro.py -> naiveBayesMultinomial.py -> gettags.py

For RNN or CNN: 
Dataset -> prepro.py -> Recurrent_Neural_Network.ipynb / Convolutional_Neural_network.ipynb

File Descriptions:
prepro.py: converts dataset into id's associated with unique words and tags
naiveBayesMultinomial.py: trains a naive-bayes model based on the output of prepro
Reucrrent_Neural_Network.ipynb: trains a RNN model based on the output of prepro
Convolutional_Neural_network.ipynb: trains a CNN model based on the output of prepro
gettags.py: converts the results of naiveBayesMultinomial.py into human readable words
functs.py: contains helper functions for loading data and printing measurements for NN models

required libraries for all files:
- Numpy
- Pandas
- Tensorflow and Keras
- Sklearn
- csv
- nltk

IMPORTANT NOTES: 
- 'Loss_Over_Time.png' and 'Accuracy_Over_Time.png' both show plots for the respective metrics 
measured for RNN training run.

- 'log.txt' shows logged metrics for RNN training run.

- For convenience, latest proper output for both RNN and CNN notebooks is saved.

- Output for RNN notebook's last cell is large (logs for 200 epochs), you'll only see final results
at the very bottom, or if you check the respective png files.

- CNN is currently not fully functional, the code itself is included to show progress made
on that front.