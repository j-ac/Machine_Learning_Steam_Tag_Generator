
For Naive Bayes Model:
Dataset -> prepro.py -> naiveBayesMultinomial.py -> gettags.py

For RNN or CNN: 
Dataset -> prepro.py -> Recurrent_Neural_Network.ipynb / Convolutional_Neural_network.ipynb

File Descriptions:
prepro.py converts dataset into id's associated with unique words and tags
naiveBayesMultinomial.py trains a naive-bayes model based on the output of prepro
Reucrrent_Neural_Network.ipynb trains a RNN model based on the output of prepro
Convolutional_Neural_network.ipynb trains a CNN model based on the output of prepro
gettags.py converts the results of naiveBayesMultinomial.py into human readable words