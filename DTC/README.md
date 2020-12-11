# DTC

## To run locally

Simply install sklearn and numpy and you should be good to go. On windows if you have pip installed you can run `pip install sklearn` and `pip install numpy`.

## Test.py

In the test.py file there are three parameters you can play with:

**size_labeled** :
This is how many samples you wish to train the data on. In this case `1 <= size_labeled <= 300000` since our training set contains 300000 labeled samples. This allows us to reduce the size of the training set for faster results and also allows us to use part of the pre-labeled data as test data.

**size_test** :
If you are not using the full training set and you wish to use a chunk of the remaining data then you can input the number of samples you want to use as test samples here.
This is useful for testing the accuracy of the method as the samples in the train-io.txt are labeled.
In this case `size_test <= (300000 - size_labeled)`

**md** :
After a quick google I found max depth is exactly what it sounds like it is. The max depth of a decision tree can also be described as the length of the longest path from the tree root to a leaf. The root node is considered to have a depth of 0. The Max Depth value cannot exceed 30 on a 32-bit machine.

#### Output of test.py

This will print to console the predicted labels, followed by the actaul labels, followed by the percentage of labels correctly guessed.

#### Results from test.py

1. The first thing I noticed here was that adjusting the md seemed to randomly affect the accuracy of the predictions. It seems up until a certain depth the accuracy improves and at a certain point it begins to decrease again. I presume I will have the most fun trying to calculate the best parameters for the best prediction accuracy.

2. I noticed it returned results, even from large datasets, relatively fast. #

3. The other results are as follows:
   - With 300 labeled samples used to train the classifier at a max depth of 2, 54/100 labels were correctly predicted at an accuracy of 54%
   - With 300 labeled samples used to train the classifier at a max depth of 30, 52/100 labels were correctly predicted at an accuracy of 52%
   - With 300 labeled samples used to train the classifier at a max depth of 60, 52/100 labels were correctly predicted at an accuracy of 52%
   - With 100000 labeled samples used to train the classifier at a max depth of 2, 497/1000 labels were correctly predicted at an accuracy of 49.7%
   - With 100000 labeled samples used to train the classifier at a max depth of 30, 505/100 labels were correctly predicted at an accuracy of 50.5%
   - With 100000 labeled samples used to train the classifier at a max depth of 60, 491/100 labels were correctly predicted at an accuracy of 49.1%

## Main.py

Running the main.py file will train the classifer (clf) on the entire training set ( totalt of 300000 samples from `train-io.txt`).
It will then attempt to predict the labels for the samples in `test-in.txt` and will output the results to file called `test.out.txt`, with each label on its own line.

## Issues

- The optimal md value and how to find it is elusive to me right now.
