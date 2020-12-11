# KNN

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

**k** :
K is number of nearest neighbours that can "vote" on the class the piece of data should be classified as.

#### Output of test.py

This will print to console the predicted labels, followed by the actaul labels, followed by the percentage of labels correctly guessed.

#### Results from test.py

1. As with other methods, finding the correct k value will be difficult.

2. I noticed it returned results, even from large datasets, relatively fast.

3. The other results are as follows:
   - With 300 labeled samples used to train the classifier at a k value of 2, 53/100 labels were correctly predicted at an accuracy of 53%
   - With 300 labeled samples used to train the classifier at a k value of 7, 49/100 labels were correctly predicted at an accuracy of 49%
   - With 300 labeled samples used to train the classifier at a k value of 15, 48/100 labels were correctly predicted at an accuracy of 48%
   - With 100000 labeled samples used to train the classifier at a k value of 2, 520/1000 labels were correctly predicted at an accuracy of 52%
   - With 100000 labeled samples used to train the classifier at a k value of 7, 493/100 labels were correctly predicted at an accuracy of 49.3%
   - With 100000 labeled samples used to train the classifier at a k value of 15, 475/100 labels were correctly predicted at an accuracy of 47.5%

## Main.py

Running the main.py file will train the classifer (clf) on the entire training set ( totalt of 300000 samples from `train-io.txt`).
It will then attempt to predict the labels for the samples in `test-in.txt` and will output the results to file called `test.out.txt`, with each label on its own line.

## Issues

- The optimal k value seems to be a lower k value, I will need to look into this more.
- The results are not as accurate as the DTC, but that could change with the right k value
- Warnings occur when running the program that may need to be looked into too.
