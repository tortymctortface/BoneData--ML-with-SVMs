# MLP

## To run locally

Simply install torch, numpy, sklearn and seaborn and you should be good to go. On windows if you have pip installed you can run `pip install PACKAGE_NAME`.

## Running test.py

In the test.py file there are four parameters you can play with:

**EPOCHS**
This is the number of times you want your model to train its weights using the training dataset. As a general rule of thumb, the more epochs you have generally means a higher accuracy rate and a lower loss but also means your model will take loner to train.

**size_labeled** :
This is how many samples you wish to train the data on. In this case `1 <= size_labeled <= 300000` since our training set contains 300000 labeled samples. This allows us to reduce the size of the training set for faster results and also allows us to use part of the pre-labeled data as test data.

**size_test** :
If you are not using the full training set and you wish to use a chunk of the remaining data then you can input the number of samples you want to use as test samples here.
This is useful for testing the accuracy of the method as the samples in the train-io.txt are labeled.
In this case `size_test <= (300000 - size_labeled)`

**LEARNING_RATE** :
The learning rate is the size of the steps you wish to take as your model learns. To big or too small a learning rate and you may risk getting stuck in local minima.
Side Note : I played with a decaying learning rate but the accuracy was so low on all of my tests that I had to scrap it.

#### Output of test.py

This will print to console the predicted labels followed by the confusion matrix and the fitness report.

#### Results from test.py

1. I found that testing this on a large scale quite difficult as trainig the model on a large dataset (somnething with 290000 training samples) took almost 5 hours. This is why many of my tests are in smaller batches.

2. Some of the test results are as follows:
   - With 10000 labeled samples used to train the classifier at a LEARNING_RATE of 0.001 and 20 EPOCHS, 58/100 labels were correctly predicted at an accuracy of 58%. The confusion matrix for this result was [[18,31][11,40]]
   - With 10000 labeled samples used to train the classifier at a LEARNING_RATE of 0.001 and 50 EPOCHS, 62/100 labels were correctly predicted at an accuracy of 62%. The confusion matrix for this result was [[23,26][12,39]]
   - With 10000 labeled samples used to train the classifier at a LEARNING_RATE of 0.001 and 70 EPOCHS, 63/100 labels were correctly predicted at an accuracy of 63%. The confusion matrix for this result was [[24,25][12,39]]
   - With 290000 labeled samples used to train the classifier at a LEARNING_RATE of 0.0001 and 120 EPOCHS, 9003/10000 labels were correctly predicted at an accuracy of 90%. The confusion matrix for this result was [[4083,593][403,4920]]

## Running main.py

Running the main.py file will train the model on the entire training set ( totalt of 300000 samples from `train-io.txt`).
It will then attempt to predict the labels for the samples in `test-in.txt` and will output the results to file called `test.out.txt`, with each label on its own line.
The trained model will also be saved to a file named `my_model.pth.tar`once you are on the 120th epoch. To load this see `test-loaded.py`

## Running test-loaded.py

Running `test-loader.py` will load the trained model in the file named `my_model.pth.tar`. I have saved a model trained on 300000 labeled samples, with 120 epochs and a learning rate of 0.0001. This is in the `Data` folder but will be overwritten when you run the `main.py`locally. You can choose the number of samples from the `test-in.txt` you wish to predict for using the _size_test_ parameter

## Issues

- It was difficult to test large scale due to time constraints.
- Methods such as decaying learning rate seemed useless but I may have put it together wrong so I will need to look into this further.
