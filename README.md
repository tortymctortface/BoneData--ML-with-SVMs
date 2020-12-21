# Overview of project

This is an assignment in the basics of supervised machine learning as part of a final year machine learning module.

In this project I aim to find the most accurate binary classifier to predict the labels of the 10000 samples in the `text-in.txt`file.
Each sample is a line of 12 numbers and each sample will either have a corresponding label of 0 or 1.
To train my classifiers I have a file `train-io.txt` which contains 300000 pre-labeled samples.
Accuracy will be measured on the total number of labels correctly predicted.

The model will be graded with 1 point for every correctly predicted 0 and 3 points for every correctly predicted 1, so because of this the model will need to be bias towards 1 outputs.

## File overview

The `Compressed_Data` folder contains the two compressed versions of the `text-in.txt` file and the `train-io.txt` file. Once decompressed the files are saved in a folder called `Data`.

The `Data` folder contains my pre-tained model (`my_model.pth.tar`) along with my predictions for the `test-in.txt` in a file called `test-out.txt`.

The `Answer for assignment` folder contains a backup of my `test-out.txt` incase it is overwritten.

The folder `Sklearn`contains `SVM`,`DTC` and `KNN`, which all contain a different method of classifying the data in the `test-in.txt` file. For training purposes they each use the labeled data in the `train-io.txt` file. The output of each is a file called `test-out.txt`, also in the Data folder, which consists of 10000 lines with either a singular `1` or `0` on each line corresponding to the lines in the `test-in.txt` file.

The folder `PyTorch`contains `MLP` which works in the same way as above but using the pytorch library instead of sklearn for the classifier.

For a more detailed discussion on how each method preformed please see the _README_'s in each of the folders above.

## General local setup

To test each of these methods locally you will need to decompress both of the files in the `Compressed_Data` to a folder called `Data`. Also remember to run each file from the `ML-Methods--Python` directory. More details for each method can be found in their individual `README's`

## Answer to assignment

- For the answer for my assignement I have included a `test-out.txt` file in the `Data` folder which contains my 10000 predicted outputs for each of the samples in the `test-in.txt`file. I used the PyTorch library as I found it easiest to achieve the highest accuracy in after some tests on each model. For more information on this see the README.md in the `PyTorch` folder.

- To run this locally simply decompress the files in `Compressed Data` into the `Data` directory.
- To use my pre-trained model, set the _size_labeled_ in `PyTorch\CNN\test-loaded.py` to 10000 and run it. _Note_ : This will only output to console.
- To retrain the model using my default parameters simply run the `PyTorch\CNN\main.py`. _Caution_ : This will overwrite my output in the `Data\test-out.txt`file. If this happens there is a backup in the `Answer for assignment` folder.

**References**

1. https://pythonprogramming.net/training-deep-learning-neural-network-pytorch/
2. https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
3. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
4. Stack Overflow and GeeksforGeeks
