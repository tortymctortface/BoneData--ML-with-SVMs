# Overview of project

In this project I aim to find the most accurate binary classifier to predict the labels of the 10000 samples in the `text-in.txt`file.
Each sample is a line of 12 numbers and each sample will either have a corresponding label of 0 or 1.
To train my classifiers I have a file `train-io.txt` which contains 300000 pre-labeled samples.
Accuracy will be measured on the total number of labels correctly predicted.

## File overview

The `Compressed_Data` folder contains the two compressed versions of the `text-in.txt` file and the `train-io.txt` file. Once decompressed the files are saved in a folder called `Data`.

The folders `SVM`,`DTC` and `KNN` all contain a different method of classifying the data in the `test-in.txt` file. For training purposes they each use the labeled data in the `train-io.txt` file. The output of each is a file called `test-out.txt`, also in the Data folder, which consists of 10000 lines with either a singular `1` or `0` on each line corresponding to the lines in the `test-in.txt` file.

For a more detailed discussion on how each method preformed please see the _README_'s in each of the folders above.

## General local setup

To test each of these methods locally you will need to decompress both of the files in the `Compressed_Data` to a folder called `Data`. Also remember to run each file from the `ML-Methods--Python` directory. More details for each method can be found in their individual _README_'s
