from sklearn import svm
import numpy as np
from sklearn.tree import DecisionTreeClassifier 

###########################################
#Set the parameters (see README for details)
###########################################

size_labeled = 300
size_test = 100
md = 60

###########################################
#Training the clf (classifier)
###########################################

#z is the number of samples you want to use from the training set. This starts from line 1 and goes to line z , inclusive and is set above using size_labeled 
def create_labeled_data(z):
    minlist = list()
    xlist = list()
    ylist = list()
    with open(("Data/train-io.txt"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i <= z:
                for word in line.split():
                    minlist.append(word)
                for x, word in enumerate(minlist):
                    if x == 12:
                        ylist.append(word)
                        minlist.remove(word)
                xlist.append(minlist)
                minlist = list()
    return xlist, ylist

x, Y = create_labeled_data(size_labeled)
# convert x into a numpy array
X = np.array(x)
# create instance of svm classifier
clf = DecisionTreeClassifier(max_depth = md)
# fit the 12-D input to the correct output of 1 or 0
clf.fit(X, Y)

#####################################################################
#Selecting the test set and predicting their classification (1 or 0)
#####################################################################

#to test the ability of this method I added a way to split the training data into a smaller training set and with a section of the training data being used as unlabeled test data 
def create_test_data(z, a):
    minlist = list()
    queries = list()
    answers = list()
    with open(("Data/train-io.txt"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > z and i <= z + a:
                for word in line.split():
                    minlist.append(word)
                for x, word in enumerate(minlist):
                    if x == 12:
                        answers.append(word)
                        minlist.remove(word)
                queries.append(minlist)
                minlist = list()
    return queries, answers


qs, ans = create_test_data(size_labeled, size_test)
# predict either 1 or 0 for each input
predicted_ans = clf.predict(qs)
print(predicted_ans)
print(ans)

#########################################################################
#Homemade accuracy function based entirely on the number of true outputs
#########################################################################


def accuracy(ans, pr, a):
    accuracy_rate = 0
    for i, letter in enumerate(ans):
        if letter == pr[i - 1]:
            accuracy_rate += 1
    percentage_accuracy_rate = (accuracy_rate / a) * 100
    print(
        accuracy_rate,
        " out of ",
        a,
        ". This is an accuracy rate of ",
        percentage_accuracy_rate,
        "%",
    )


accuracy(ans, predicted_ans, size_test)
