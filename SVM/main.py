from sklearn import svm
import numpy as np
from sklearn.svm import SVC

###########################################
#Training the clf (classifier)
###########################################

def create_labeled_data():
    minlist = list()
    xlist = list()
    ylist = list()
    with open(("Data/train-io.txt"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i <= 300000:
                for word in line.split():
                    minlist.append(word)
                for x, word in enumerate(minlist):
                    if x == 12:
                        ylist.append(word)
                        minlist.remove(word)
                xlist.append(minlist)
                minlist = list()
    return xlist, ylist

x, Y = create_labeled_data()
# convert x into a numpy array
X = np.array(x)
# create instance of svm classifier
clf = SVC(kernel="linear", C=100)
# fit the 12-D input to the correct output of 1 or 0
clf.fit(X, Y)


##############################################################
#Read in test-in and predict labels
##############################################################

def test_data():
    minlist = list()
    queries = list()
    with open(("Data/test-in.txt"), "r", encoding="utf-8") as f:
         for i, line in enumerate(f):
            if i <= 10000:
                for word in line.split():
                    minlist.append(word)
                queries.append(minlist)
                minlist = list()
    return queries

qs = test_data()
# predict either 1 or 0 for each input
predicted_ans = clf.predict(qs)
print(predicted_ans)

##############################################################
#Output result to test-out.txt file
##############################################################

def output_file (predicted_ans):
    with open(("Data/test-out.txt"), "w", encoding="utf-8") as f:
        for label in predicted_ans:
            # write line to output file
            f.write(label)
            f.write("\n")
            
output_file(predicted_ans)

