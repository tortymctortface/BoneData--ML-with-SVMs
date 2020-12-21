import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

###########################################
#Set the parameters (see README for details)
###########################################

BATCH_SIZE = 64
size_test = 10

#####################################################################
#Selecting the test x and y set
#####################################################################

#to test the ability of this method I added a way to split the training data into a smaller training set and with a section of the training data being used as unlabeled test data 
def create_test_data(a):
    minlist = list()
    queries = list()
    answers = list()
    with open(("Data/test-in.txt"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < a:
                for word in line.split():
                    minlist.append(float(word))
                queries.append(minlist)
                minlist = list()
    return queries

X_test = create_test_data(size_test)

#####################################################################
#Standaradize the input
#####################################################################

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

## test data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
test_data = testData(torch.FloatTensor(X_test))

test_loader = DataLoader(dataset=test_data, batch_size=1)

###################################################################################
#Build the model
###################################################################################

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12 and we have 4 layers.
        self.layer_1 = nn.Linear(12, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 64)
        self.layer_4 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = binaryClassification()
model.to(device)
#Load Model   
checkpoint = torch.load("Data/my_model.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
    
#############################################################################################
#Time to predict - The torch.round has been biased towards 1 if the output is .44444 or above 
# as there are more points for correct true outputs than for correct false outputs
#############################################################################################
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        if (float(y_test_pred)) <= 0.44444:
            y_pred_tag = torch.round(torch.tensor([[0.1]]))
        else:
            y_pred_tag = torch.round(torch.tensor([[0.9]]))
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

print(y_pred_list)