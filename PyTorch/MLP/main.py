import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split


###########################################
#Set the parameters (see README for details)
###########################################

EPOCHS = 120
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
size_labeled = 300000
size_test = 10000

###########################################
#Selecting the train x and y sets
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
                    minlist.append(float(word))
                for x, word in enumerate(minlist):
                    if x == 12:
                        ylist.append(int(word))
                        minlist.remove(word)
                xlist.append(minlist)
                minlist = list()
    return xlist, ylist

X_train, y_train = create_labeled_data(size_labeled)


#####################################################################
#Selecting the test x and y set
#####################################################################

#to test the ability of this method I added a way to split the training data into a smaller training set and with a section of the training data being used as unlabeled test data 
def create_test_data(a):
    minlist = list()
    queries = list()
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
#Standaradize the datasets
#####################################################################

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

## train data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_data = trainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))

## test data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
test_data = testData(torch.FloatTensor(X_test))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
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
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#############################################################################################
#Saving our model
#############################################################################################
def save_checkpoint(state,filename = "Data/my_model.pth.tar"):
    print ("Saving model...")
    torch.save(state, filename)
    print ("Model Saved")
    
########################################################################################
#Time to train the model
########################################################################################

model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    if e == 120:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict}
        save_checkpoint(checkpoint)
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
      
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} ')
    
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

##############################################################################################
#output predictions to file
##############################################################################################

def output_predictions(y_pred_list):
    y_pred_list = str(y_pred_list)
    print(y_pred_list)
    with open(("Data/test-out.txt"), "w", encoding="utf-8") as f:
        for i in range(len(y_pred_list)):
            if ((y_pred_list[i]) == "0") and ((y_pred_list[i+1]) == "."):
                f.write(y_pred_list[i])
                f.write("\n")
            elif((y_pred_list[i]) == "1") and ((y_pred_list[i+1]) == "."):
                f.write(y_pred_list[i])
                f.write("\n")    

output_predictions(y_pred_list)
