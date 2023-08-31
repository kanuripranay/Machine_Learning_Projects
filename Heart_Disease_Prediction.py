# STEP 1

# DATA ACQUISITION

## import common modules for data analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## How to read in the data as a Dataset

ds = pd.read_csv("C:/Users/prana/Heart Disease Dataset/archive/heart.csv")

## To see how does the dataset looks like
ds.head()

ds.tail()

## To see the dimensions of dataset
ds.shape

## To see basic discription of dataset
pd.set_option('display.float_format', lambda x: '%.3f' % x)
ds.describe().transpose()


# STEP 2

# DATA FILTERING

# To find the missing values
ds.isnull()

# To see what percentage of missing values are present in each column
ds.isnull().sum()/len(ds)*100

# To check if there are any dublicate rows
ds.duplicated().sum()


# STEP 3

# MAKING A MACHINE LEARNING MODEL

# checking the distribution of Target Variable
ds['target'].value_counts()

# 1--> Defective Heart
# 0--> Non-Defective Heart

X = ds.drop(columns='target',axis=1)
Y = ds['target']

print(X)

print(Y)

# Splitting the Data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Model Training

model = LogisticRegression()

# training the LogisticRegression model with Training Data
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy of Training Data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy of Training Data = ',training_data_accuracy)

# Model Evaluation
# Accuracy of Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy of Test Data = ',test_data_accuracy)

# Building a Predictive System

input_data = (30,0,0,112,149,0,1,125,0,1.6,1,0,2)
 
# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person does not have a Heart Disease')
else:
    print('The person has a Heart Disease')

input_data = (51,1,0,140,298,0,1,122,1,4.2,1,3,3)
 
# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person does not have a Heart Disease')
else:
    print('The person has a Heart Disease')