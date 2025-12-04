# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:20:47 2025

@author: darth
"""

#Creating a training and testing set and seeing how the results are for each model
import pandas as pd
from geopy.distance import geodesic as GP
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xbg
from xgboost import XGBClassifier


#Creating the variables needed
tran = pd.read_csv("transaction3.csv")
user = pd.read_csv("user3.csv")

#Finding the distance of transaction from home location for a user
df = pd.merge(tran, user, on = "userid", how = "inner")
# print(df.head())

# print(GP((12, 15), (10, 10)).mi)

#apply to create a new column. axis = 1 lets it know the row, not the column
df["tranDistFromHome"] = df.apply(
    lambda row : GP(
        (row["tran_lat"], row["tran_lon"]), (row["home_lat"], row["home_lon"])
        ).mi,
    axis = 1
    )

# print(df["tranDistFromHome"])

df_sorted = df.sort_values(
    by = ["userid", "time"])

df_sorted["time"] = pd.to_datetime(df_sorted["time"])

df_sorted["prevTime"] = df_sorted.groupby("userid")["time"].shift(1)


df_sorted["timeSinceLastTran"] = df_sorted.apply(
    lambda row: (row["time"]) - (row["prevTime"])
    if pd.notnull(row["prevTime"]) else None,
    axis = 1
    )

df_sorted["hoursSinceLastTran"] = df_sorted["timeSinceLastTran"].dt.total_seconds() / 3600

#Take in a row of sorted. If the row before is the same user then do it
#Creating two columns that shift the lat and lon values by one for each user
df_sorted[["prevTranLat", "prevTranLon"]] = df_sorted.groupby("userid")[["tran_lat", "tran_lon"]].shift(1)

#each row affected
df_sorted["distFromLastTran"] = df_sorted.apply(
    lambda row: GP(
        (row["tran_lat"], row["tran_lon"]),
        (row["prevTranLat"], row["prevTranLon"])
        ).mi
    #if the row with a shifted value is not null do this, else set to None
    if pd.notnull(row["prevTranLat"]) else None,
    axis = 1 #the lambda is the row, not column, which is axis = 0
    )

data = df_sorted

mymap = {False:0, True:1}
data["isFraud"] = data["isFraud"].map(mymap)
#Remove rows of NA from the dataframe as break model. 
data = (data[data["distFromLastTran"].notna()])

#Train Test split
Train = data[data["userid"] <= 150]
Test = data[data["userid"] > 150]

#Logistic regression
logxTrain = Train[["amount", "age", "tranDistFromHome", "hoursSinceLastTran", "distFromLastTran"]]
logxTrain = sm.add_constant(logxTrain)
logyTrain = Train["isFraud"]
logxTest = Test[["amount", "age", "tranDistFromHome", "hoursSinceLastTran", "distFromLastTran"]]
logxTest = sm.add_constant(logxTest)
logyTest = Test["isFraud"]

model = sm.Logit(logyTrain, logxTrain).fit()


y_probLog = model.predict(logxTest)
y_predictLog = (y_probLog >= 0.37).astype(int)
#Summary of model
# print(model.summary())
#Lag 3, 5, and Volume are statistically relevent
conMLog = confusion_matrix(logyTest, y_predictLog)

print("Logistic Regression")
print(conMLog)
# Display confusion matrix
fig2 = plt.Figure()
sns.heatmap(conMLog, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix Logistic Regression")
plt.show()

conAcc = (conMLog[0][0]+conMLog[1][1])/((conMLog[1][1] + conMLog[1][0])+(conMLog[0][0] + conMLog[0][1]))
print(conAcc)

#QDA model
xTrain = Train[["amount", "age", "tranDistFromHome", "hoursSinceLastTran", "distFromLastTran"]]
yTrain = Train["isFraud"]
xTest = Test[["amount", "age", "tranDistFromHome", "hoursSinceLastTran", "distFromLastTran"]]
yTest = Test["isFraud"]

model2 = QDA()
model2.fit(xTrain, yTrain)

# y_predict = model.predict(X)
y_prob = model2.predict_proba(xTest)[:, 1]  # Probability of fraud
threshold = 0.36  # try 0.05–0.2 depending on class ratio
y_predictQDA = (y_prob >= threshold).astype(int)
conMQDA = confusion_matrix(yTest, y_predictQDA)

print("QDA")
print(conMQDA)
# Display confusion matrix
fig2 = plt.Figure()
sns.heatmap(conMQDA, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

conAcc = (conMQDA[0][0]+conMQDA[1][1])/((conMQDA[1][1] + conMQDA[1][0])+(conMQDA[0][0] + conMQDA[0][1]))
print(conAcc)


#Tensorflow Model
xTrain = Train[["amount", "age", "tranDistFromHome", "hoursSinceLastTran", "distFromLastTran"]]
yTrain = Train["isFraud"]
xTest = Test[["amount", "age", "tranDistFromHome", "hoursSinceLastTran", "distFromLastTran"]]
yTest = Test["isFraud"]


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, activation = "relu", input_shape=(5,)),#Input, change the number if I change the number of input variables
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(5, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
    ])

#Adam optimizer as normal and accuracy score
#As it is a binary classification, l
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs = 20)
model.evaluate(xTest, yTest)

y_prob = model.predict(xTest)
threshold = 0.45  # try 0.05–0.2 depending on class ratio
y_predictTF = (y_prob >= threshold).astype(int)
conMTF = confusion_matrix(yTest, y_predictTF)

print(conMTF)
# Display confusion matrix
fig2 = plt.Figure()
sns.heatmap(conMTF, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

conAcc = (conMTF[0][0]+conMTF[1][1])/((conMTF[1][1] + conMTF[1][0])+(conMTF[0][0] + conMTF[0][1]))
print(conAcc)

#Logistic regression
modelTrainX = Train[["amount", "age", "tranDistFromHome", "hoursSinceLastTran", "distFromLastTran"]]
modelTrainy = Train["isFraud"]
modelTestX = Test[["amount", "age", "tranDistFromHome", "hoursSinceLastTran", "distFromLastTran"]]
modelTesty = Test["isFraud"]

model = XGBClassifier(
    n_estimators = 60,
    learning_rate = 0.09,
    max_depth = 3,
    subsample = 0.8,
    colsample_bytree = 0.8
    )

model.fit(modelTrainX, modelTrainy)

y_predict = model.predict(modelTestX)
conMLog = confusion_matrix(modelTesty, y_predict)

print("Boosted Tree")
print(conMLog)
# Display confusion matrix
fig2 = plt.Figure()
sns.heatmap(conMLog, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix Logistic Regression")
plt.show()

conAcc = (conMLog[0][0]+conMLog[1][1])/((conMLog[1][1] + conMLog[1][0])+(conMLog[0][0] + conMLog[0][1]))
print(conAcc)


