# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 21:32:07 2025

@author: acrossler
"""

import pyodbc
import pandas as pd
from geopy.distance import geodesic as GP
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Create the connection
conn = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};"
                      "Server=localhost\\SQLEXPRESS2022;"
                      "Database=CreditCards;"
                      "Trusted_Connection=yes;"
                      "TrustServerCertificate=yes;")

query = "SELECT [transactionid],[userid],CAST([time] as datetime2) AS time,[amount],[state],[city],[country],[tran_lat],[tran_lon],[merchantType],[isFraud] FROM [Transaction]"
query2 = "SELECT [userid],[home_lat],[home_lon],[state],[city],[country],[age],[marital_status],[isMale]FROM [User]"
transaction = pd.read_sql(query, conn)
user = pd.read_sql(query2, conn)
print(transaction)
print(user)
#Close it after I run my queries
conn.close()

transaction.to_csv("transaction3.csv", index = False)
user.to_csv("user3.csv", index = False)


#Creating the variables needed
tran = transaction


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

# , "hoursSinceLastTran", "distFromLastTran", "tranDistFromHome"
#Logistic regression
# X = data[["hoursSinceLastTran", "distFromLastTran", "tranDistFromHome"]]
X = data[["hoursSinceLastTran", "distFromLastTran", "tranDistFromHome"]]
X = sm.add_constant(X)
y = data["isFraud"]

model = sm.Logit(y, X).fit()


y_prob = model.predict(X)
y_predict = (y_prob >= 0.35).astype(int)
#Summary of model
print(model.summary())
#Lag 3, 5, and Volume are statistically relevent
conM = confusion_matrix(y, y_predict)

print(conM)
# Display confusion matrix
fig2 = plt.Figure()
sns.heatmap(conM, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()