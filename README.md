# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

Step 3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

Step 4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

Step 6. Stop

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Alan Samuel Vedanayagam
RegisterNumber: 212223040012
*/
```
```
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/570bcf05-e9ea-4671-9856-4350efde3e8a)

```
d1=df.copy()
d1=d1.drop(["sl_no","salary"],axis=1)
d1.head()
```
![image](https://github.com/user-attachments/assets/f8b1cf6e-c5d4-4d84-81aa-0a213581926a)

```
d1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/3192d5a0-693d-49e3-ad96-f6876c477e78)
```
d1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/2c51d03c-9027-454c-9a2b-e6568420caec)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['gender']=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
```
![image](https://github.com/user-attachments/assets/5afbc431-39c1-4fa7-abdf-c004c01c7fbf)
```
x=d1.iloc[:, : -1]
x
```
![image](https://github.com/user-attachments/assets/2de3bbd3-3780-4848-bddc-c08fc2bad141)
```
y=d1["status"]
y
```
![image](https://github.com/user-attachments/assets/71001520-b338-4d2b-84fb-9ae5f2d12132)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/8de43793-a9bf-4953-94cd-3fd93de1727d)
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/40ece254-212a-4bb3-a474-8b349a625a46)
```
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/c0c78472-2848-47cf-9774-2e276d1b2088)
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
![image](https://github.com/user-attachments/assets/be483f25-99aa-40ce-9f7d-83d864cda6c7)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
