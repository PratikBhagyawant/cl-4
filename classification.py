import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

df=pd.read_csv('heart_disease.csv')
df.head()

df.isnull().sum()

df.describe()

df.info()

df.dtypes

x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

model1=SVC(kernel='linear')
model2=LogisticRegression()
model3=RandomForestClassifier()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

y_pred1=model1.predict(x_test)
y_pred2=model2.predict(x_test)
y_pred3=model3.predict(x_test)

print("Classification report of SVC\n",classification_report(y_pred1,y_test))
print("Classification report of Logistic Regression\n",classification_report(y_pred2,y_test))
print("Classification report of Random Forest Classifier\n",classification_report(y_pred3,y_test))

print("Confusion Matrix of SVC")
sns.heatmap(confusion_matrix(y_test,y_pred1),annot=True,cmap='coolwarm')

print("Confusion Matrix of Logistic Regression")
sns.heatmap(confusion_matrix(y_test,y_pred2),annot=True,cmap='coolwarm')

print("Confusion Matrix of Random Forest")
sns.heatmap(confusion_matrix(y_test,y_pred3),annot=True,cmap='coolwarm')

print("Accuracy Score of SVC:", accuracy_score(y_test, y_pred1))
print("Accuracy Score of Logistic Regression:", accuracy_score(y_test, y_pred2))
print("Accuracy Score of Random Forest:", accuracy_score(y_test, y_pred3))