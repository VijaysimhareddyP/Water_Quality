import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df = pd.read_csv(r"C:\Users\water_potability.csv")
df
df.isnull().sum()
df.boxplot('ph')
df.describe()
df['ph'] = df['ph'].fillna(value = df['ph'].mean())
df['Sulfate'] = df['Sulfate'].fillna(value = df['Sulfate'].mean())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(value = df['Trihalomethanes'].mean())
df.isnull().sum()
df
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_test
y_pred
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
pickle.dump(model,open('note_modelthree.pkl','wb'))
