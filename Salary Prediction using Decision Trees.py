import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('Salary Data.csv')
df.head(10)
df.info()
df.dropna(inplace=True)
df.drop_duplicates()
df.info()
df['Gender'].unique()
gender_label = LabelEncoder()
df['Gender']=gender_label.fit_transform(df['Gender'])
df.head()
df['Gender'].value_counts().plot(kind='pie')
df['Education Level'].value_counts().plot(kind='pie')
edu_label_encoder = LabelEncoder()
df['Education Level'] = edu_label_encoder.fit_transform(df['Education Level'])
job_title_encoder = LabelEncoder()
df['Job Title']=job_title_encoder.fit_transform(df['Job Title'])
df.head()

Y = df['Salary']
X = df.drop(['Salary'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
model=DecisionTreeRegressor(max_depth=5)
model.fit(x_train, y_train)
print(model.score(x_test, y_test)*100)
from sklearn import metrics
y_pred = model.predict(x_test)
print('r2 : ', metrics.r2_score(y_pred, y_test)*100)
# Scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.show()
