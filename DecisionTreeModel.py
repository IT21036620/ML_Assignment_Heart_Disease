#!/usr/bin/env python
# coding: utf-8

# In[20]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


# In[16]:


#load the dataset
df = pd.read_csv('heart_2020_cleaned.csv')


# In[15]:


#displays the first few rows of a DataFrame
df.head()


# In[3]:


#check for missing values
df.isnull().sum()


# In[4]:


#get summary statistics
df.describe()


# In[12]:


#find unique values
df.nunique()


# In[6]:


#preprocessing
df = df[df.columns].replace({'Yes':1, 'No':0, 'Male':1, 'Female':0, 'No, borderline diabetes':0,'Yes (during pregnancy)':1, 'Y':1, 'N':0})
df.head()


# In[8]:


#Split dataset for training and testing
y=df['HeartDisease']
X=df.drop(['HeartDisease', 'AgeCategory', 'Race', 'GenHealth'], axis=1)


# In[9]:


#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[17]:


#Train the decision tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[23]:


# predict the test set resut
predictions_tree = clf.predict(X_test)

# Accuracy of predicted data with the actual data
train = clf.score(X_train, y_train)
test = clf.score(X_test, y_test)

print(f"training score = {train}")
print(f"testing score = {test}")


# In[25]:


# Create DataFrame with actual and predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions_tree})

# Print the DataFrame (optional)
print(comparison_df.head(10))


# In[21]:


#Calculate and print performance metrics
predictions_tree = clf.predict(X_test)
accuracy_tree = clf.score(X_test, y_test)
confusion_matrix_tree = confusion_matrix(y_test, predictions_tree)
precision = precision_score(y_test, predictions_tree)
recall = recall_score(y_test, predictions_tree)
f1 = f1_score(y_test, predictions_tree)

print("Confusion matrix for Decision Tree")
print(confusion_matrix_tree)
print(f"Accuracy for Decision Tree = {accuracy_tree*100}%")
print(f"Precision for Decision Tree = {precision:.4f}") 
print(f"Recall for Decision Tree = {recall:.4f}") 
print(f"F1-score for Decision Tree = {f1:.4f}")  


# In[14]:


#visualize confusion matrix
plt.figure(figsize=(5, 4)) 
plt.imshow(confusion_matrix_tree, cmap=plt.cm.Blues)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()


# In[ ]:




