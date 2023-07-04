#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#import dataset
df=pd.read_csv("C:/Martin james/Data Science/Python Project/Algorithm/Salary_Data.csv")
df.head()
df.shape
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values


# In[3]:


X


# In[4]:


# Visualizing data
sns.barplot(x='YearsExperience',y='YearsExperience', data=df)


# In[5]:


# Splitting Dataset
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test =train_test_split(X,Y, test_size=1/3, random_state=0)


# In[6]:


# Fitting Simple Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, Y_train)


# In[7]:


# Predict the test results:
y_pred=lr.predict(X_test)
y_pred


# In[8]:


# Visualizing of Results- Train Set
plt.scatter(X_train,Y_train, color='purple')
plt.plot(X_train, lr.predict(X_train), color='red')
plt.title('Salary~Experience(Train Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[9]:


# Visualization of Results- Test Dataset
plt.scatter(X_test, Y_test, color='blue')
plt.plot(X_test,lr.predict(X_test),color='red')
plt.title('Salary~Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[10]:


#Calculation of Residuals
from sklearn import metrics
print('MAE', metrics.mean_absolute_error(Y_test,y_pred))
print('MSE', metrics.mean_squared_error(Y_test,y_pred))
print('RMSE', np.sqrt(metrics.mean_absolute_error(Y_test,y_pred)))


# # Logitic Regression

# In[11]:


# Logistic regression is mainly used in clssification in supervised learning e.g the probability that people will repay loan or not
# Usually calculated used using sigmoid curve
# Formulae is p=(1/ (1+e^-z))


# In[12]:


# Implementation of logistic Regression

# loading libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')


# In[13]:


#Loading dataset
# The question of the study is whether a person will be able to buy a SUV car based on age and gender?
social_network= pd.read_csv("C:\Martin james\Data Science\Python Project\Algorithm\SocialNetworkAds.csv")
social_network


# In[14]:


# Extracting the independent and dependent variable

X= social_network.iloc[:,[2,3]].values


# In[15]:


y= social_network.iloc[:, 4].values


# In[16]:


# visualizing data- by drawing a correlation map
sns.heatmap(social_network.corr())


# In[17]:


# Split dataset into training and testing set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=0)


# In[18]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
Sc_X= StandardScaler()
X_train= Sc_X.fit_transform(X_train)
X_test=Sc_X.fit_transform(X_test)


# In[19]:


# Fit logistic Regression to Training Dataset:
from sklearn.linear_model import LogisticRegression
LogR=LogisticRegression(random_state=0)
LogR.fit(X_train,y_train)


# In[20]:


# Predicting the Test Results
y_pred=LogR.predict(X_test)
y_pred


# In[21]:


#Visualization of the Train Set Results
from matplotlib.colors import ListedColormap
X_set,y_set= X_train, y_train
X1, X2= np.meshgrid(np.arange(start=X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step =0.01),
                  np.arange(start=X_set[:,1].min() - 1, stop = X_set[:, 1].max() + 1, step =0.01))
plt.contourf(X1,X2, LogR.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75, cmap = ListedColormap(('blue','yellow')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               c= ListedColormap(('red','green'))(i), label=j)
plt.title('Logistic Regression(Training Set)')
plt.xlabel('Age')
plt.ylabel('Expected Salary')
plt.legend()
plt.show()


# In[22]:


#Visualization of the Test Set Results
from matplotlib.colors import ListedColormap
X_set,y_set= X_test, y_test
X1, X2= np.meshgrid(np.arange(start=X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step =0.01),
                  np.arange(start=X_set[:,1].min() - 1, stop = X_set[:, 1].max() + 1, step =0.01))
plt.contourf(X1,X2, LogR.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha= 0.75, cmap = ListedColormap(('purple','grey')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               c= ListedColormap(('red','green'))(i), label=j)
plt.title('Logistic Regression(Test Set)')
plt.xlabel('Age')
plt.ylabel('Expected Salary')
plt.legend()
plt.show()


# In[23]:


# Evaluating the Model using Confuion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm


# In[24]:


print(63+5+8+24)


# In[25]:


(63+24)/100


# # Decision Tree and K Nearest Neighbour

# In[26]:


# Implimentation of Decision Tree
# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


# importing dataset

kyphosis= pd.read_csv('C:\Martin james\Data Science\Python Project\Algorithm\kyphosis.csv')
kyphosis


# In[28]:


# Extracting independent variable
X=kyphosis.drop('Kyphosis', axis=1)
X


# In[29]:


#Extracting dependent variable
y=kyphosis['Kyphosis']


# In[30]:


#Data Analysis

sns.barplot(x='Kyphosis', y='Age', data=kyphosis)


# In[31]:


sns.pairplot(kyphosis, hue='Kyphosis', palette='Set1')


# In[32]:


# Visualize Data

plt.figure(figsize=(27,7))
sns.countplot(x='Age', hue='Kyphosis', data=kyphosis, palette='Set1')


# In[33]:


# Spliting Dataset i nto Training and Testing Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.3, random_state=100)


# In[34]:


X= kyphosis.drop('Kyphosis', axis=1)
y=kyphosis['Kyphosis']


# In[35]:


X.head()


# In[36]:


X= kyphosis.iloc[:,[1,2,3]].values


# In[37]:


y.head()


# In[38]:


# Training Dataset
from sklearn.tree import DecisionTreeClassifier
Dtree= DecisionTreeClassifier()
Dtree.fit(X_train,y_train)


# In[39]:


# Predicting of the Model
ypred= Dtree.predict(X_test)
ypred


# In[40]:


# Evaluating the Accuracy of the Model
from sklearn.metrics import classification_report
print(classification_report(y_test,ypred))


# In[41]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, ypred))


# In[42]:


# Interpretation of Results
15+7+2+1


# In[43]:


# The maximum diagonal value is 16
# The prediction accuracy is 16/25 = 0.64; 64% of the dataset was collectly classified


# In[44]:


# Random Forest
#Training Dataset
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)


# In[45]:


# predicting of dataset
y_pred=rfc.predict(X_test)


# In[46]:


#Evaluating the model
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,y_pred))


# In[47]:


print(classification_report(y_test,y_pred))


# In[48]:


# Interpretation of the result
""" 
*We get 18/25 with random forest which is a better prediction than with the decision tree, that is 16/25.
*Random forest classification is applied to help in improving the performance of decision tree classification.
* We use mutiple trees in random forest classification unlike in the decision tree where we just use one.
* We take an average of all the trees for each observations which improves the prediction
"""


# In[49]:


# KNearest Neighbours (KNN)
#Used mainly for classification- using the same social_network data that was uploaded earlier

social_network


# In[50]:


# Extracting dependent and independent variables

X= social_network.iloc[:,[2,3]].values


# In[51]:


y= social_network.iloc[:,4].values


# In[52]:


# Spliting Data set into Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=100)


# In[53]:


# Feature Scaling 
from sklearn.preprocessing import StandardScaler
scl=StandardScaler()
X_train= scl.fit_transform(X_train)
X_test=scl.fit_transform(X_test)


# In[54]:


#Training of Data set using K-NN

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train,y_train)


# In[55]:


# Prediction of Results
y_predi= classifier.predict(X_test)
y_predi


# In[56]:


# Evaluation of Results
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_predi))


# In[57]:


#Interpreting the results

# Main diagnol results 

print(72+3+6+39)


# In[58]:


# Accuracy of prediction
print((72+39)/120*100)


# In[59]:


# Thus, the predictions made using the model are 92.5% accurate


# In[62]:


# Visualization of Training Data
X_Set, y_set= X_train, y_train
X1,X2= np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop= X_set[:,0].max()+1, step= 0.01),
                  np.arange(start= X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step=0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap= ListedColormap(('purple', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate (np.unique(y_set)):
     plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1],
    c = ListedColormap(('purple', 'green'))(i), label =j)
plt.title('KNN (Train_set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[61]:


# Visualization of Test Data Set
X_set, y_set= X_test, y_test
X1,X2 = np.meshgrid(np.arange(start= X_set[:,0].min()-1, stop =X_set[:,0].max()+1, step=0.01),
                   np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1, step=0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap =ListedColormap(('grey','purple')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i, j in enumerate(np.unique(y_set)):
     plt.scatter(X_set[y_set==j,0], X_set[y_set==j,1],
                 c = ListedColormap(('purple', 'green'))(i), label =j)
plt.title('KNN (Test_set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
                 


# In[ ]:




