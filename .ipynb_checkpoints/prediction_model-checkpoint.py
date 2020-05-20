#!/usr/bin/env python
# coding: utf-8

# # Prediction of Winner using Logistic Regression
# ---

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# ---
# ## For analysis and insights on this dataset visit <a href="https://github.com/anurag-ux/2019-general-election/blob/master/Analysis.ipynb">here</a> 

# ---
# # Table of Contents
# ##### Navigate through here.
# - [Dataset Overview](#-)
# - [Data Preprocessing](#_)
# - [Handling missing values](#.)
# - [Managing categorical values](#,)
# - [Final overview before modelling data](#')
# - [Learning and Predicting](#~)
# - [Checking the accuracy of our model](#`)

# ###### -
# ---
# # Brief Overview of Dataset

# In[2]:


df=pd.read_csv('dataset/LS_2.0.csv')
df.head(5)


# ###### _
# ---
# # Data Preprocessing
# ---

# In[3]:


# removing non-essential columns
df.drop(['STATE','CONSTITUENCY', 'NAME','SYMBOL','GENERAL\nVOTES', 'POSTAL\nVOTES', 'OVER TOTAL ELECTORS \nIN CONSTITUENCY','OVER TOTAL VOTES POLLED \nIN CONSTITUENCY'],axis=1,inplace=True) 


# In[4]:


## cleaning data for assets column
df['ASSETS'].replace(['Not Available','Nil',np.nan,'`','Telangana'],'0',inplace=True)
a=df['ASSETS'].apply(lambda x:x.split()[1] if len(x.split())>1 else x.split()[0])
b=a.apply(lambda x:x.split(','))
c=b.apply(lambda x:''.join(x))
df['ASSETS']=c
df['ASSESTS']=df['ASSETS'].astype(int)

## cleaning data for liabilities column
df['LIABILITIES'].replace(['Not Available','Nil','NIL',np.nan,'`','Telangana'],'0',inplace=True)
a=df['LIABILITIES'].apply(lambda x:x.split()[1] if len(x.split())>1 else x.split()[0])
b=a.apply(lambda x:x.split(','))
c=b.apply(lambda x:''.join(x))
df['LIABILITIES']=c
df['LIABILITIES']=df['LIABILITIES'].astype(int)


# In[5]:


# renaming some columns (duh!)
df.columns=['WINNER', 'PARTY', 'GENDER', 'CRIMINAL_CASES', 'AGE', 'CATEGORY','EDUCATION', 'ASSETS', 'LIABILITIES', 'TOTAL_VOTES', 'TOTAL_ELECTORS','ASSESTS']


# 
# ##### .
# ---
# # Handling missing values
# ---

# In[6]:


# visualizing the columns with null values
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='summer')
plt.show()


# In[7]:


# checking the number of null values in each feature
df.isnull().sum()


# #### We see that there is a strong chance that some rows have missing data in many features
# #### Even if we try to fill the data for numerical values such as criminal cases, it is not possible to predict the gender or the cateogry of a candidate
# #### Also our dataset has nearly 2200 rows so it is feasible to drop 245 rows for better working of our model

# In[8]:


# dropping the null values
df.dropna(inplace=True)
df.isnull().sum()


# In[9]:


### Noticed that the criminal cases feature has 'Not Available' value which would give an error while training our data
df['CRIMINAL_CASES']=df['CRIMINAL_CASES'].apply(lambda x:0 if x=='Not Available' else x)


# ###### ,
# ---
# # Managing categorical values
# ---

# In[10]:


# we have the following categorical features in our dataset
df[['PARTY','GENDER','CATEGORY','EDUCATION']].head()


# #### Lets focus on the PARTY feature and take a look how many categories are there in the dataset

# In[11]:


df['PARTY'].value_counts()


# ### As we see there are more than 130 parties. we can map all the parties with less than 10 candidates as 'Others'

# In[12]:


df['PARTY']=df['PARTY'].apply(lambda x:x if df['PARTY'].value_counts().loc[x]>10 else 'Others')
df['PARTY'].value_counts()


# ### lets look at the education feature and see what can be done

# In[13]:


df['EDUCATION'].value_counts()


# ### for better working of our model lets merge some values

# In[14]:


df.drop('ASSETS',axis=1,inplace=True)
ill=['Others','Not Available','Illiterate','Post Graduate\n']
df['EDUCATION']=df['EDUCATION'].apply(lambda x:'Illiterate' if x in ill else x)
df['EDUCATION'].value_counts()


# ## Now that we have cleaned our categorical features we need to convert them into dummy values and merge them into our dataframe for the algorithm to recognize them.

# In[15]:


dummy=pd.get_dummies(df[['GENDER','CATEGORY','PARTY','EDUCATION']],drop_first=True)
final=pd.concat([df,dummy],axis=1)


# ### Now we drop our previous categorical columns

# In[16]:


final.drop(['PARTY','GENDER','CATEGORY','EDUCATION'],axis=1,inplace=True)


# ###### '
# ---
# # Now that we have pre-processed our data for the machine learning algorithm lets take a final look at it

# In[17]:


final.head()


# ###### ~
# ----
# # Learning and Predicting

# ---
# #### First we break our data into the design matrix X and output vector y

# In[18]:


X=final.drop('WINNER',axis=1)
y=final['WINNER']


# ### now we split our data for training and testing

# In[19]:


### we'll keep 80% of our data for training and 20% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)


# ## Scaling our data to standard normal distribution

# In[20]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train.loc[:,['CRIMINAL_CASES', 'AGE','LIABILITIES', 'TOTAL_VOTES', 'TOTAL_ELECTORS', 'ASSESTS']]=ss.fit_transform(X_train.loc[:,['CRIMINAL_CASES', 'AGE','LIABILITIES', 'TOTAL_VOTES', 'TOTAL_ELECTORS', 'ASSESTS']])
X_test.loc[:,['CRIMINAL_CASES', 'AGE','LIABILITIES', 'TOTAL_VOTES', 'TOTAL_ELECTORS', 'ASSESTS']]=ss.transform(X_test.loc[:,['CRIMINAL_CASES', 'AGE','LIABILITIES', 'TOTAL_VOTES', 'TOTAL_ELECTORS', 'ASSESTS']])
df.columns


# ### Lets train our data

# In[43]:


from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train,y_train)


# ### Predicting the values in our test set

# In[44]:


predict=lgr.predict(X_test)


# #### Now we have succesfully trained and predicted values for our data so lets check how well the algorithm performed

# ###### `
# ---
# # Checking the accuracy of our model (using k-fold cross validation)

# In[47]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lgr, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# #### Classification Report

# In[24]:


print(classification_report(y_test,predict))


# ## Great!! Our model has an accuracy of 92.07%
# # Thanks for your time!!
# -----

# In[ ]:




