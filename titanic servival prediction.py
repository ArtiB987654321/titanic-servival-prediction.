#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
train = pd.read_csv("C:/Users/admin/Desktop/archive (1)/Titanic-Dataset.csv")


# In[3]:


train


# In[4]:


train.head()


# In[5]:


train.shape


# In[6]:


import seaborn as sns

sns.countplot(x='Survived', data = train)


# In[7]:


#those who did not survived(more than 500) are greater than those who survived(nearly 300)


# In[8]:


sns.countplot(x='Survived', hue='Sex', data = train, palette ='winter')


# In[9]:


# Analysis: 0 represents not survived and 1 is for survives
# women are thrice more likely to survive than males.


# In[10]:


sns.countplot(x='Survived', hue='Pclass', data=train)


# In[11]:


## Analysis: the passangers who did not survived belong to the 3rd class.
## 1st class passangers are more likely to survive


# In[12]:


train['Age'].plot.hist()


# In[13]:


# we notice that highest age group travelling are among the young age between 20-40.
# very few passangers in age group 70-80


# In[14]:


train['Fare'].plot.hist(bins=20, figsize=(10,5))


# In[15]:


# we observe that most of the tickets bought are under fare 100
# and very few are on the higher side of fare i.e. 220-500 range


# In[16]:


sns.countplot(x='SibSp', data=train, palette='rocket')


# In[17]:


# we notice that most of the passangers do not have their siblings aboard.


# In[18]:


train['Parch'].plot.hist()


# In[19]:


sns.countplot(x='Parch', data=train, palette='summer')


# In[20]:


# the number of parent and siblings who aboard the ship are less


# In[21]:


# Data wrangling means the cleaning the data, removing the null values, 
# droping unwanted columns, adding new ones if needed


# In[22]:


train.isnull().sum()


# In[23]:


# age and cabin has most null values. and embarked too has null values
# we can plot it on heat map


# In[24]:


sns.heatmap(train.isnull(), cmap='spring')


# In[25]:


# here yellow color is showing the null values, highest in cabin followed by age


# In[26]:


sns.boxplot(x='Pclass', y='Age', data=train)


# In[27]:


# we can obseved than older age group are travelling more in class 1 and 2
# campaired to class 3


# In[28]:


# the hue parameter determines which column in the data frame should be used for color encoding .


# In[29]:


# we will drop a few column now


# In[30]:


train.head()


# In[31]:


train.drop('Cabin', axis = 1, inplace = True)


# In[32]:


train.head(3) # dropped the cabin column


# In[33]:


train.dropna(inplace=True)


# In[34]:


sns.heatmap(train.isnull(), cbar=False)


# In[35]:


# this shows that we don't have any null values, we can also checkk it:


# In[36]:


train.isnull().sum()


# In[37]:


train.head(2)


# In[38]:


# we will convert the few columns (strings) into categoriat data to apply logistic regression


# In[39]:


pd.get_dummies(train['Sex']).head()


# In[40]:


sex=pd.get_dummies(train['Sex'], drop_first=True)
sex.head(3)


# In[41]:


# we have droped the first column because only one column is sufficient to determine
# the gender of the passenger either wiil be male(1) or not(0), that means a female


# In[42]:


embark=pd.get_dummies(train['Embarked'])


# In[43]:


embark.head(3)


# In[44]:


# C  stands for Cherbourg, Q for Queenstown, S for Southhampton.
# we can drop any one of the column as we can infer from the two columns itself


# In[45]:


embark=pd.get_dummies(train['Embarked'], drop_first=True)


# In[46]:


embark.head(3)


# In[47]:


# if both values are 0 then passanger is travelling in 1st class.


# In[48]:


Pcl=pd.get_dummies(train['Pclass'], drop_first=True)
Pcl.head(3)                


# In[49]:


# our data is now converted into categorial data


# In[50]:


train=pd.concat([train, sex, embark, Pcl], axis=1)


# In[51]:


train.head(3)


# In[52]:


# deleting the unwanted columns


# In[53]:


train.drop(['Name',"PassengerId", 'Pclass','Ticket','Sex','Embarked'], axis=1, inplace=True)


# In[54]:


train.head(3)


# In[55]:


X=train.drop('Survived', axis=1)
y=train['Survived']


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)


# In[58]:


print(X.shape,X_train.shape,X_test.shape)


# In[59]:


# Logistic Regression


# In[60]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)


# In[61]:


model=LogisticRegression()


# In[62]:


model.fit(X_train, y_train)


# In[63]:


prediction=model.predict(X_test)


# In[64]:


from sklearn.metrics import classification_report


# In[67]:


from sklearn.metrics import classification_report


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


confusion_matrix(y_test, prediction)


# In[71]:


from sklearn.metrics import accuracy_score


# In[72]:


accuracy_score(y_test,prediction)


# In[ ]:




