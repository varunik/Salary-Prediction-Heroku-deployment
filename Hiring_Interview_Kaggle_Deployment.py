#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


hiring = pd.read_csv(r"C:\\Users\\varun\\Deployment_Flask_Master\\Hiring_sample_for_deployment\\hiring.csv")


# In[3]:


hiring.isnull().sum()


# In[4]:


hiring


# In[5]:


hiring['experience'].fillna(0,inplace = True)


# In[6]:


hiring['test_score(out of 10)'].fillna(hiring['test_score(out of 10)'].mean(), inplace = True)


# In[7]:


hiring


# In[8]:


def convert_exp_numtoint(exp):
    mydict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,0:0}
    return mydict[exp]


# In[9]:


convert_exp_numtoint('two')


# In[10]:


experience_list = hiring['experience'].tolist()
experience_list


# In[11]:


hiring['experience']=list(map(convert_exp_numtoint,experience_list))


# In[12]:


hiring


# In[13]:


x = hiring.iloc[:,0:3]


# In[14]:


x


# In[15]:


y = hiring.iloc[:,-1]


# In[16]:


y


# In[17]:


# from sklearn import tree
# ctree = tree.DecisionTreeRegressor()
# from sklearn.model_selection import train_test_split
# from sklearn import metrics


# In[156]:


# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[157]:


# y_test


# In[158]:


# ctree.fit(x_train,y_train)


# In[159]:


# y_train_pred = ctree.predict(x_train)
# metrics.accuracy_score(y_train,y_train_pred)


# In[160]:


# y_test_pred = ctree.predict(x_test)
# metrics.accuracy_score(y_test,y_test_pred)


# In[161]:


# df1 = pd.DataFrame()
# df1['a'] = y_train
# df1['b'] = y_train_pred


# In[162]:


# df1


# In[163]:


# df2 = pd.DataFrame()
# df2['a'] = y_test
# df2['b'] = y_test_pred


# In[164]:


# df2


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


regressor = LinearRegression()


# In[20]:


regressor.fit(x,y)


# In[21]:


import pickle


# In[22]:


#saving model to disk


# In[23]:


pickle.dump(regressor,open('model_Flask_master.pk1','wb'))


# In[24]:


pd.show_versions()


# In[ ]:




