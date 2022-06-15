#!/usr/bin/env python
# coding: utf-8

# ### PROYECTO p1

# Lectura de datos:  https://docs.google.com/document/d/1zTVhqSv-rKO5ki9c6h9dKqMSgrYWopX7RdarSlTL46I/edit?usp=sharing

# In[293]:


from skimage import io
img_src='https://content.healthwise.net/resources/13.3/es-us/media/medical/hw/s_h9991362_004.jpg'
image=io.imread(img_src)
io.imshow(image)
io.show()


# In[294]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("dis2.csv") 
print(data) 


# In[295]:


data.head()


# In[296]:


data.columns


# In[297]:


data.tail()


# In[298]:


pd.unique(data['referral source'])


# In[299]:


data['class'].describe()


# In[300]:


data['age'].min()
data['age'].max()
data['age'].count()


# In[301]:


class_counts = data.groupby('class')['referral source'].count()
print(class_counts)


# In[302]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Creaemos una gráfica de barras
class_counts.plot(kind='bar');


# In[339]:


grupos = data.groupby('age')


# In[340]:


age_counts=data.groupby('age')['class'].count()
print(age_counts)
age_counts.plot(subplots=True,figsize=(10,10),sharey=False)
plt.show()


# MAYORES a 90

# In[341]:


data_df = pd.read_csv("dis2.csv")
data[data_df.age >= '90']


# ### P2 clasificacion no supervisado

# In[342]:


import csv
import numpy as np
from sklearn import preprocessing
with open('dis.csv') as f:
    datos = list(csv.reader(f, delimiter=","))

X_train= np.array(data)


# In[343]:


try:
    scaler = preprocessing.StandardScaler().fit(X_train)
    print("datos procesados", scaler)
except:
    print ("datos con string")


# In[344]:


X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  # apply scaling on training data
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('logisticregression', LogisticRegression())])

pipe.score(X_test, y_test) 


# In[345]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)


# In[346]:


#mapeo de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
np.percentile(X_train[:,0], [0, 25, 50, 75, 100]) 


# In[347]:


pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
X_lognormal


# In[348]:


pt.fit_transform(X_lognormal)


# In[349]:


plt.hist(X_train)

