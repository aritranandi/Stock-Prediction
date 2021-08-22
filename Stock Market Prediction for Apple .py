#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader as pdr
key="5bbde61d68063a2e63c34865c8466dac476a7960"


# In[2]:


df = pdr.get_data_tiingo('AAPL', api_key=key)


# In[3]:


df.to_csv('AAPL.csv')


# In[4]:


import pandas as pd


# In[5]:


df=pd.read_csv('AAPL.csv')


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df1=df.reset_index()['close']


# In[9]:


df1


# In[10]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[11]:


import numpy as np


# In[12]:


df1


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[14]:


print(df1)


# In[15]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[16]:


training_size,test_size


# In[17]:


train_data


# In[18]:


import numpy

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]  
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[19]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[20]:


print(X_train.shape), print(y_train.shape)


# In[21]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[23]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[24]:


model.summary()


# In[25]:



model.summary()


# In[46]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[27]:


import tensorflow as tf


# In[28]:


tf.__version__


# In[29]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[30]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[31]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[32]:


math.sqrt(mean_squared_error(ytest,test_predict))


# In[33]:


look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[34]:


len(test_data)


# In[35]:



x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[36]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[37]:


temp_input


# In[38]:


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[39]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[40]:


import matplotlib.pyplot as plt


# In[41]:


len(df1)


# In[42]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[43]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[44]:


df3=scaler.inverse_transform(df3).tolist()


# In[45]:


plt.plot(df3)


# In[ ]:




