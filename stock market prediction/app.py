from json import load
from pyexpat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st



start='2010-01-01'
end='2019-12-31'

st.title('stock Trend Prediction')

user_input=st.text_input("Enter Stock Ticker","AAPL")
df= data.DataReader(user_input,'yahoo', start,end)
df.head()

#Describing data
st.subheader("Date from 2010-2019")
st.write(df.describe())

#Visualization
st.subheader('Closing Price Vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price Vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


#splitting data into training and testng
data_train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_train.shape)
print(data_test.shape)

from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler(feature_range=(0,1))

data_train_array = Scaler.fit_transform(data_train)


#splitting data into xtrain and y train


#loadmodel
model=load_model('hrkmodel.h5')


#testing part
past_100days=data_train.tail(100)
final_df=past_100days.append(data_test,ignore_index=True)
input_data=Scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

#making predictions

y_predicted=model.predict(x_test)
scaler=Scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#final graph
st.subheader("prediction Vs Original")
fig2=plt.figure(figsize=(12,6))

plt.plot(y_test,'b',label="Original Price")
plt.plot(y_predicted,'r',label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)