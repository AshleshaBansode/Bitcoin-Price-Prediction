# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:18:15 2018

@author: prachi
"""
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime as DT
import datetime
from datetime import date, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil import parser
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler


def predictions(df, Daily_Price, count, loop_counter):
    
    df_test = Daily_Price[:(len(Daily_Price))]
    df_train= Daily_Price
    top_date = df_test.index[0]
    top_date_1 = parser.parse(top_date)
    plus_two_days_series = pd.Series([0.0],index=[((top_date_1 + datetime.timedelta(days=1)).strftime('%d-%b-%y'))])
    #plus_two_days_series
    df_test = plus_two_days_series.append(df_test)
    print("df_test length: "+str(len(df_test)))
    working_data = [plus_two_days_series,df_train]
    working_data = pd.concat(working_data)
    working_data = working_data.reset_index()
    working_data['index'] = pd.to_datetime(working_data['index'])
    working_data = working_data.set_index('index')
    
        #Seasonal Decomposition
    s = sm.tsa.seasonal_decompose(df.Close.values, freq=60)
    if count == loop_counter-1:
        print("Inside Seasonal Decomposition: "+ str(count))
        trace1 = go.Scatter(x = np.arange(0, len(s.trend), 1),y = s.trend,mode = 'lines',name = 'Trend',
            line = dict(color = ('rgb(244, 146, 65)'), width = 4))
        trace2 = go.Scatter(x = np.arange(0, len(s.seasonal), 1),y = s.seasonal,mode = 'lines',name = 'Seasonal',
            line = dict(color = ('rgb(66, 244, 155)'), width = 2))
        
        trace3 = go.Scatter(x = np.arange(0, len(s.resid), 1),y = s.resid,mode = 'lines',name = 'Residual',
            line = dict(color = ('rgb(209, 244, 66)'), width = 2))
        
        trace4 = go.Scatter(x = np.arange(0, len(s.observed), 1),y = s.observed,mode = 'lines',name = 'Observed',
            line = dict(color = ('rgb(66, 134, 244)'), width = 2))
        
        data = [trace1, trace2, trace3, trace4]
        layout = dict(title = 'Seasonal decomposition', xaxis = dict(title = 'Time'), yaxis = dict(title = 'Price, USD'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='seasonal_decomposition')

    def create_lookback(dataset, check_previous=1):
        X, Y = [], []
        for i in range(len(dataset) - check_previous):
            a = dataset[i:(i + check_previous), 0]
            X.append(a)
            Y.append(dataset[i + check_previous, 0])
        return np.array(X), np.array(Y)
    
    training_set = df_train.values
    len(training_set)
    training_set = np.reshape(training_set, (df_train.size , 1))
    test_set = df_test.values
    test_set = np.reshape(test_set, (df_test.size, 1))
    
    #scale datasets
    scaler = MinMaxScaler()
    training_set = scaler.fit_transform(training_set)
    test_set = scaler.transform(test_set)
    
    look_back = 1
    X_train, Y_train = create_lookback(training_set, look_back)
    X_test, Y_test = create_lookback(test_set, look_back)
    X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(256))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, Y_train, epochs=100, batch_size=1000, shuffle=False, 
                        validation_data=(X_test, Y_test), 
                        callbacks = [EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1)])
    
    #if count == loop_counter-1:
    print("Inside loss graph: "+ str(count))
    trace1 = go.Scatter(
        x = np.arange(0, len(history.history['loss']), 1),
        y = history.history['loss'],
        mode = 'lines',
        name = 'Train loss',
        line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')
    )
    trace2 = go.Scatter(
        x = np.arange(0, len(history.history['val_loss']), 1),
        y = history.history['val_loss'],
        mode = 'lines',
        name = 'Test loss',
        line = dict(color=('rgb(244, 146, 65)'), width=2)
    )
    
    data = [trace1, trace2]
    
    if count == loop_counter-1:  #Plotting Graph
        layout = dict(title = 'Train and Test Loss during training', 
                      xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='training_process')
    
    # add one additional data point to align shapes of the predictions and true labels
    X_test = np.append(X_test, scaler.transform(working_data.iloc[-1][0]))
    X_test = np.reshape(X_test, (len(X_test), 1, 1))
    
    # get predictions and then make some transformations to be able to calculate RMSE properly in USD
    prediction = model.predict(X_test)
    prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
    Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
    prediction2_inverse = np.array(prediction_inverse[:,0][1:])
    Y_test2_inverse = np.array(Y_test_inverse[:,0])
    
    #Pranav_Changes
    Y_test2_inv = ([float('naN')])
    Y_test2_inv_new = Y_test2_inverse[0:((len(Y_test2_inverse))-1)]   
    Y_test_inv_test = np.concatenate((Y_test2_inv,Y_test2_inv_new))
    
    
    
    print("Inside result graph: "+str(count))
    trace1 = go.Scatter(
        x = np.arange(0, len(prediction2_inverse), 1),
        y = prediction2_inverse,
        mode = 'lines',
        name = 'Predicted price',
        hoverlabel= dict(namelength=-1),
        line = dict(color=('rgb(244, 146, 65)'), width=2)
    )
    trace2 = go.Scatter(
        x = np.arange(0, len(Y_test2_inverse), 1),
        y = Y_test2_inverse,
        mode = 'lines',
        name = 'True price',
        line = dict(color=('rgb(66, 244, 155)'), width=2)
    )
    data = [trace1, trace2]
    
    if count == loop_counter-1: #Prining Graph 2
        layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted',
                     xaxis = dict(title = 'Day number'), yaxis = dict(title = 'Price, USD'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='results_demonstrating25')
    
    
    #mean = np.mean((Y_test2_inv_new[:10])-((np.round((prediction2_inverse[:(len(prediction2_inverse)-1)]),decimals=2,out=None)))[:10])
    mean = 1.0
    prediction_after_mean = prediction2_inverse[:(loop_counter)] + mean
    Actual_Data = np.roll(Y_test2_inv_new,1, axis=None)
    Actual_Data[0] = np.nan
    Pre_Predicted_Data = (np.round_((prediction2_inverse[:(len(prediction2_inverse)-1)]), decimals=2,out=None))
    Bias_Adjusted_Prediction = ((((Actual_Data - Pre_Predicted_Data)+Pre_Predicted_Data))-150)
    temp = ([float('naN')])
    Bias_Adjusted_Prediction = np.concatenate((Bias_Adjusted_Prediction,temp))
    Bias_Adjusted_Prediction[0]=np.round(Pre_Predicted_Data[0],decimals=2,out=None)
    Bias_Adjusted_Prediction_Final = np.concatenate((prediction_after_mean,Bias_Adjusted_Prediction[(loop_counter):]))
    
    for i in range (loop_counter):
        Y_test_inv_test[i] = float('naN')
        
    if count == loop_counter-1: #Printing final Graph
        Test_Dates = df_test.index[:(loop_counter+5)]
        
        Test_Dates = Test_Dates[::-1]
        Y_test_inv_test = Y_test_inv_test[::-1]
        Bias_Adjusted_Prediction_Final = Bias_Adjusted_Prediction_Final[::-1]
        Y_test_inv_test = Y_test_inv_test[(len(Y_test_inv_test))-(len(Test_Dates)):]
        Bias_Adjusted_Prediction_Final = Bias_Adjusted_Prediction_Final[((len(Bias_Adjusted_Prediction_Final))-(len(Test_Dates))):]

        trace1 = go.Scatter(x=Test_Dates, y=Y_test_inv_test, name= 'Actual Price', 
                           line = dict(color = ('rgb(66, 244, 155)'),width = 2))
        trace2 = go.Scatter(x=Test_Dates, y=Bias_Adjusted_Prediction_Final, name= 'Predicted Price',
                           line = dict(color = ('rgb(244, 146, 65)'),width = 2))
        data = [trace1, trace2]
        
       
        layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
                     xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='D:/work/SOC/project/FlaskV5/Flask/templates/Dated_Prediction')
    #, filename='C:/Users/pratiksha/results_demonstrating_25_1'
    
    next_day_predicted =  pd.Series([prediction2_inverse[0]],index=[df_test.index[0]])
    return next_day_predicted

def daily(input_date):
    py.init_notebook_mode(connected=True)
    df = pd.read_csv('D:/work/SOC/project/FlaskV3/Flask/coinmarket.csv')
    df.columns=['','Date','Open','High','Low','Close','Volume','Market Cap']
    #Series is generated for closing values and Date as index
    Daily_Price=pd.Series(df['Close'].values,index=df['Date'])
    now = (DT.today()).date()
    yesterday = now - timedelta(days=1) # date - days
    print("Yesterday"+str(yesterday))
    input_date = DT.strptime(input_date, '%Y-%m-%d')
    print(type(input_date))
    print("input date - " + str(input_date))
    loop_counter = abs((input_date.date() - yesterday).days)
    for i in range(loop_counter):
        print("**$$ Recurrenc No. (Predicting Value for Day->) : "+str(i+1))
        next_day_predicted = predictions(df, Daily_Price, i,loop_counter)
        print("Iteration: "+str(i))
        print("next_day_predicted: "+ str(next_day_predicted))
        Daily_Price = next_day_predicted.append(Daily_Price)
        array=[]
        array.insert(0, {'': 0, 'Date': next_day_predicted.index[0], 'Open': 0,'High': 0, 'Low':0, 'Close': next_day_predicted[0], 'Volume':0, 'Market Cap': 0})
        df = pd.concat([pd.DataFrame(array),df], ignore_index=True)
        len(df)
        print("Next Dy Predicted Series: \n")
        print(next_day_predicted)
        print("\n\nDaily Price Head :\n")
        print(Daily_Price.head(10))
        print("\n\n DF :\n")
        print(df.head(10))


def weekly():
    py.init_notebook_mode(connected=True)
    df = pd.read_csv('D:/work/SOC/project/FlaskV3/Flask/coinmarket.csv')
    df.columns=['','Date','Open','High','Low','Close','Volume','Market Cap']
    #Series is generated for closing values and Date as index
    Daily_Price=pd.Series(df['Close'].values,index=df['Date'])
    now = (DT.today()).date()
    input_date = now + timedelta(days=6)
    yesterday = now - timedelta(days=1) # date - days
    #input_date = DT.strptime(input_date, '%Y-%m-%d')
    print(type(input_date))
    print("input date - " + str(input_date))
    loop_counter = abs((input_date - yesterday).days)
    for i in range(loop_counter):
        print("**$$ Recurrenc No. (Predicting Value for Day->) : "+str(i+1))
        next_day_predicted = predictions(df, Daily_Price, i,loop_counter)
        print("Iteration: "+str(i))
        print("next_day_predicted: "+ str(next_day_predicted))
        Daily_Price = next_day_predicted.append(Daily_Price)
        array=[]
        array.insert(0, {'': 0, 'Date': next_day_predicted.index[0], 'Open': 0,'High': 0, 'Low':0, 'Close': next_day_predicted[0], 'Volume':0, 'Market Cap': 0})
        df = pd.concat([pd.DataFrame(array),df], ignore_index=True)
        len(df)
        print("Next Dy Predicted Series: \n")
        print(next_day_predicted)
        print("\n\nDaily Price Head :\n")
        print(Daily_Price.head(10))
        print("\n\n DF :\n")
        print(df.head(10))
