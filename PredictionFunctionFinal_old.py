"""
Created on Sun Nov 25 11:18:15 2018
@author: prachi
"""
import warnings
warnings.filterwarnings("ignore")

import datetime
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


def predictions(df, Daily_Price):
    
    df_test = Daily_Price[:8]
    df_train= Daily_Price
    top_date = df_test.index[0]
    top_date_1 = parser.parse(top_date)
    plus_one_days_series = pd.Series([0.0],index=[((top_date_1 + datetime.timedelta(days=1)).strftime('%d-%b-%y'))])
    plus_one_days_series
    df_test = plus_one_days_series.append(df_test)
    print("df_test length: "+str(len(df_test)))
    working_data = [plus_one_days_series,df_train]
    working_data = pd.concat(working_data)
    working_data = working_data.reset_index()
    working_data['index'] = pd.to_datetime(working_data['index'])
    working_data = working_data.set_index('index')
    
        #Seasonal Decomposition
    s = sm.tsa.seasonal_decompose(df.Close.values, freq=60)
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
    history = model.fit(X_train, Y_train, epochs=50, batch_size=16, shuffle=False, 
                        validation_data=(X_test, Y_test), 
                        callbacks = [EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1)])
    
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
    layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted',
                 xaxis = dict(title = 'Day number'), yaxis = dict(title = 'Price, USD'))
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='results_demonstrating25')
        
        
        
    Test_Dates = df_test.index
    trace1 = go.Scatter(x=Test_Dates, y=Y_test2_inverse, name= 'Actual Price', 
                       line = dict(color = ('rgb(66, 244, 155)'),width = 2))
    trace2 = go.Scatter(x=Test_Dates, y=prediction2_inverse, name= 'Predicted Price',
                       line = dict(color = ('rgb(244, 146, 65)'),width = 2))
    data = [trace1, trace2]
    layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
                 xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='results_demonstrating25_1')
    #, filename='C:/Users/pratiksha/results_demonstrating_25_1'
    
    next_day_predicted =  pd.Series([prediction2_inverse[0]],index=[Test_Dates[0]])
    return next_day_predicted


def init(date):
    print(date)
    py.init_notebook_mode(connected=True)
    df = pd.read_csv('D:/work/SOC/project/FlaskV3/Flask/coinmarket.csv')
    df.columns=['','Date','Open','High','Low','Close','Volume','Market Cap']
    #Series is generated for closing values and Date as index
    Daily_Price=pd.Series(df['Close'].values,index=df['Date'])
    next_day_predicted = predictions(df, Daily_Price)
    print("next_day_predicted: "+ str(next_day_predicted))
    Daily_Price = next_day_predicted.append(Daily_Price)
   
