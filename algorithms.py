text_color = 'violet'

def lr(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st 
    
    
    temp_df = df
    st.caption(f':{text_color}[**Actual dataset:**]')
    st.dataframe(temp_df)
    st.line_chart(data=df['Close'], x=None, y=None, width=0, height=0, use_container_width=True)
    df = pd.DataFrame(df, columns=['Date','Close'])# Create a new DataFrame with only closing price and date
    df = df.reset_index()# Reset index column so that we have integers to represent time 

    # Import package for splitting data set
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.20,random_state=0)

    # Reshape index column to 2D array for .fit() method
    X_train = np.array(train.index).reshape(-1, 1)
    y_train = train['Close']

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create test arrays
    X_test = np.array(test.index).reshape(-1, 1)
    y_test = test['Close']

    # Generate array with predicted values
    y_pred = model.predict(X_test)

    
    st.caption(f':{text_color}[**Predicted values:**]')
    
    # prediction
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    pred_df.head()
    
    col1, col2 = st.columns([.4,1], gap='medium')
    with col1:
        st.dataframe(pred_df)
    
    Acc = []
    # Measure the Accuracy Score
    from sklearn.metrics import r2_score
    Acc.append(r2_score(y_test, y_pred))
    with col2:
        st.markdown("Accuracy score of the predictions: :blue[**{0}**]".format(r2_score(y_test, y_pred)*100)) 
        st.line_chart(data=pred_df, x=None, y=None, width=0, height=0, use_container_width=True)
        
        
def rfr(df):
    import pandas as pd
    import numpy as np
    import math
    import datetime as dt
    from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
    from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
    from sklearn.preprocessing import MinMaxScaler

    from itertools import cycle

    # ! pip install plotly
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import streamlit as st 
    
    
    temp_df = df
    st.caption(f':{text_color}[**Actual dataset:**]')
    st.dataframe(temp_df)
    df = pd.DataFrame(df, columns=['Date','Close'])# Create a new DataFrame with only closing price and date
    df = df.reset_index()# Reset index column so that we have integers to represent time 
    df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)

    
    closedf = df[['date','close']]
    fig = px.line(closedf, x=closedf.date, y=closedf.close,labels={'date':'Date','close':'Close Stock'})
    fig.update_traces(marker_line_width=2, opacity=0.6)
    fig.update_layout(title_text='Stock close price chart', font_size=15, font_color='white')
#     fig.update_xaxes(showgrid=False)
#     fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    close_stock = closedf.copy()
    del closedf['date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

    training_size=int(len(closedf)*0.65)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
    
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    
    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(X_train, y_train)
    
    # Lets Do the prediction 

    train_predict=regressor.predict(X_train)
    test_predict=regressor.predict(X_test)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)
    
    # Transform back to original form

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
    
    # shift train predictions for plotting

    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
#     print("Train predicted data: ", trainPredictPlot.shape)

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
#     print("Test predicted data: ", testPredictPlot.shape)
    
    names = cycle(['Original close price','Train predicted close price','Test predicted close price'])
    plotdf = pd.DataFrame({'date': close_stock['date'],
                           'original_close': close_stock['close'],
                          'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                          'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['test_predicted_close']],
                  labels={'value':'Stock price','date': 'Date'})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price', font_size=15,legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    
    out_arr = np.nan_to_num(trainPredictPlot) 
    out_arr2 = np.nan_to_num(testPredictPlot) 
    arr = out_arr + out_arr2
    from sklearn.metrics import r2_score
    st.markdown(f"Accuracy score of the predictions: :{text_color}[**{format(r2_score(plotdf['original_close'], arr)*100)}**]") 
    
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def svm(df):
    # Import Python Librariesimport numpy as np
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from itertools import cycle
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVR
    import pandas_datareader.data as web
    import datetime as dt
    
    temp_df = df
    st.caption(f':{text_color}[**Actual dataset:**]')
    st.dataframe(temp_df)
    df = pd.DataFrame(df, columns=['Date','Close'])# Create a new DataFrame with only closing price and date
    df = df.reset_index()# Reset index column so that we have integers to represent time 
    df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close"}, inplace= True)

    
    closedf = df[['date','close']]
    fig = px.line(closedf, x=closedf.date, y=closedf.close,labels={'date':'Date','close':'Close Stock'})
    fig.update_traces(marker_line_width=2, opacity=0.6)
    fig.update_layout(title_text='Stock close price chart', font_size=15, font_color='white')
#     fig.update_xaxes(showgrid=False)
#     fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    df = df[['close']]
    df['Prediction'] = df['close'].shift(-15)
    X = np.array(df.drop(['Prediction'],1))
    #Remove the last 15 rows
    X = X[:-15]
    
    y = np.array(df['Prediction'])
    # Remove Last 15 rows
    y = y[:-15]
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # SVM Model
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # Train the model 
    svr.fit(x_train, y_train)
    svm_confidence = svr.score(x_test, y_test)
    forecast = np.array(df.drop(['Prediction'],1))[-15:]
    svm_prediction = svr.predict(forecast)
    
    names = cycle(['Original close price','Predicted close price'])
    

    fig = px.line(df,x=None, y=[df['close'],df['Prediction']],
                  labels={'value':'Stock price','': ''})
    fig.update_layout(title_text='Comparision between original close price vs predicted close price', font_size=15,legend_title_text='Close Price')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    
    from sklearn.metrics import r2_score
    st.markdown(f"Confidence score of the predictions: :{text_color}[**{svm_confidence}**]") 
    
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)