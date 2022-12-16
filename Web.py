#Importing all the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st




def main():
    
    st.set_page_config(
    page_title="Stock market Prediction",
    page_icon=":chart:",
    layout="centered",
#     initial_sidebar_state="expanded",
    )
    
    name = 'TATACOFFEE.NS'

    st.title('Stock Price Prediction')
    st.subheader('Algorithm - Linear Regression')
    st.subheader('Stock Name - '+ name)
    # Importing dataset

    data = pd.read_csv("./Data/"+name+".csv")
    df = pd.DataFrame(data)
    st.subheader('Actual dataset: ')
    st.dataframe(df)
    st.line_chart(data=df['High'], x=None, y=None, width=0, height=0, use_container_width=True)
    st.subheader('Dataset Details: ')
    st.dataframe(df.describe())

    # showing column wise %ge of NaN values they contains 

    for i in df.columns:
      print(i,"\t-\t", df[i].isna().mean()*100)

    # Choosin stock values for any company 

    cormap = df.corr()
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cormap, annot = True)

    def get_corelated_col(cor_dat, threshold): 
      # Cor_data to be column along which corelation to be measured 
      #Threshold be the value above which of corelation to considered
      feature=[]
      value=[]

      for i ,index in enumerate(cor_dat.index):
        if abs(cor_dat[index]) > threshold:
          feature.append(index)
          value.append(cor_dat[index])

      df = pd.DataFrame(data = value, index = feature, columns=['corr value'])
      return df

    top_corelated_values = get_corelated_col(cormap['Close'], 0.60)
#     top_corelated_values

    df = df[top_corelated_values.index]
    df.head()
#     df.shape
    sns.pairplot(df)
    plt.tight_layout()

    X = df.drop(['Close'], axis=1)
    y = df['Close']

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X.head()


    #now lets split data in test train pairs

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

    Acc = []

    from sklearn.linear_model import LinearRegression

    # model training

    model_1 = LinearRegression()
    model_1.fit(X_train, y_train)


    # prediction
    y_pred_1 = model_1.predict(X_test)
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_1})
    pred_df.head()
    st.subheader('Predicted values: ')
    st.dataframe(pred_df)


    # Measure the Accuracy Score

    from sklearn.metrics import r2_score

    print("Accuracy score of the predictions: {0}".format(r2_score(y_test, y_pred_1)))
    Acc.append(r2_score(y_test, y_pred_1))
    st.markdown("Accuracy score of the predictions: {0}".format(r2_score(y_test, y_pred_1))) # see *


    st.line_chart(data=pred_df, x=None, y=None, width=0, height=0, use_container_width=True)
    
    plt.figure(figsize=(8,8))
    plt.ylabel('Close Price', fontsize=16)
    plt.plot(pred_df)
    plt.legend(['Actual Value', 'Predictions'])
    plt.show()

#     st.balloons()
if __name__ == '__main__':
    main()