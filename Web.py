import streamlit as st

import algorithms as algo

import pandas as pd
import os
 
st.set_page_config(
        page_title="Stock Price Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
text_color = 'violet'

def main():
    st.title(f':{text_color}[Stock Price Prediction]')
    
    col1, col2 = st.columns([.5,1], gap='medium')
    
    with col1:
        st.text('Please Select Stock')
        sd = []
        path = './data'
        for x in os.listdir(path):
#             if x.endswith(".csv"):
                # Prints only text file present in My Folder
                sd.append(x)
#         all_data = pd.read_csv('./../Data/allData.csv')
        stock = st.selectbox(
            'Stock',
            label_visibility='collapsed',
            options=map(lambda x: x.rsplit('.', 1)[0], sd))
        st.text('Please Select Algorithm')
        algorithm = st.selectbox(
            "Algoritm",("Linear Regression","RFR","SVM"),
            label_visibility='collapsed')
        
    with col2:
        load_csv_data = st.checkbox("Upload data")
        if(load_csv_data):
            uploaded_file = st.file_uploader("Upload file", type='csv', label_visibility='collapsed')
        else:
            uploaded_file = None
        if uploaded_file is not None:
            uploaded_file.seek(0)
            stock_data = pd.read_csv(uploaded_file)
            df = pd.DataFrame(stock_data)
            stock = uploaded_file.name.rsplit('.', 1)[0]
        else:
            stock_data = pd.read_csv("./Data/"+stock+".csv")
            df = pd.DataFrame(stock_data)
    
    col1, col2, col3 = st.columns([1.2,.5,1])
    placeholder = st.empty()
    
    with col1:
        st.caption(f'Stock - :{text_color}[**{stock}**]')
        st.caption(f'Algorithm - :{text_color}[**{algorithm}**]')
    
    with col2:
        if st.button("Predict", type='primary'):
            if algorithm == 'Linear Regression':
                with placeholder.container():
                    algo.lr(df)
            if algorithm == 'RFR':
                with placeholder.container():
                    algo.rfr(df)
            if algorithm == 'SVM':
                with placeholder.container():
                    algo.svm(df)
    with col3:
        if st.button("Clear"):
            placeholder.empty()
        
        
if __name__ == "__main__":
    main()