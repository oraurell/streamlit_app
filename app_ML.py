import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


# Import profiling capability
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML stuff

from pycaret.regression import *

with st.sidebar:
    st.image('image.png')
    st.title('Auto Machine Learning')
    choice = st.radio("Selección", ["Upload/Dataset", "Profiling","Time Series Forecasting", "ML Predict price"])
    st.info("Esta app ejecuta algoritmos de machine learning de manera automática")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload/Dataset":
    
    file = st.file_uploader("Sube tu archivo csv/xlsx:")
    if file :
        try:
            df = pd.read_csv(file, index_col = None)
            df.to_csv("sourcedata.csv", index = None)
        except:
            df = pd.read_excel(file, index_col = None)
            df.to_csv("sourcedata.csv", index = None) 
        st.dataframe(df)
    
    button_data_1 = st.button("Datos Titanic ML")
    if button_data_1:
        df = pd.read_csv('titanic.csv', index_col = None)
        df.to_csv("sourcedata.csv", index = None)
        st.dataframe(df)
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv().encode('utf-8'),
            file_name='titanic.csv',
            mime='text/csv')
    
    button_data_2 = st.button("Datos Pasajeros TimeSeries")
    if button_data_2:
        df = pd.read_csv('AirPassengers.csv', index_col = None)
        df.to_csv("sourcedata.csv", index = None)
        st.dataframe(df)
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv().encode('utf-8'),
            file_name='airpassengers.csv',
            mime='text/csv')  
    
    button_data_3 = st.button("Datos Pisos Barcelona")
    if button_data_3:
        df = pd.read_csv('housing-barcelona.csv', index_col = None)
        df.to_csv("sourcedata.csv", index = None)
        st.dataframe(df)
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv().encode('utf-8'),
            file_name='housing-barcelona.csv',
            mime='text/csv')  

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)

if choice == "Time Series Forecasting":
    st.info("Serie de datos")
    st.dataframe(df)
    fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Air Passenger Travel")
    st.plotly_chart(fig)
    df.columns = ["ds","y"]
    df['ds']= pd.to_datetime(df['ds']) 
    
    model = Prophet()
    model.fit(df)
    fig_1 = model.plot_components(model.predict(df))
    st.write(fig_1)
    
    st.info("Resultados test Prophet")
    forecast = model.predict(df[132:])
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    fig_2 = model.plot(forecast)
    st.write(fig_2)
    y_true = df['y'][-12:].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    st.info('Error Absoluto Medio = MAE: %.3f' % mae)
    
    st.info("Predicción próximos 5 años")
    future = model.make_future_dataframe(periods=12*5, freq='M')
    forecast = model.predict(future)
    forecast = np.round(forecast, decimals = 3)
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']].tail())
    fig_3 = model.plot(forecast)
    st.write(fig_3)
    st.download_button(
        label="Download data as CSV",
        data = forecast.to_csv(),
        file_name='predict_housing-barcelona.csv',
        mime='text/csv')  

if choice == "ML Predict price":
    st.info("Serie de datos")
    st.dataframe(df.head())
    
    st.info("Limpieza de datos")
    df = df[['price','room','space']]
    df['price'] = df['price'].str.extract("(\d*\.?\d+)", expand=True)
    df['price'] = df['price'].str.replace(r"(\d)\.", r"\1")
    df['room'] = df['room'].str.extract("(\d*\.?\d+)", expand=True)
    df['space'] = df['space'].str.extract("(\d*\.?\d+)", expand=True)
    df = df.apply (pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df[['room','space','price']]
    st.dataframe(df.head())

    regression= setup(data = df, target = 'price', normalize = True, html = False)
    setup_df = pull()

    st.info("Setup de modelos")
    st.dataframe(setup_df)
    best_model = compare_models(sort = 'MAE')
    compare_df = pull()
    st.info("Resultados de modelos")
    st.dataframe(compare_df)


