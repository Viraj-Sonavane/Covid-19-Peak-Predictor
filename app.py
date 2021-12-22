import streamlit as st
import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import geopandas as gpd


from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import itertools
import ast

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Bidirectional
from keras.layers.advanced_activations import LeakyReLU
import streamlit as st

import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.arima import ADFTest

st.set_page_config(layout="wide")

st.title('Covid-19 Peak Wave Predictor')

#Fetching datset using url
DATE_COLUMN = 'date'
DATA_URL = ('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')


add_selectbox = st.sidebar.selectbox(
    "Choose a Model for Predictions",
    ("None","LSTM", "ARIMA", "FBProphet","Retuned_FBProphet")
)

cdata = pd.read_csv(DATA_URL)

cdata.drop(['excess_mortality_cumulative_per_million','excess_mortality_cumulative_absolute','excess_mortality_cumulative','excess_mortality','human_development_index','hospital_beds_per_thousand','handwashing_facilities','male_smokers','female_smokers','gdp_per_capita','diabetes_prevalence','cardiovasc_death_rate','extreme_poverty','gdp_per_capita','population_density','reproduction_rate','continent',],axis=1, inplace= True)
cdata.drop(['icu_patients','icu_patients_per_million','hosp_patients','hosp_patients_per_million','weekly_hosp_admissions','weekly_hosp_admissions_per_million','weekly_icu_admissions','weekly_icu_admissions_per_million'],axis=1,inplace= True)
cdata.fillna(0,inplace=True)

cnation = (cdata['location'].unique())
selected_nation = st.selectbox('Select nation for prediction', cnation)
#@st.cache(allow_output_mutation=True, show_spinner=True,suppress_st_warning=True)

#----------------------------------------------------------------------------------------------------------------------------------#
#Data Interpolation
#----------------------------------------------------------------------------------------------------------------------------------#
#Ploting location vs new cases per million
def Data_analysis(nation):
    def NewCases(nation):    
        fig = plt.figure(figsize=(5,5))
        plt.title('New Covid-19 cases per million in '+ nation, fontsize=30,color='white')
        graph = sns.lineplot(data=cdata[cdata['location'].isin([nation])].sort_values(by='date'), x='date', y='new_cases_per_million')
        graph.xaxis.set_major_locator(mdates.DayLocator(interval = 60))
        plt.xticks(rotation = 'vertical')
        st.plotly_chart(fig,use_container_width=True)
    
    def Pos(nation):    
        fig = plt.figure(figsize=(5,5))
        plt.title('New Covid-19 deaths per million in '+ nation, fontsize=30,color='white')
        graph = sns.lineplot(data=cdata[cdata['location'].isin([nation])].sort_values(by='date'), x='date', y='new_deaths_per_million')
        graph.xaxis.set_major_locator(mdates.DayLocator(interval = 60))
        plt.xticks(rotation = 'vertical')
        st.plotly_chart(fig,use_container_width=True)

    def Vacc(nation):    
        fig = plt.figure(figsize=(5,5))
        plt.title('Total Covid-19 cases in '+ nation, fontsize=30,color='white')
        graph = sns.lineplot(data=cdata[cdata['location'].isin([nation])].sort_values(by='date'), x='date', y='total_cases')
        graph.xaxis.set_major_locator(mdates.DayLocator(interval = 60))
        plt.xticks(rotation = 'vertical')
        st.plotly_chart(fig,use_container_width=True)

    NewCases(nation)
    Vacc(nation)
    Pos(nation)

#----------------------------------------------------------------------------------------------------------------------------------#
#Basic FBProphet Model
#----------------------------------------------------------------------------------------------------------------------------------#
def FBProphet_forecast(nation):
  n_years = st.slider('Months of prediction:', 12, 1)
  period = n_years * 30

  data = cdata[cdata['location'].isin([nation])].sort_values(by="date")[['date',"new_cases_per_million"]]
  data.columns = ['ds','y']
  model = Prophet(daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=True,interval_width=0.95)
  model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
  model.fit(data)
  future = model.make_future_dataframe(periods=period, freq='D')
  pred = model.predict(future)
  model.plot(pred);
  model.plot_components(pred);
  st.write(f'Forecast plot for next {period} Days')
  fig1 = plot_plotly(model, pred)
  fig1.update_layout(plot_bgcolor = "lightgrey")
  st.plotly_chart(fig1,use_container_width=True)

  st.write("Forecast components")
  fig2 = model.plot_components(pred)
  st.write(fig2)

#----------------------------------------------------------------------------------------------------------------------------------#
#FBprophet Model Hyperparameter Tunning
#----------------------------------------------------------------------------------------------------------------------------------#


@st.cache(allow_output_mutation=True, show_spinner=True,suppress_st_warning=True)
def Tunning(nation):  
    data = cdata[cdata['location'].isin([nation])].sort_values(by="date")[['date',"new_cases_per_million"]]
    data.columns = ['ds','y']

    def create_param_combinations(**param_dict):
        param_iter = itertools.product(*param_dict.values())
        params =[]
        for param in param_iter:
            params.append(param) 
        params_df = pd.DataFrame(params, columns=list(param_dict.keys()))
        return params_df

    def single_cv_run(history_df, metrics, param_dict):
        m = Prophet(**param_dict)
        m.fit(history_df)
        df_cv = cross_validation(m, initial='400 days', period='90 days', horizon = '180 days')
        df_p = performance_metrics(df_cv).mean().to_frame().T
        df_p['params'] = str(param_dict)
        #df_p = df_p.loc[:, metrics]
        df_p = df_p.reindex(columns = metrics)
        return df_p

    param_grid = {  
                'changepoint_prior_scale': [0.005, 0.05, 0.5, 5],
                'changepoint_range': [0.8, 0.9],
                'seasonality_prior_scale':[0.1, 1, 10.0],
                'seasonality_mode': ['multiplicative', 'additive'],
                'growth': ['linear'],
                'yearly_seasonality': [5, 10, 20]
              }

    metrics = ['horizon', 'rmse', 'mape', 'params'] 
    results = []
    params_df = create_param_combinations(**param_grid)
    for param in params_df.values:
        param_dict = dict(zip(params_df.keys(), param))
        cv_df = single_cv_run(data,  metrics, param_dict)
        results.append(cv_df)
    results_df = pd.concat(results).reset_index(drop=True)
    best_param = results_df.loc[results_df['mape'] == min(results_df['mape']), ['params']]
    print(f'\n The best param combination is {best_param.values[0][0]}')
    m = best_param.values[0][0]
    return m


#FBProphet Model Build
def Final_FBProphe_Forecast(nation):
    n_years = st.slider('Months of prediction:', 12, 1)
    period = n_years * 30

    q = Tunning(nation)
    n = ast.literal_eval(q)
    m = ast.literal_eval(q)
    dd = []
    for key, value in m.items():
        print(str(key), str(value))
        dd.append((key,value))

    changepoint_prior_scale = dd[0][0]
    changepoint_prior_scale_no = dd[0][1]

    changepoint_range = dd[1][0]
    changepoint_range_no = dd[1][1]

    seasonality_prior_scale = dd[2][0]
    seasonality_prior_scale_no = dd[2][1]

    seasonality_mode = dd[3][0]
    seasonality_mode_no = dd[3][1]

    growth = dd[4][0]
    growth_no = dd[4][1]

    yearly_seasonality = dd[5][0]
    yearly_seasonality_no = dd[5][1]

    #Build, fit and Predict
    data = cdata[cdata['location'].isin([nation])].sort_values(by="date")[['date',"new_cases_per_million"]]
    data.columns = ['ds','y']
    model = Prophet(changepoint_prior_scale= changepoint_prior_scale_no, 
                    changepoint_range= changepoint_range_no, 
                    seasonality_prior_scale= seasonality_prior_scale_no, 
                    seasonality_mode= seasonality_mode_no, 
                    growth= growth_no, 
                    yearly_seasonality= yearly_seasonality_no)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(data)
    future = model.make_future_dataframe(periods=period, freq='D')
    pred = model.predict(future)
    model.plot(pred);

    st.write(f'Forecast plot for {period} Days')
    fig1 = plot_plotly(model, pred)
    fig1.update_layout(plot_bgcolor = "lightgrey")
    st.plotly_chart(fig1,use_container_width=True)

#----------------------------------------------------------------------------------------------------------------------------------#
#LSTM Model
#----------------------------------------------------------------------------------------------------------------------------------#
def LST(nation):
    #Getting Data with respect to countries
    n_years = st.slider('Months of prediction:', max_value=8, min_value=1,value=5)
    period = n_years * 30
    st.info("Note: For a flat forecast....move the slider to improve the accuracy of forecast")
    country = nation
    ndata = cdata
    ndata['Date'] = pd.to_datetime(ndata['date'])
    df2 = ndata.copy()
    df2.set_index('Date', inplace= True)
    tdata = df2[df2['location'].isin([country])]
    tdata.dropna(inplace= True)
    ndata2 = tdata[['new_cases_per_million']]

    values = ndata2.values
    values = values.reshape((len(values),1))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    X = scaler.inverse_transform(normalized)
    
    #Splitting data into 70:30 ratios
    split_percent = 0.7
    split = int(split_percent*len(X))

    X_train = X[:split]
    X_test = X[split:]

    date_train = ndata2.index[:split]
    date_test = ndata2.index[split:]

    #training data by looking back at 18 month of data
    look_back = 18
    train_gen = TimeseriesGenerator(X_train, X_train, length= look_back, batch_size= 64)
    test_gen = TimeseriesGenerator(X_test, X_test, length= look_back, batch_size= 64)
   
    
    # Set layers for LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(85, activation= 'relu', input_shape= (look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer= 'Adam', loss= 'mse')
    num_epochs = 500
    
    def EPC(train,epoc):
        hist = lstm_model.fit(train,epochs=epoc,verbose=2)
        return hist
    history = EPC(train_gen,num_epochs)
    loss_train = history.history['loss']
    epochs = range(1,num_epochs+1)
    prediction = lstm_model.predict(test_gen)

    X = X.reshape((-1))

    
    def predict(num_prediction, model):
        prediction_list = X[-look_back:]
    
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
        
        return prediction_list

    # create future dates for x-axis    
    def predict_dates(num_prediction):
        last_date = ndata2.index.values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates
 
    # num_prediction = number of days into future
    num_prediction = period
    forecast = predict(num_prediction, lstm_model)
    forecast_dates = predict_dates(num_prediction)

    # plot data
    trace1 = go.Scatter(
        x = ndata2.index,
        y = X,
        mode = 'lines',
        name = 'Original Data'
    )
    trace2 = go.Scatter(
        x = forecast_dates,
        y = forecast,
        mode = 'lines',
        name = 'Forecasted Data'
    )
    layout = go.Layout(
        title = 'LSTM ' +country+ ' Forecast',
        xaxis = {'title' : 'Date'},
        yaxis = {'title' : 'new cases'}
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout,)
    fig.update_layout(
    autosize=False,
    width=1000,
    height=500,
    margin=dict(l=50,r=50,b=100,t=100,pad=4),
    paper_bgcolor="LightSteelBlue",
    )
    st.plotly_chart(fig,use_container_width=True)

#----------------------------------------------------------------------------------------------------------------------------------#
#ARIMA Model
#----------------------------------------------------------------------------------------------------------------------------------#

def ARMA(nation):
    n_years = st.slider('Months of prediction:', 1, 2)
    period = n_years * 30

    ndata = cdata
    ndata.head(2)
    ndata['Date'] = pd.to_datetime(ndata['date'])
    df2 = ndata.copy()
    df2.set_index('Date', inplace= True)
    ndata2 = df2[df2['location'].isin([nation])]
    ndata2.dropna(inplace= True)
    ndata2.drop(['new_people_vaccinated_smoothed_per_hundred','new_people_vaccinated_smoothed','iso_code','location','date','total_cases','new_cases','new_cases_smoothed','total_deaths','new_deaths','new_deaths_smoothed','total_cases_per_million','new_cases_smoothed_per_million','total_deaths_per_million','new_deaths_per_million','new_deaths_smoothed_per_million','new_tests','total_tests','total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed','new_tests_smoothed_per_thousand','positive_rate','tests_per_case','tests_units','total_vaccinations','people_vaccinated','people_fully_vaccinated','total_boosters','new_vaccinations','new_vaccinations_smoothed','total_vaccinations_per_hundred','people_vaccinated_per_hundred','people_fully_vaccinated_per_hundred','total_boosters_per_hundred','new_vaccinations_smoothed_per_million','stringency_index','population','median_age','aged_65_older','aged_70_older','life_expectancy'],axis=1,inplace= True)

    fdata = ndata2

    def models(fdata1):
        model = pm.auto_arima(fdata1, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=7,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
        return model


    smodel = models(fdata)

    n_periods = period - 20
    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(fdata.index[-1], periods = n_periods, freq='W')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)


    fig = plt.figure(figsize = (5, 5))
   
    # Plot
    plt.plot(fdata)
    plt.plot(ndata2.new_cases_per_million,color='blue')
    plt.plot(fitted_series, color='Orange')
    plt.title('Upcoming COVID-19 wave in '+ nation, fontsize=30,color='white')
    
    st.plotly_chart(fig,use_container_width=True)

#----------------------------------------------------------------------------------------------------------------------------------#
#Calling Models
#----------------------------------------------------------------------------------------------------------------------------------#
with st.spinner('Loading the predictions...please wait.....this may take a while'):
    if (add_selectbox=='LSTM'):
        try:
            LST(selected_nation)
        except:
            st.error("Something went wrong while building the Model")
        
    if(add_selectbox=='FBProphet'):
        try:
            FBProphet_forecast(selected_nation)
        except:
            st.error("Something went wrong while building the Model")

    if(add_selectbox=='Retuned_FBProphet'):
        try:
            Final_FBProphe_Forecast(selected_nation)   
        except:
            st.error("Something went wrong while building the Model")
        
    if(add_selectbox=='ARIMA'):
        try:
            ARMA(selected_nation)  
        except:
            st.error("Something went wrong while building the Model")

    if(add_selectbox=='None'):
            Data_analysis(selected_nation)  
            
st.success('Success!')

#----------------------------------------------------------------------------------------------------------------------------------#
