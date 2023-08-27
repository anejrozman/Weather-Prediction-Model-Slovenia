'''
analysis.py is a script of functions for data analysis and model training that
are used in the notebook results.ipynb.

@Author: Anej Rozman
''' 

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


####################### FILLING IN NAN VALUES ###########################

# Forward fill (not many NaN values)
# Back fill in case of starting NaN
def fill_na(train_data, test_data):
    for i in list(train_data):
        train_data[i] = train_data[i].ffill()
        if train_data[i].isna().any().any():
            train_data[i] = train_data[i].bfill()

    for i in list(test_data):
        test_data[i] = test_data[i].ffill()
        if test_data[i].isna().any().any():
            test_data[i] = test_data[i].bfill()
    return train_data, test_data

#################### TIME FEATURE #######################################

# TIME FEATURE (time index of when the data was measured)
# NG, LJ, MS, NM have data in 10 minute intervals
# PO, CE, JE have data in 30 minute intervals

# create indexes for train_data
def set_train_index(now, train_data, train_interval, yearly_interval, short_regions, long_regions):
    for i in range(3):
        t = now - i * yearly_interval
        
        # create 10 min index 
        short_train_index = []
        current_datetime = t - train_interval
        while current_datetime < t + dt.timedelta(days=1):
            short_train_index.append(current_datetime)
            current_datetime += dt.timedelta(minutes=10)
        short_train_index = pd.to_datetime(short_train_index)
        
        for j in short_regions:
            try:
                train_data[f'{j}{str(i)}'] = train_data[f'{j}{str(i)}'].set_index(short_train_index, drop=True)
            except KeyError:
                print(f'{j}{str(i)} not found, check if data was imported correctly')
        
        # create 30 min index 
        current_datetime = t - train_interval
        long_train_index = []
        while current_datetime < t + dt.timedelta(days=1):
            long_train_index.append(current_datetime)
            current_datetime += dt.timedelta(minutes=30)
        long_train_index = pd.to_datetime(long_train_index)
        
        for j in long_regions:
            try:
                train_data[f'{j}{str(i)}'] = train_data[f'{j}{str(i)}'].set_index(long_train_index, drop=True)
            except KeyError:
                print(f'{j}{str(i)} not found, check if data was imported correctly')
    
    return train_data

# create indexes for test_data
def set_test_index(test_data, short_regions, long_regions):
    current = dt.datetime.now() - dt.timedelta(minutes=8)  # speculation (data doesn't get updated instantly)

    # create 10 min index
    short_remainder = current.minute % 10
    short_now = current - dt.timedelta(minutes=short_remainder)
    short_now = short_now.replace(second=0, microsecond=0)

    short_test_index = []
    for i in range(len(test_data['NG'])):
        short_test_index.append(short_now - i * dt.timedelta(minutes=10))
    short_test_index = pd.to_datetime(short_test_index)

    for j in short_regions:
        try:
            test_data[j] = test_data[j].set_index(short_test_index, drop=True)
        except KeyError:
            print(f'{j} not found, check if data was imported correctly')

    # create 30 min index
    long_remainder = current.minute % 30
    long_now = current - dt.timedelta(minutes=long_remainder)
    long_now = long_now.replace(second=0, microsecond=0)

    long_test_index = []
    for i in range(len(test_data['PO'])):
        long_test_index.append(short_now - i * dt.timedelta(minutes=30))
    long_test_index = pd.to_datetime(long_test_index)

    for j in long_regions:
        try:
            test_data[j] = test_data[j].set_index(long_test_index, drop=True)
        except KeyError:
            print(f'{j} not found, check if data was imported correctly')

    return test_data

    
def hot_encode(train_data, test_data):
    for i in test_data:
        test_data[i] = test_data[i].sort_index(ascending=False)

    # list for hot encoding
    hours = [f"hour_{i}" for i in range(24)]

    # one hot encode the hour in train_data
    for i in train_data:
        train_data[i]['hour'] = train_data[i].index.hour
        train_data[i] = pd.get_dummies(train_data[i], 
                                       columns=['hour'], 
                                       prefix=['hour'])
        for j in hours:
            train_data[i][j] = train_data[i][j].replace(False, 0).replace(True, 1)

    # one hot encode the hour in test_data
    for i in test_data:
        test_data[i]['hour'] = test_data[i].index.hour
        test_data[i] = pd.get_dummies(test_data[i], 
                                      columns=['hour'], 
                                      prefix=['hour'])
        for j in hours:
            test_data[i][j] = test_data[i][j].replace(False, 0).replace(True, 1)
    return train_data, test_data


###################### LAG FEATURES #####################################

# LAG FEATURES (lagged values of temperature and moisture)

def create_lags(train_data, test_data, num_lags, columns):
    # create lag features for train_data
    for i in train_data:
        for lag in range(1, num_lags + 1):
            for j in columns:
                train_data[i][f'{j}_lag_{lag}'] = train_data[i][f'{j}'].shift(lag)
        train_data[i] = train_data[i].dropna()

    # create lag features for test_data
    for i in test_data:
        for lag in range(1, num_lags + 1):
            for j in columns:
                test_data[i][f'{j}_lag_{lag}'] = test_data[i][f'{j}'].shift(lag)
        test_data[i] = test_data[i].dropna()
    return train_data, test_data

##################### MODEL TRAINING ####################################

def train_models(train_data):
    # dict of models for each region
    all_models = {}

    for i in train_data:
        X = train_data[i].drop(['temp'], axis=1)
        y = train_data[i]['temp']
        model = LinearRegression()
        model.fit(X, y)
        all_models[i] = model
    return all_models


######################## MODEL TESTING ##################################

# test each of the models on test_data and choose based on MSE

def test_models(all_models, test_data):
    regions = ['NG', 'LJ', 'MS', 'NM', 'PO', 'CE', 'JE']
    predictions = {}
    models = {}

    for i in test_data:
        X = test_data[i].drop(['temp'], axis=1)
        y = test_data[i]['temp']
        
        for j in regions:
            for k in range(3):
                predictions[f'{j}{str(k)}'] = mean_squared_error(
                    all_models[f'{j}{str(k)}'].predict(X), 
                    y
                )
            
            best = 0
            for l in range(1, 3):
                if predictions[f'{j}{str(l)}'] < predictions[f'{j}{str(l-1)}']:
                    best = l
            
            models[j] = all_models[f'{j}{str(best)}']
    
    return models



############# FORECAST OF FUTUTRE CONDITIONS USING ARIMA ##################

# in the future

############# FUNCTIONS FOR JUPYTER NOTEBOOK VISUALIZATION ################

# function for predictions of models for one region
def compare_region_models(region, all_models, test_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_data[region].index,
        y=test_data[region]['temp'],
        name='Actual'
    ))
    for i in range(3):
        fig.add_trace(go.Scatter(
            x=test_data[region].index,
            y=all_models[f'{region}{str(i)}'].predict(
                test_data[region].drop(['temp'], axis=1)
                ),
            name=f'Predicted {str(i)}'
        ))
    fig.update_layout(
        title=f'Comparison of models for {region} region',
        xaxis_title='Time',
        yaxis_title='Temperature [°C]',
        legend_title='Models',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig.show()

# compute r^2 score for each region
def show_r2(all_models, test_data, regions):
    r2 = {}
    for i in test_data:
        X = test_data[i].drop(['temp'], axis=1)
        y = test_data[i]['temp']
        for j in regions:
            for k in range(3):
                r2[f'{j}{str(k)}'] = r2_score(
                    all_models[f'{j}{str(k)}'].predict(X), 
                    y
                    )
    return r2

# compute mean squared error for each region
def show_mse(all_models, test_data, regions):
    mse = {}
    for i in test_data:
        X = test_data[i].drop(['temp'], axis=1)
        y = test_data[i]['temp']
        for j in regions:
            for k in range(3):
                mse[f'{j}{str(k)}'] = mean_squared_error(
                    all_models[f'{j}{str(k)}'].predict(X), 
                    y
                    )
    return mse

# show correlation among variables for a region
def show_corr(train_data, region):
    correlation_matrix = train_data[region].corr()
    fig = px.imshow(correlation_matrix)
    fig.show()

# show graphs dependence of temperature on each variable
def show_dependence(train_data, region):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    fig.suptitle(f'Dependence of temperature on variables for {region} region', fontsize=16)

    variables = ['moisture', 'wind_speed', 'wind_dir', 'rain']
    colors = ['blue', 'green', 'red', 'purple']

    row, col = 0, 0
    for var, color in zip(variables, colors):
        # Sort the data by the variable's values
        sorted_indices = np.argsort(train_data[region][var])
        sorted_temp = train_data[region]['temp'].values[sorted_indices]

        axes[row, col].scatter(train_data[region][var].values[sorted_indices], sorted_temp, c=color)
        axes[row, col].set_xlabel(var)
        axes[row, col].set_ylabel('Temperature [°C]')
        axes[row, col].set_title(f'Temp vs. {var}')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

        col += 1
        if col > 1:
            col = 0
            row += 1
        

    plt.tight_layout()
    plt.show()


# visual graph of vse for each region
def mse_graph(mse, regions):
    fig = go.Figure()
    for i in regions:
        fig.add_trace(go.Bar(
            x=[2021,2022,2023],
            y=[mse[f'{i}0'], mse[f'{i}1'], mse[f'{i}2']],
            name=f'{i}'
        ))
    fig.update_layout(
        title='Mean Squared Error for each region',
        xaxis_title='Model year',
        yaxis_title='MSE',
    )
    fig.show()






    





