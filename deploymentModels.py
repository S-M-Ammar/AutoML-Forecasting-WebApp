import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from darts import TimeSeries
from darts.models import KalmanFilter
from darts.utils import timeseries_generation as tg
from darts.metrics import mape
from darts.models import NBEATSModel
from sklearn.metrics import r2_score
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def nbeats_v1_model_full(df,forecasting_col,future_limit):
    print("Nbeats_v1 Stars....")
    target = df

    train = target
    val = target

    train = TimeSeries.from_dataframe(train)
    val = TimeSeries.from_dataframe(val)

    model_nbeats = NBEATSModel(
    input_chunk_length=10,
    output_chunk_length=5,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    n_epochs=50,
    nr_epochs_val_period=1,
    batch_size=5,
    model_name="nbeats_interpretable_run",
    )

    model_nbeats.fit(series=train, verbose=True)

    future = pd.DataFrame()
    future[forecasting_col] = df[forecasting_col].to_list()

    
    start_limit = len(future)
    end_limit = start_limit + future_limit

    remaning_values = [0 for x in range(start_limit,end_limit)]
    remaning_data_frame = pd.DataFrame()
    remaning_data_frame[forecasting_col] = remaning_values

    future = future.append(remaning_data_frame,ignore_index=True)

    future_series = TimeSeries.from_series(future)

    pred_series = model_nbeats.historical_forecasts(
    future_series,
    start=start_limit,
    retrain=False,
    verbose=False,)

    pred_array = np.absolute(pred_series.univariate_values())
    return pred_array
  


def nbeats_v2_model_full(df,forecasting_col,future_limit):
    print("Nbeats_v1 Stars....")
    target = df
    limit = len(target)

    train = target
    val = target

    train = TimeSeries.from_dataframe(train)
    val = TimeSeries.from_dataframe(val)

    model_nbeats = NBEATSModel(
    input_chunk_length=5,
    output_chunk_length=2,
    generic_architecture=False,
    num_stacks=10,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    n_epochs=10,
    nr_epochs_val_period=1,
    batch_size=5,
    model_name="nbeats_interpretable_run",
    )

    model_nbeats.fit(series=train, verbose=True)

    future = pd.DataFrame()
    future[forecasting_col] = df[forecasting_col].to_list()

    
    start_limit = len(future)
    end_limit = start_limit + future_limit

    remaning_values = [0 for x in range(start_limit,end_limit)]
    remaning_data_frame = pd.DataFrame()
    remaning_data_frame[forecasting_col] = remaning_values

    future = future.append(remaning_data_frame,ignore_index=True)

    future_series = TimeSeries.from_series(future)

    pred_series = model_nbeats.historical_forecasts(
    future_series,
    start=start_limit,
    retrain=False,
    verbose=False,)

    pred_array = np.absolute(pred_series.univariate_values())
    return pred_array


def xgboost_model_full(df,forecasting_col,date_col,future_limit,type,end_date):

    print("Xgboost Starts....")
    reg = xgb.XGBRegressor(n_estimators=10000)
    df[forecasting_col] = df[forecasting_col].astype(float)
    df.reset_index(inplace=True)
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['dayofmonth'] = df[date_col].dt.day
    df['weekofyear'] = df[date_col].dt.weekofyear
    df['week']  =  df[date_col].dt.week
    df['weekday'] = df[date_col].dt.weekday
    df['daysInMonth'] = df[date_col].dt.days_in_month
    df['is_quater_start'] = (df[date_col].dt.is_quarter_start).astype('float')
    df['is_quater_end'] =(df[date_col].dt.is_quarter_end).astype('float')
    df['is_leap_year'] = (df[date_col].dt.is_leap_year).astype('float')

    scaler_0 = MinMaxScaler()
    scaler_1 = MinMaxScaler()
    scaler_2 = MinMaxScaler()
    scaler_3 = MinMaxScaler()
    scaler_4 = MinMaxScaler()
    scaler_5 = MinMaxScaler()
    scaler_6 = MinMaxScaler()
    scaler_7 = MinMaxScaler()
    scaler_8 = MinMaxScaler()
    scaler_9 = MinMaxScaler()
    scaler_10 = MinMaxScaler()
    scaler_11 = MinMaxScaler()
    scaler_12 = MinMaxScaler()

    df['dayofweek'] = scaler_0.fit_transform(df[['dayofweek']])
    df['quarter'] = scaler_1.fit_transform(df[['quarter']])
    df['month'] = scaler_2.fit_transform(df[['month']])
    df['year'] = scaler_3.fit_transform(df[['year']])
    df['dayofyear'] = scaler_4.fit_transform(df[['dayofyear']])
    df['dayofmonth'] = scaler_5.fit_transform(df[['dayofmonth']])
    df['weekofyear'] = scaler_6.fit_transform(df[['weekofyear']])
    df['week']  =  scaler_7.fit_transform(df[['week']])
    df['weekday'] = scaler_8.fit_transform(df[['weekday']])
    df['daysInMonth'] = scaler_9.fit_transform(df[['daysInMonth']])
    df['is_quater_start'] = scaler_10.fit_transform(df[['is_quater_start']])
    df['is_quater_end'] = scaler_11.fit_transform(df[['is_quater_end']])
    df['is_leap_year'] = scaler_12.fit_transform(df[['is_leap_year']])

    limit = len(df[forecasting_col])

    train_X_data = df[['dayofweek','dayofyear','dayofmonth','weekofyear','week','weekday','daysInMonth','is_leap_year']].iloc[0:limit,:]
    train_Y_data = df[[forecasting_col]].iloc[0:limit,:]


    reg.fit(train_X_data, train_Y_data,
        eval_set=[(train_X_data, train_Y_data)],
        early_stopping_rounds=1000,
        verbose=False)


    future_dates = None
    if(type=='day'):
        future_dates = pd.date_range(start=str(end_date), periods=future_limit)
    elif(type=='week'):
        future_dates = pd.date_range(start=str(end_date), periods=future_limit,freq="W")
    elif(type=='month'):
        future_dates = pd.date_range(start=str(end_date), periods=future_limit,freq="M")
    elif(type=='year'):
        future_dates = pd.date_range(start=str(end_date), periods=future_limit,freq="Y")
    
    test_df = pd.DataFrame()
    test_df[date_col] = future_dates

    test_df['dayofweek'] = test_df[date_col].dt.dayofweek
    test_df['quarter'] = test_df[date_col].dt.quarter
    test_df['month'] = test_df[date_col].dt.month
    test_df['year'] = test_df[date_col].dt.year
    test_df['dayofyear'] = test_df[date_col].dt.dayofyear
    test_df['dayofmonth'] = test_df[date_col].dt.day
    test_df['weekofyear'] = test_df[date_col].dt.weekofyear
    test_df['week']  =  test_df[date_col].dt.week
    test_df['weekday'] = test_df[date_col].dt.weekday
    test_df['daysInMonth'] = test_df[date_col].dt.days_in_month
    test_df['is_quater_start'] = (test_df[date_col].dt.is_quarter_start).astype('float')
    test_df['is_quater_end'] =(test_df[date_col].dt.is_quarter_end).astype('float')
    test_df['is_leap_year'] = (test_df[date_col].dt.is_leap_year).astype('float')


    test_df['dayofweek'] = scaler_0.transform(test_df[['dayofweek']])
    test_df['quarter'] = scaler_1.transform(test_df[['quarter']])
    test_df['month'] = scaler_2.transform(test_df[['month']])
    test_df['year'] = scaler_3.transform(test_df[['year']])
    test_df['dayofyear'] = scaler_4.transform(test_df[['dayofyear']])
    test_df['dayofmonth'] = scaler_5.transform(test_df[['dayofmonth']])
    test_df['weekofyear'] = scaler_6.transform(test_df[['weekofyear']])
    test_df['week']  =  scaler_7.transform(test_df[['week']])
    test_df['weekday'] = scaler_8.transform(test_df[['weekday']])
    test_df['daysInMonth'] = scaler_9.transform(test_df[['daysInMonth']])
    test_df['is_quater_start'] = scaler_10.transform(test_df[['is_quater_start']])
    test_df['is_quater_end'] = scaler_11.transform(test_df[['is_quater_end']])
    test_df['is_leap_year'] = scaler_12.transform(test_df[['is_leap_year']])

    test_df = test_df[['dayofweek','dayofyear','dayofmonth','weekofyear','week','weekday','daysInMonth','is_leap_year']]
    preds= reg.predict(test_df)
    return preds
