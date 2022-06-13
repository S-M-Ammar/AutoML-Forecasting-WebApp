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
from deploymentModels import nbeats_v1_model_full , nbeats_v2_model_full

Queue = [] # for the threading purpose

def mape(actual,pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

def clean_parse_data(df,forecasting_col,date_col):
    try:
        df = df[df[forecasting_col].notna()]
        df = df[df[date_col].notna()]
        df[forecasting_col]=df[forecasting_col].astype('float')
        df = df[[date_col,forecasting_col]]
        df[date_col] = pd.to_datetime(df[date_col])
        return df
        # start_date = DateProcessing[date_column_name].iat[0]
        # end_date = df[date_col].iat[-1]
    except Exception as e:
        print("Here we go 1")
        print(e)
        return None

def kalman_filter(df,forecasting_col,date_col,start_date,type):
    try:
        copy = df.copy()
        dates=None
        if(type=='day'):
            dates = pd.date_range(start=str(start_date), periods=len(df[forecasting_col]))
        elif(type=='week'):
            dates = pd.date_range(start=str(start_date), periods=len(df[forecasting_col]),freq="W")
        elif(type=='month'):
            dates = pd.date_range(start=str(start_date), periods=len(df[forecasting_col]),freq="M")
        elif(type=='year'):
            dates = pd.date_range(start=str(start_date), periods=len(df[forecasting_col]),freq="Y")
        
        copy[date_col] = dates[0:len(df[forecasting_col])]
        copy.set_index(date_col,inplace=True)
        target = copy[forecasting_col]
        target = TimeSeries.from_series(target,fill_missing_dates=True, freq=None)
        KF = KalmanFilter()
        KF.fit(target)
        clean = KF.filter(target)
        clean = clean.pd_dataframe()
        df_new = pd.DataFrame()
        df_new[forecasting_col] = clean[forecasting_col].values
        df_new[date_col] = dates[0:len(df[forecasting_col])]
        return df_new
    except Exception as e:
        print("freq error")
        print(e)
        return None


    

def nbeats_v1_model(df,forecasting_col):
    print("Nbeats_v1 Stars....")
    target = df
    limit = int((80/100) * len(target))

    train = target.iloc[0:limit]
    val = target.iloc[limit:]

    train = TimeSeries.from_dataframe(train)
    val = TimeSeries.from_dataframe(val)

    model_nbeats = NBEATSModel(
    input_chunk_length=2,
    output_chunk_length=1,
    generic_architecture=True,
    num_stacks=10,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    n_epochs=50,
    nr_epochs_val_period=1,
    batch_size=2,
    model_name="nbeats_interpretable_run",
    )

    model_nbeats.fit(series=train, val_series=val, verbose=True)

    future = pd.DataFrame()
    future[forecasting_col] = df[forecasting_col].to_list()[0:limit]
    
    start_limit = len(future)

    end_limit = len(df)
    end_limit_flag = False
    if(end_limit>14):
        end_limit_flag = True
        end_limit = start_limit + 14

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
    val_array = np.absolute(val.univariate_values())
    if(end_limit_flag==True):
        val_array = val_array[0:14]
    error = mean_absolute_error(pred_array,val_array)
    Queue.append(("n_beats_v1",error))
    print("Nbeats_v1 Ends....")


def nbeats_v2_model(df,forecasting_col):
    print("Nbeats_v2 Starts....")
    target = df
    limit = int((80/100) * len(target))

    train = target.iloc[0:limit]
    val = target.iloc[limit:]

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
    n_epochs=50,
    nr_epochs_val_period=1,
    batch_size=5,
    model_name="nbeats_interpretable_run",
    )

    model_nbeats.fit(series=train, val_series=val, verbose=True)

    future = pd.DataFrame()
    future[forecasting_col] = df[forecasting_col].to_list()[0:limit]
    
    start_limit = len(future)

    end_limit = len(df)
    end_limit_flag = False
    if(end_limit>14):
        end_limit_flag = True
        end_limit = start_limit + 14

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
    val_array = np.absolute(val.univariate_values())
    if(end_limit_flag==True):
        val_array = val_array[0:14]
    error = mean_absolute_error(pred_array,val_array)
    Queue.append(("n_beats_v2",error))
    print("Nbeats_v2 Ends....")

def xgboost_model(df,forecasting_col,date_col):

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

    df['dayofweek'] = scaler_0.fit_transform(df[['dayofweek']])
    df['quarter'] = scaler_1.fit_transform(df[['quarter']])
    df['month'] = scaler_2.fit_transform(df[['month']])
    df['year'] = scaler_3.fit_transform(df[['year']])
    df['dayofyear'] = scaler_4.fit_transform(df[['dayofyear']])
    df['dayofmonth'] = scaler_5.fit_transform(df[['dayofmonth']])
    df['weekofyear'] = scaler_6.fit_transform(df[['weekofyear']])
    df['week']  =  scaler_7.fit_transform(df[['week']])
    df['weekday'] = scaler_8.fit_transform(df[['weekday']])
    df['daysInMonth'] = scaler_8.fit_transform(df[['daysInMonth']])
    df['is_quater_start'] = scaler_9.fit_transform(df[['is_quater_start']])
    df['is_quater_end'] = scaler_10.fit_transform(df[['is_quater_end']])
    df['is_leap_year'] = scaler_11.fit_transform(df[['is_leap_year']])

    limit = int((85/100) * len(df[forecasting_col]))

    test_date = df[[date_col]]

    train_X_data = df[['dayofweek','dayofyear','dayofmonth','weekofyear','week','weekday','daysInMonth','is_leap_year']].iloc[0:limit,:]
    train_Y_data = df[[forecasting_col]].iloc[0:limit,:]

    test_X_data =  df[['dayofweek','dayofyear','dayofmonth','weekofyear','week','weekday','daysInMonth','is_leap_year']].iloc[limit:,:]
    test_Y_data =  df[[forecasting_col]].iloc[limit:,:]

    reg.fit(train_X_data, train_Y_data,
        eval_set=[(train_X_data, train_Y_data)],
        early_stopping_rounds=1000,
        verbose=False)

    preds= reg.predict(test_X_data)
    true_ = test_Y_data[forecasting_col]
    error = mean_absolute_error(preds,true_)

    Queue.append(("xgboost",error))
    print("Xgboost Ends....")

def fourth_model():

    pass


def start(df,forecasting_col,date_col,type,future_units):
    try:
        df = clean_parse_data(df,forecasting_col,date_col)
        start_date = df[date_col].iat[0]
        new_df = kalman_filter(df,forecasting_col,date_col,start_date,type)
        end_date = df[date_col].iat[-1] 
        new_df.set_index(date_col,inplace=True)
       
        # imply threading here...

        # t1 = threading.Thread(target=xgboost_model, args=(new_df.copy(),forecasting_col,date_col,))
        # t2 = threading.Thread(target=nbeats_v2_model, args=(new_df.copy(),forecasting_col))
        # t3 = threading.Thread(target=nbeats_v1_model, args=(new_df.copy(),forecasting_col,))
        # t1.start()
        # t2.start()
        # t1.join()
        # t2.join()
        # t3.start()
        # t3.join()
        print("---------------------------------------------------")
        
        # winner_candidate = None
        # for x in Queue:
        #     print(x)

        prediction = nbeats_v2_model_full(new_df,forecasting_col,future_units)
        future_dates = None
        if(type=='day'):
            future_dates = pd.date_range(start=str(end_date), periods=future_units)
        elif(type=='week'):
            future_dates = pd.date_range(start=str(end_date), periods=future_units,freq="W")
        elif(type=='month'):
            future_dates = pd.date_range(start=str(end_date), periods=future_units,freq="M")
        elif(type=='year'):
            future_dates = pd.date_range(start=str(end_date), periods=future_units,freq="Y")
        
        future_dates_str = [str(x) for x in future_dates]
        print(prediction)
        print(future_dates_str)
        return prediction , future_dates_str

    except Exception as e:
        print(e)
        return [] , []





# model_nbeats = NBEATSModel(
#     input_chunk_length=15,
#     output_chunk_length=7,
#     generic_architecture=False,
#     num_blocks=3,
#     num_layers=4,
#     layer_widths=512,
#     n_epochs=50,
#     nr_epochs_val_period=1,
#     batch_size=2,
#     model_name="nbeats_interpretable_run",
# )