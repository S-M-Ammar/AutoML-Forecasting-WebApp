from tkinter.tix import InputOnly
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
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score

Queue = []

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

# def train_test_split(df):
#     try:
#         #using 80% for test
#         total_length = len(df)
#         train_limit = int((80/100)*total_length)
#         train = df.iloc[0:train_limit,:]
#         test = df.iloc[train_limit:,:]
#         return train,test
#     except Exception as e:
#         print("1234.....")
#         print(e)
    

def nbeats_v1_model(df,forecasting_col):
    target = df
    limit = int((85/100) * len(target))

    train = target.iloc[0:limit]
    val = target.iloc[limit:]

    

    train = TimeSeries.from_dataframe(train)
    val = TimeSeries.from_dataframe(val)
    series = TimeSeries.from_dataframe(target)

    model_nbeats = NBEATSModel(
    input_chunk_length=3,
    output_chunk_length=1,
    generic_architecture=True,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    n_epochs=10,
    nr_epochs_val_period=1,
    batch_size=5,
    model_name="nbeats_interpretable_run",
    )

    model_nbeats.fit(series=train, val_series=val, verbose=True)

    future_Total = [0 for x in range(0,len(val))]
    future = pd.DataFrame()
    future[forecasting_col] = future_Total
    limit_before_append = len(df[forecasting_col])
    x = df[forecasting_col].append(future[forecasting_col],ignore_index=True)
    future_series = TimeSeries.from_series(x)

    pred_series = model_nbeats.historical_forecasts(
    future_series,
    start=limit_before_append,
    retrain=False,
    verbose=True,
    )
    print("Mape : ----------------------")
    print(len(pred_series))
    print(len(val))
    print(mape(pred_series,val))
    print("\n\n")

def nbeats_v2_model():
    pass

def xgboost_model():
    pass



def start(df,forecasting_col,date_col,type,future_units):
    try:
        df = clean_parse_data(df,forecasting_col,date_col)
        start_date = df[date_col].iat[0]
        new_df = kalman_filter(df,forecasting_col,date_col,start_date,type)
        # end_date = df[date_col].iat[-1] 
        new_df.set_index(date_col,inplace=True)
        nbeats_v1_model(new_df,forecasting_col)

        return "good"

    except Exception as e:
        print(e)
        return None

