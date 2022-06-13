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
    limit = len(target)

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

    model_nbeats.fit(series=train, verbose=False)

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
    n_epochs=50,
    nr_epochs_val_period=1,
    batch_size=5,
    model_name="nbeats_interpretable_run",
    )

    model_nbeats.fit(series=train, verbose=False)

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

