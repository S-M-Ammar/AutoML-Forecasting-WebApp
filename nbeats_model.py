# import numpy as np
# import pandas as pd
# from darts.metrics import mape
# from darts import TimeSeries
# from darts.models import NBEATSModel
# from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
# from darts.metrics import mape, r2_score

# def perform_training_and_prediction(df,forecasting_column,date_column):

#     df[date_column] = pd.to_datetime(df[date_column])
#     # df.drop(labels=['Date'],axis=1,inplace=True)
#     target = df[forecasting_column]
    
#     train_length = (80/100)*len(df)
#     train = target.iloc[0:train_length]
#     val = target.iloc[train_length:]
    
#     train = TimeSeries.from_dataframe(train)
#     val = TimeSeries.from_dataframe(val)
#     series = TimeSeries.from_dataframe(target)

#     model_nbeats = NBEATSModel(
#     input_chunk_length=15,
#     output_chunk_length=7,
#     generic_architecture=True,
#     num_blocks=3,
#     num_layers=4,
#     layer_widths=512,
#     n_epochs=10,
#     nr_epochs_val_period=1,
#     batch_size=5,
#     model_name="nbeats_interpretable_run",
#     )
    
#     model_nbeats.fit(series=train, val_series=val, verbose=False)

#     pass
