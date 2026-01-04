import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processedData.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


#split train and test 
horizon = 12 

train = df.iloc[:-horizon] # 0 to last 12 months
test = df.iloc[-horizon:] #start at last 12 and stop at the end 

#rename columns for prophet compatibility
train.sort_values(inplace=True) #ensure sorted by date
train_prophet = train.reset_index().rename(columns={"Week":"ds", "GIC Rates":"y"})


#fit the prophet model using default parameters since we arent sure of seasonality/trend yet and dont want to force any
prophet_model = Prophet()
prophet_model.fit(train_prophet)

#forecast

