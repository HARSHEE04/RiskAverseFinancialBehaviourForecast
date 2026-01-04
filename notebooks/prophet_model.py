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
train_prophet = train.reset_index().rename(columns={"Week":"ds", "GIC Rates":"y"})
train_prophet.sort_values(by = "ds", inplace=True) #ensure sorted by date

#fit the prophet model using default parameters since we arent sure of seasonality/trend yet and dont want to force any
prophet_model = Prophet()
prophet_model.fit(train_prophet)

#forecast
future = prophet_model.make_future_dataframe(periods=horizon, freq='M')
forecast = prophet_model.predict(future)

#prophet makes predictions for entire dataframe including train, extract only test period
prophet_forecast = forecast.tail(horizon)[["ds","yhat"]]

#now merge with test set for evaluation
test_reset = test.reset_index()

test_reset.columns = ["ds", "GIC Rates"] #rename for merging
#merge on ds 
merged=pd.merge(test_reset, prophet_forecast, on="ds", how="inner")

#rename the yhat for clarity
merged.rename(columns={"yhat":"Prophet_Forecast"}, inplace=True)

#evaluation metrics
mae = mean_absolute_error(merged['GIC Rates'], merged['Prophet_Forecast'])
rmse = np.sqrt(mean_squared_error(merged['GIC Rates'], merged['Prophet_Forecast']))


metrics_row = pd.DataFrame({
    'Model':["Prophet"],
    'MAE':[mae],
    'RMSE':[rmse]
})

print("_____________________________")
print("Prophet Model Performance on Test Set")
print(metrics_row)

