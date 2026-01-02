import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processedData.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

#split train and test 
horizon = 12 

train = df.iloc[:-horizon] # 0 to last 12 months
test = df.iloc[-horizon:] #start at last 12 and stop at the end 


#train ETS model
ets_model = ETSModel(train['GIC Rates'], error='add', seasonal_periods=12) #automatic selection of trend and seasonality
ets_fit = ets_model.fit()

#forecast
ets_forecast= ets_fit.forecast(steps=horizon)
test['ETS_Forecast'] = ets_forecast.values #put forecasted values on test df under ETS_Forecast for evaluation later on 


#perform full forecast on entire data
y_full= df["GIC Rates"]

ETS_full_model =ETSModel(y_full, error='add', seasonal_periods=12)
ETS_full_fit = ETS_full_model.fit()

future_steps = 12
ETS_future_forecast = ETS_full_fit.forecast(steps=future_steps) 


#plot the forecast

fig = go.Figure()

fig.add_trace(go.Scatter(
    x= train.index,
    y= train['GIC Rates'],
    mode='lines',
    name='Train Data'
))

fig.add_trace(go.Scatter(
    x= test.index,
    y= test['GIC Rates'],
    mode='lines',
    name='Test Data',
    line = dict(color= 'black')
))

fig.add_trace(go.Scatter(x=test.index,
                        y = test['ETS_Forecast'],
                        mode='lines',
                        name='ETS Forecast',
                        line=dict(dash="dash", color="blue")
)
)

fig.update_layout(
    title ="ETS Model On GIC Rates Searches Using 12 Month Horizon",
    xaxis_title="Year",
    yaxis_title="GIC Rates Search Index"
)

fig.write_image(BASE_DIR / 'figures' / 'etsForecastFig.png')


#test to see what model captured
print("Trend:",ets_fit.model.trend)
print("Seasonanality:",ets_fit.model.seasonal)

#metric table row for ETS model
mae = mean_absolute_error(test['GIC Rates'], test['ETS_Forecast'])
rmse = np.sqrt(mean_squared_error(test['GIC Rates'], test['ETS_Forecast']))


metrics_row = pd.DataFrame({
    'Model':["ETS"],
    'MAE':[mae],
    'RMSE':[rmse]
})

print(metrics_row)