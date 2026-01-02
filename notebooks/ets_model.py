import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


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