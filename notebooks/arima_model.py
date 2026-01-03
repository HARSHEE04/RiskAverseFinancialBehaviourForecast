import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.stattools import adfuller,kpss



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processedData.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

horizon = 12 

train = df.iloc[:-horizon] # 0 to last 12 months
test = df.iloc[-horizon:] #start at last 12 and stop at the end 

#stationarity Check Using ADF Unit Root Test (Diagnositic so can be run on entire df)
result = adfuller(df["GIC Rates"])
print("Diagonstic Stationarity Check on Entire Data")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

#p- value = 0.14 which is > 0.05 thus confirming non stationarity



#Model testing and evaluation   (Checking to model forecasts properly using train test split)

#make train stationary
#log first
train_log = np.log(train["GIC Rates"])

#difference to remove trend
train_log_diff = train_log.diff().dropna()

#recheck stationarity
adf_stat = adfuller(train_log_diff)
print("_____________________________")
print("Stationarity Check After Transformations on Train Data")
print('ADF Statistic: %f' % adf_stat[0])
print('p-value: %f' % adf_stat[1])

# p-value now 0.07 which is borderline stationary, run KPSS to confirm later if needed
kpss_stat, p_value, lags, crit_vals = kpss(train_log_diff, regression='c') # c for level stationarity

print("_____________________________")
print("KPSS Check on Transformed Train Data")
print("KPSS Statistic:", kpss_stat)
print("p-value:", p_value)