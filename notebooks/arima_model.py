import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa import arima_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller,kpss



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processedData.csv"

df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

horizon = 12 

train = df.iloc[:-horizon] # 0 to last 12 months
test = df.iloc[-horizon:] #start at last 12 and stop at the end 

#stationarity Check Using ADF Unit Root Test (Diagnositic so can be run on entire df)
result = adfuller(df["GIC Rates"])
print("_____________________________")
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
print("_____________________________")

# p-value now 0.07 which is borderline stationary, run KPSS to confirm later if needed
kpss_stat, p_value, lags, crit_vals = kpss(train_log_diff, regression='c') # c for level stationarity

print("_____________________________")
print("KPSS Check on Transformed Train Data")
print("KPSS Statistic:", kpss_stat)
print("p-value:", p_value)

#p- value is 0.1 which is > 0.05, thus series is confirmed to be stationary, no further transformations needed

#___________________________________________________________________________________
# p,q,d selection using ACF and PACF plots

#compute ACF and PACF values since plotly doesnt have built in functions

max_lag = 24 

acf_values = acf(train_log_diff, nlags=max_lag)
pacf_values = pacf(train_log_diff, nlags=max_lag, method="ywm")

lags =list(range(len(acf_values)))

#plot acf 

acf_fig = go.Figure()

acf_fig.add_trace(go.Bar(
    x=lags
    , y=acf_values,
    name='ACF'
))

acf_fig.update_layout(
    title="ACF Plot (Log-Differenced Series)",
    xaxis_title="Lag",
    yaxis_title="Autocorrelation",
    showlegend=False
)

acf_fig.write_image(BASE_DIR / 'figures' / 'acfPlot.png')
#plot pacf

fig_pacf = go.Figure()  

fig_pacf.add_trace(go.Bar(
    x=lags
    , y=pacf_values,
    name='PACF'
))

fig_pacf.update_layout(
    title="PACF Plot (Log-Differenced Series)",
    xaxis_title="Lag",
    yaxis_title="Partial Autocorrelation",
    showlegend=False
)


fig_pacf.write_image(BASE_DIR / 'figures' / 'pacfPlot.png')
