import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose




BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "rawData.csv"
df = pd.read_csv(DATA_PATH)


#Basic data cleaning and formatting

#parse the date
df["Week"] = pd.to_datetime(df["Week"], format="mixed")

#set date as index
df=df.set_index("Week")
df = df.sort_index()

#set frequency by standardizing monthly
df = df.resample("M").mean()

#forward fill as safety net
df = df.ffill()

New_Path = DATA_PATH = BASE_DIR / "data" / "processedData.csv"
df.to_csv(New_Path, index=True)


#perform basic EDA to discover data
df2 = pd.read_csv(New_Path, index_col = 0, parse_dates=True)


#visualize raw and processsed data for trends and seasonality
rawFig = px.line(df2,x=df.index, y = "GIC Rates", title = "Search Index For GIC Rates (RAW)")
rawFig.write_image(BASE_DIR /'figures'/'rawDataFig.png')

processedFig = px.line(df2,x=df.index, y = "GIC Rates", title = "Search Index For GIC Rates (PROCESSED)")
processedFig.write_image(BASE_DIR /'figures'/'processedDataFig.png')


#rolling average to confirm trend/structure change using 6 months window

window = 6
rollingMean = df2.rolling(window=window).mean()
rollingFig = px.line(rollingMean, x=rollingMean.index, y="GIC Rates", title=f"{window}-Month Rolling Mean of GIC Rates Search Index")
rollingFig.write_image(BASE_DIR /'figures'/'rollingMeanFig.png')

#rollwing variance (std) to confirm the stationarity of the data
rollingStd = df2.rolling(window=window).std()
rollingStdFig=px.line(rollingStd, x=rollingStd.index, y="GIC Rates", title=f"{window}-Month Rolling Standard Deviation of GIC Rates Search Index")
rollingStdFig.write_image(BASE_DIR /'figures'/'rollingStdFig.png')


#confirm absence of seasonality using decomposition
decomp =seasonal_decompose(df2, model ='additive', period=12) #addtive since no proportional seasonality observed
decompFig = go.Figure()

#now add the traces for observed, trend, seasonal and residual
decompFig.add_trace(go.Scatter(x= df2.index,y=decomp.observed, name='Observed'))
decompFig.add_trace(go.Scatter(x= df2.index,y=decomp.trend, name='Trend'))
decompFig.add_trace(go.Scatter(x= df2.index,y=decomp.seasonal, name='Seasonal'))
decompFig.add_trace(go.Scatter(x= df2.index,y=decomp.resid, name='Residual'))

decompFig.update_layout(title="Seasonal Decomposition")
decompFig.write_image(BASE_DIR / 'figures' / 'decompositionFig.png')
