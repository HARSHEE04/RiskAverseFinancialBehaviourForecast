import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processedData.csv"


df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
df.sort_index(inplace=True) #ensure dates sorted first for proper split


#split train and test 
horizon = 12 

train = df.iloc[:-horizon] # 0 to last 12 months
test = df.iloc[-horizon:] #start at last 12 and stop at the end 

#naive forecast model (persistence model) where forecast is last observed value
test['Naive_Forecast'] = train['GIC Rates'].iloc[-1] #last value of GIC Rates is our forecast for test set

#visualize train, test and forecast
fig = go.Figure()

#train trace
fig.add_trace(go.Scatter(
    x=train.index,
    y=train['GIC Rates'],
    mode='lines',
    name='Train Data'
))


#test data
fig.add_trace(go.Scatter(
    x= test.index,
    y= test['GIC Rates'],
    mode='lines',
    name='Test Data',
    line = dict(color= 'black')
))

#forecast trace
fig.add_trace(go.Scatter(x=test.index,
                        y = test['Naive_Forecast'],
                        mode='lines',
                        name='Naive Forecast',
                        line=dict(dash="dash", color="red")
)
)

#vertical lines for train and test
fig.add_vline(x=train.index[0], line_dash="dot", line_color="black")


#final layout
fig.update_layout(
    title ="Naive Forecaset Model On GIC Rates Searches Using 12 Month Horizon",
    xaxis_title="Date",
    yaxis_title="GIC Rates Search Index",
    template="plotly_white",
    legend = dict(x=0.01, y=0.99)
)


fig.write_image(BASE_DIR / 'figures' / 'naiveForecastFig.png')


mae = mean_absolute_error(test['GIC Rates'], test['Naive_Forecast'])
rmse = np.sqrt(mean_squared_error(test['GIC Rates'], test['Naive_Forecast']))


metrics_row = pd.DataFrame({
    'Model':["Naive"],
    'MAE':[mae],
    'RMSE':[rmse]
})

print(metrics_row)