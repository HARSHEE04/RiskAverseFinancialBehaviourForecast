# Modeling Risk-Averse Financial Behavior Using Public Search Data

This project analyzes and forecasts population-level **risk-averse financial behavior** using Google Trends search interest in **Guaranteed Investment Certificates (GICs)** in Canada. Search interest in *“GIC rates”* is treated as a proxy for conservative investment intent, and multiple classical time-series models are evaluated over a **12-month forecast horizon**.

👉 **Full case study (business framing + technical deep dive):**  
[Using Time-Series Models to Forecast Risk-Averse Financial Behavior](https://medium.com/@harshetasharma5/using-time-series-models-to-forecast-risk-averse-financial-behavior-465fd3a25112)

---

## 1. Project Overview

- **Objective:** Forecast short-term changes in risk-averse financial behavior using public search data as a behavioral signal.  
- **Scope:**
  - Five years of Google Trends data for *“GIC rates”* in Canada  
  - Weekly data aggregated to monthly frequency  
  - Univariate time-series forecasting  
  - Emphasis on model comparison, interpretability, and robustness  
- **Models evaluated:**
  - Naive baseline (persistence model)  
  - Exponential Smoothing (ETS, non-seasonal)  
  - ARIMA(1,1,1)  
  - Prophet  

Based on out-of-sample evaluation, **non-seasonal ETS** produced the lowest forecasting error and was selected for final forecasting.

---

## 2. Repository Structure

```text
.
├─ data/
│  ├─ rawData.csv
│  └─ processedData.csv
│
├─ figures/
│  ├─ rawDataFig.png
│  ├─ processedDataFig.png
│  ├─ rollingMeanFig.png
│  ├─ rollingStdFig.png
│  ├─ decompositionFig.png
│  ├─ naiveForecastFig.png
│  ├─ etsForecastFig.png
│  ├─ acfPlot.png
│  └─ pacfPlot.png
│
├─ notebooks/
│  ├─ EDA_TimeSeries.py
│  ├─ baseline_model.py
│  ├─ ets_model.py
│  ├─ arima_model.py
│  └─ prophet_model.py
│
├─ src/
├─ requirements.txt
└─ README.md

```
## 3. Installation and Setup
Clone the repository

bash
git clone https://github.com/HARSHEE04/RiskAverseFinancialBehaviourForecast.git
cd RiskAverseFinancialBehaviourForecast
Create and activate a virtual environment (recommended)

bash
python -m venv .venv
Windows

bash
.venv\Scripts\activate
macOS / Linux

bash
source .venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt

## 4. How To Run The Analysis
The project is organized as separate Python scripts, each corresponding to a stage of the time-series workflow.

Recommended execution order

bash
python notebooks/EDA_TimeSeries.py
python notebooks/baseline_model.py
python notebooks/ets_model.py
python notebooks/arima_model.py
python notebooks/prophet_model.py
Each script:

Loads the processed dataset

Trains the corresponding model

Computes evaluation metrics

Saves output figures to the figures/ directory


## 5. Model Evaluation Summary

Models were evaluated using a time-based train–test split, with the final 12 months held out for testing.

| Model                       | MAE   | RMSE  |
|-----------------------------|-------|-------|
| Naive Baseline              | 3.57  | 3.93  |
| Exponential Smoothing (ETS) | 3.46  | 3.83  |
| ARIMA (1,1,1)               | 3.58  | 3.94  |
| Prophet                     | 23.20 | 23.95 |

**Key findings**

- Exponential Smoothing (ETS) outperformed all alternatives.  
- ARIMA did not improve upon the naive baseline.  
- Prophet performed poorly due to weak seasonality and an abrupt macroeconomic regime shift.  



## 6. Tech Stack
Language: Python

Data & Analysis: pandas, numpy

Visualization: plotly, matplotlib

Time-Series Modeling: statsmodels (ETS, ARIMA), prophet

Evaluation: scikit-learn (MAE, RMSE)
