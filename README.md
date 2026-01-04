# Modeling Risk-Averse Financial Behavior Using Public Search Data

This project analyzes and forecasts population-level **risk-averse financial behavior** using Google Trends search interest in **Guaranteed Investment Certificates (GICs)** in Canada. Search interest in *“GIC rates”* is treated as a proxy for conservative investment intent, and multiple classical time-series models are evaluated over a **12-month forecast horizon**.

👉 **Full case study (business framing + technical deep dive):**  
https://medium.com/@harshetasharma5/using-time-series-models-to-forecast-risk-averse-financial-behavior-465fd3a25112

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
```text
---

## 1. Project Overview
