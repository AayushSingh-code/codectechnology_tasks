import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Download historical stock data
# ---------------------------------------------------------

TICKER = "AAPL"       # Change to any supported stock ticker
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True
)

if data.empty:
    raise ValueError("No stock data was downloaded.")

# Handle MultiIndex columns returned by some yfinance versions
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.dropna()

print("Downloaded data:")
print(data.head())


# ---------------------------------------------------------
# 2. Create prediction features
# ---------------------------------------------------------

# Previous day's closing price
data["Prev_Close"] = data["Close"].shift(1)

# Previous 5-day closing price
data["Prev_5_Close"] = data["Close"].shift(5)

# Moving averages
data["MA_5"] = data["Close"].rolling(window=5).mean()
data["MA_20"] = data["Close"].rolling(window=20).mean()

# Daily return
data["Return"] = data["Close"].pct_change()

data = data.dropna()


features = [
    "Prev_Close",
    "Prev_5_Close",
    "MA_5",
    "MA_20",
    "Return"
]

X = data[features]
y = data["Close"]


# ---------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------

# IMPORTANT:
# For time-series data, don't randomly shuffle the data.

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# ---------------------------------------------------------
# 4. Train Linear Regression model
# ---------------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ---------------------------------------------------------
# 5. Make predictions
# ---------------------------------------------------------

predictions = model.predict(X_test)


# ---------------------------------------------------------
# 6. Evaluate model
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-----------------")
print(f"MAE  : ${mae:.2f}")
print(f"RMSE : ${rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ---------------------------------------------------------
# 7. Plot actual vs predicted prices
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.index,
    y_test.values,
    label="Actual Price",
    color="blue"
)

plt.plot(
    y_test.index,
    predictions,
    label="Predicted Price",
    color="red"
)

plt.title(f"{TICKER} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 8. Predict the next trading day's price
# ---------------------------------------------------------

latest = data.iloc[-1]

latest_features = pd.DataFrame([{
    "Prev_Close": latest["Close"],
    "Prev_5_Close": data["Close"].iloc[-5],
    "MA_5": data["Close"].iloc[-5:].mean(),
    "MA_20": data["Close"].iloc[-20:].mean(),
    "Return": latest["Return"]
}])

next_price = model.predict(latest_features)[0]

print("\nNext Trading Day Prediction")
print("---------------------------")
print(f"Current price : ${latest['Close']:.2f}")
print(f"Predicted     : ${next_price:.2f}")
requirements.txt
numpy
pandas
matplotlib
scikit-learn
yfinance
