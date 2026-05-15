import pandas as pd
import numpy as np
import yfinance as yf
from ta.volatility import AverageTrueRange

# =========================
# STOCK LIST
# =========================

stocks = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS"
]

# =========================
# SUPER TREND FUNCTION
# =========================

def supertrend(df, period=11, multiplier=2):

    atr = AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=period
    ).average_true_range()

    hl2 = (df['High'] + df['Low']) / 2

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = [True] * len(df)

    for i in range(1, len(df)):

        if df['Close'].iloc[i] > upperband.iloc[i - 1]:
            supertrend[i] = True

        elif df['Close'].iloc[i] < lowerband.iloc[i - 1]:
            supertrend[i] = False

        else:
            supertrend[i] = supertrend[i - 1]

            if supertrend[i] and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]

            if not supertrend[i] and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

    df['Supertrend'] = np.where(supertrend, lowerband, upperband)

    return df

# =========================
# SCANNER
# =========================

buy_signals = []
sell_signals = []

for stock in stocks:

    try:

        print(f"\nChecking {stock}")

        df = yf.download(
            stock,
            interval="1h",
            period="5d",
            progress=False,
            auto_adjust=True
        )

        if df.empty:
            print("No data")
            continue

        print(df.tail(3))

        df = supertrend(df)

        close = df['Close']
        st = df['Supertrend']

        print(f"Previous Close: {close.iloc[-2]}")
        print(f"Previous ST   : {st.iloc[-2]}")

        print(f"Current Close : {close.iloc[-1]}")
        print(f"Current ST    : {st.iloc[-1]}")

        # BUY crossover
        bull = (
            close.iloc[-2] < st.iloc[-2]
            and close.iloc[-1] > st.iloc[-1]
        )

        # SELL crossunder
        bear = (
            close.iloc[-2] > st.iloc[-2]
            and close.iloc[-1] < st.iloc[-1]
        )

        if bull:
            print(f"BUY SIGNAL on {stock}")
            buy_signals.append(stock)

        elif bear:
            print(f"SELL SIGNAL on {stock}")
            sell_signals.append(stock)

        else:
            print("No signal")

    except Exception as e:
        print(f"Error on {stock}: {e}")

# =========================
# FINAL RESULTS
# =========================

print("\n====================")
print("FINAL RESULTS")
print("====================")

print("\nBUY SIGNALS:")
print(buy_signals)

print("\nSELL SIGNALS:")
print(sell_signals)