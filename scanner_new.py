import os
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf

from ta.volatility import AverageTrueRange

# =========================================================
# SETTINGS
# =========================================================

INTERVAL = "1h"
PERIOD = "5d"

SENSITIVITY = 2
ATR_PERIOD = 11

OUTPUT_FOLDER = "charts"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# STOCK LIST
# =========================================================

stocks = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "SBIN.NS",
    "BEL.NS",
    "ADANIPORTS.NS",
    "LT.NS",
    "ITC.NS",
]

# =========================================================
# TRADINGVIEW SUPERTREND
# =========================================================

def tradingview_supertrend(df, period=11, factor=2):

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=period
    ).average_true_range()

    hl2 = (df["High"] + df["Low"]) / 2

    upperband = hl2 + factor * atr
    lowerband = hl2 - factor * atr

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):

        # FINAL UPPER BAND
        if (
            upperband.iloc[i] < final_upperband.iloc[i - 1]
            or df["Close"].iloc[i - 1] > final_upperband.iloc[i - 1]
        ):
            final_upperband.iloc[i] = upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i - 1]

        # FINAL LOWER BAND
        if (
            lowerband.iloc[i] > final_lowerband.iloc[i - 1]
            or df["Close"].iloc[i - 1] < final_lowerband.iloc[i - 1]
        ):
            final_lowerband.iloc[i] = lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i - 1]

        # DIRECTION
        if i == 1:
            direction.iloc[i] = 1

        elif supertrend.iloc[i - 1] == final_upperband.iloc[i - 1]:

            if df["Close"].iloc[i] > final_upperband.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1

        else:

            if df["Close"].iloc[i] < final_lowerband.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

        # SUPERTREND VALUE
        if direction.iloc[i] == -1:
            supertrend.iloc[i] = final_lowerband.iloc[i]
        else:
            supertrend.iloc[i] = final_upperband.iloc[i]

    df["Supertrend"] = supertrend
    df["Direction"] = direction

    return df

# =========================================================
# MAIN SCANNER
# =========================================================

buy_signals = []
sell_signals = []

for stock in stocks:

    try:

        print("\n================================================")
        print(f"Checking {stock}")
        print("================================================")

        # =========================================================
        # DOWNLOAD DATA
        # =========================================================

        df = yf.download(
            stock,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        # Fix MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print("No data")
            continue

        # Keep only OHLCV
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        df.dropna(inplace=True)

        print(df.tail())

        # =========================================================
        # SUPERTREND
        # =========================================================

        df = tradingview_supertrend(
            df,
            period=ATR_PERIOD,
            factor=SENSITIVITY
        )

        # =========================================================
        # BUY / SELL SIGNALS
        # =========================================================

        df["BUY"] = (
            (df["Close"].shift(1) < df["Supertrend"].shift(1))
            &
            (df["Close"] > df["Supertrend"])
        )

        df["SELL"] = (
            (df["Close"].shift(1) > df["Supertrend"].shift(1))
            &
            (df["Close"] < df["Supertrend"])
        )

        # =========================================================
        # TREND
        # =========================================================

        df["TREND"] = np.where(
            df["Close"] > df["Supertrend"],
            "BULLISH",
            "BEARISH"
        )

        latest_trend = df["TREND"].iloc[-1]

        # =========================================================
        # GET LAST SIGNAL
        # =========================================================

        signals = df[
            (df["BUY"] == True) |
            (df["SELL"] == True)
        ]

        latest_signal = "NONE"
        signal_time = None

        if not signals.empty:

            last_row = signals.iloc[-1]

            if last_row["BUY"]:
                latest_signal = "BUY"
                buy_signals.append(stock)

            elif last_row["SELL"]:
                latest_signal = "SELL"
                sell_signals.append(stock)

            signal_time = signals.index[-1]

        # =========================================================
        # PRINT SIGNAL INFO
        # =========================================================

        print("\n---------------------------")
        print(f"Current Trend : {latest_trend}")
        print(f"Last Signal   : {latest_signal}")

        if signal_time is not None:
            print(f"Signal Time   : {signal_time}")

        print("---------------------------")

        # =========================================================
        # SHOW LAST 5 SIGNALS
        # =========================================================

        print("\nLAST 5 SIGNALS")

        if signals.empty:
            print("No signals found")

        else:

            for idx, row in signals.tail(5).iterrows():

                if row["BUY"]:
                    print(f"{idx} -> BUY")

                elif row["SELL"]:
                    print(f"{idx} -> SELL")

        # =========================================================
        # CHART PLOTS
        # =========================================================

        apds = []

        # Supertrend line
        apds.append(
            mpf.make_addplot(
                df["Supertrend"],
                color='orange',
                width=1.5
            )
        )

        # BUY markers
        buy_markers = np.where(
            df["BUY"],
            df["Low"] * 0.995,
            np.nan
        )

        apds.append(
            mpf.make_addplot(
                buy_markers,
                type='scatter',
                marker='^',
                markersize=120,
                color='lime'
            )
        )

        # SELL markers
        sell_markers = np.where(
            df["SELL"],
            df["High"] * 1.005,
            np.nan
        )

        apds.append(
            mpf.make_addplot(
                sell_markers,
                type='scatter',
                marker='v',
                markersize=120,
                color='red'
            )
        )

        # =========================================================
        # CANDLE COLORS
        # =========================================================

        mc = mpf.make_marketcolors(
            up='green',
            down='red',
            edge='inherit',
            wick='inherit',
            volume='inherit'
        )

        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            facecolor='black'
        )

        # =========================================================
        # SAVE CHART
        # =========================================================

        save_path = os.path.join(
            OUTPUT_FOLDER,
            f"{stock}.png"
        )

        mpf.plot(
            df,
            type='candle',
            style=style,
            title=f"{stock} | Trend: {latest_trend} | Last Signal: {latest_signal}",
            volume=True,
            figsize=(16, 9),
            addplot=apds,
            savefig=save_path
        )

        print(f"\nChart Saved -> {save_path}")

    except Exception as e:
        print(f"\nError in {stock}: {e}")

# =========================================================
# FINAL RESULTS
# =========================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

print("\nBUY SIGNAL STOCKS:")
print(buy_signals)

print("\nSELL SIGNAL STOCKS:")
print(sell_signals)