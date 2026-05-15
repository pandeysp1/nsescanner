import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf

warnings.filterwarnings("ignore")

# =========================================================
# SETTINGS  (mirror Pine Script inputs)
# =========================================================

INTERVAL   = "1h"
PERIOD     = "12d"   # 60d gives ~375 bars on NSE (1h); yfinance allows up to 730d for 1h

SENSITIVITY = 2        # sens
ATR_PERIOD  = 11       # supertrend atrLen

# Reversal signal (QQE) inputs
RSI_PERIOD  = 14
SF          = 5
KQE         = 4.238
THRESH_HOLD = 10

# Ichimoku-lite (diamond signal) inputs
CONVERSION_PERIODS    = 5
BASE_PERIODS          = 2
LAGGING_SPAN2_PERIODS = 5
DISPLACEMENT          = 6

OUTPUT_FOLDER = "charts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# NIFTY 250 STOCK LIST  (fetched live from NSE, hardcoded fallback)
# =========================================================

# Hardcoded Nifty LargeMidcap 250 constituents (as of May 2026).
# Used automatically if NSE fetch fails.
NIFTY250_FALLBACK = [
    # ── Nifty 50 ──────────────────────────────────────────────────────
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BEL.NS","BPCL.NS",
    "BHARTIARTL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","ITC.NS",
    "INDUSINDBK.NS","INFY.NS","JSWSTEEL.NS","JIOFIN.NS","KOTAKBANK.NS",
    "LT.NS","M&M.NS","MARUTI.NS","NTPC.NS","NESTLEIND.NS","ONGC.NS",
    "POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SHRIRAMFIN.NS","SBIN.NS",
    "SUNPHARMA.NS","TCS.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "TECHM.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS",
    # ── Nifty Next 50 ─────────────────────────────────────────────────
    "ADANIENSOL.NS","ADANIGREEN.NS","ADANIPOWER.NS","ATGL.NS","AMBUJACEM.NS",
    "DMART.NS","BANKBARODA.NS","BERGEPAINT.NS","BOSCHLTD.NS","CANBK.NS",
    "CHOLAFIN.NS","COLPAL.NS","DLF.NS","DABUR.NS","DIVISLAB.NS",
    "GODREJCP.NS","GODREJPROP.NS","HAVELLS.NS","HAL.NS","ICICIGI.NS",
    "ICICIPRULI.NS","IOC.NS","IRCTC.NS","IRFC.NS","INDUSTOWER.NS",
    "INDIGO.NS","JINDALSTEL.NS","LICI.NS","LTIM.NS","LTTS.NS",
    "MAXHEALTH.NS","MARICO.NS","MUTHOOTFIN.NS","NHPC.NS","NAUKRI.NS",
    "OFSS.NS","OIL.NS","PIIND.NS","PAGEIND.NS","PIDILITIND.NS",
    "PFC.NS","RECLTD.NS","SRF.NS","MOTHERSON.NS","SHREECEM.NS",
    "SIEMENS.NS","TATAPOWER.NS","TORNTPHARM.NS","TVSMOTOR.NS","ZOMATO.NS",
    # ── Nifty Midcap 150 ──────────────────────────────────────────────
    "ABCAPITAL.NS","ABFRL.NS","ALKEM.NS","APLLTD.NS","ASTRAL.NS",
    "AUROPHARMA.NS","AVANTIFEED.NS","BAJAJHLDNG.NS","BALKRISIND.NS","BANDHANBNK.NS",
    "BATAINDIA.NS","BHEL.NS","BIOCON.NS","BLUEDART.NS","BSOFT.NS",
    "CAMS.NS","CANFINHOME.NS","CARBORUNIV.NS","CASTROLIND.NS","CESC.NS",
    "CGPOWER.NS","CHAMBLFERT.NS","COFORGE.NS","CONCOR.NS","CRISIL.NS",
    "CROMPTON.NS","CUB.NS","CUMMINSIND.NS","DALBHARAT.NS","DEEPAKNTR.NS",
    "DELTACORP.NS","PERSISTENT.NS","DIXON.NS","EDELWEISS.NS","ELGIEQUIP.NS",
    "EMAMILTD.NS","ENDURANCE.NS","ESCORTS.NS","EXIDEIND.NS","FEDERALBNK.NS",
    "FINPIPE.NS","FORCEMOT.NS","FORTIS.NS","GAIL.NS","GLENMARK.NS",
    "GMRINFRA.NS","GPPL.NS","GRANULES.NS","GSPL.NS","HAPPSTMNDS.NS",
    "HFCL.NS","HONAUT.NS","IDFCFIRSTB.NS","IEX.NS","IIFL.NS",
    "INDHOTEL.NS","INDIAMART.NS","INDIANB.NS","INDPAINT.NS","INTELLECT.NS",
    "ISEC.NS","JKCEMENT.NS","JSL.NS","JUBLFOOD.NS","KALYANKJIL.NS",
    "KANSAINER.NS","KEI.NS","KIMS.NS","KIOCL.NS","KNRCON.NS",
    "KRBL.NS","KSB.NS","LAURUSLABS.NS","LICHSGFIN.NS","LINDEINDIA.NS",
    "LUPIN.NS","LUXIND.NS","MAHINDCIE.NS","MAHLIFE.NS","MANAPPURAM.NS",
    "MASFIN.NS","METROPOLIS.NS","MFSL.NS","MINDTREE.NS","MRF.NS",
    "NATIONALUM.NS","NAVINFLUOR.NS","NBCC.NS","NCC.NS","NIACL.NS",
    "NLCINDIA.NS","NMDC.NS","NUVOCO.NS","OBEROIRLTY.NS","OFSS.NS",
    "ORIENTELEC.NS","PATANJALI.NS","PCBL.NS","PEL.NS","PETRONET.NS",
    "PFIZER.NS","PHOENIXLTD.NS","POLYCAB.NS","POWERMECH.NS","PRAJIND.NS",
    "PRINCEPIPE.NS","PNBHOUSING.NS","RADICO.NS","RAILTEL.NS","RAJESHEXPO.NS",
    "RAMCOCEM.NS","RITES.NS","RVNL.NS","SAFARI.NS","SAIL.NS",
    "SAPPHIRE.NS","SCHAEFFLER.NS","SOLARINDS.NS","SONACOMS.NS","STAR.NS",
    "STARHEALTH.NS","SUNTV.NS","SUPREMEIND.NS","SYNGENE.NS","TANLA.NS",
    "TATACHEM.NS","TATACOMM.NS","TATAELXSI.NS","TCNSBRANDS.NS","TEAMLEASE.NS",
    "THERMAX.NS","TIINDIA.NS","TIMKEN.NS","TITAGARH.NS","TORNTPOWER.NS",
    "TRENT.NS","TRIDENT.NS","UBL.NS","UJJIVANSFB.NS","UNIONBANK.NS",
    "UPL.NS","UTIAMC.NS","VAIBHAVGBL.NS","VARDHACRLC.NS","VBL.NS",
    "VEDL.NS","VINATIORGA.NS","VOLTAS.NS","WHIRLPOOL.NS","WIPRO.NS",
    "YESBANK.NS","ZEEL.NS","ZENSARTECH.NS","ZYDUSLIFE.NS",
]


def fetch_nifty250() -> list:
    """
    Tries to download the live Nifty LargeMidcap 250 constituent list from NSE.
    Falls back to the hardcoded list if the download fails for any reason.
    """
    import requests

    url = (
        "https://archives.nseindia.com/content/indices/"
        "ind_niftylargemidcap250list.csv"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        from io import StringIO
        df_idx = pd.read_csv(StringIO(resp.text))
        # NSE CSV has a column called "Symbol"
        symbols = df_idx["Symbol"].dropna().str.strip().tolist()
        tickers = [s + ".NS" for s in symbols]
        print(f"  ✔  Fetched {len(tickers)} stocks live from NSE.")
        return tickers
    except Exception as exc:
        print(f"  ⚠  NSE fetch failed ({exc}). Using hardcoded list ({len(NIFTY250_FALLBACK)} stocks).")
        return NIFTY250_FALLBACK


print("\nFetching Nifty 250 constituent list...")
stocks = fetch_nifty250()
print(f"  Total stocks to scan: {len(stocks)}\n")

# =========================================================
# CORE MATH HELPERS
# =========================================================

def rma(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's Moving Average — exactly what Pine Script ta.rma() / ta.atr() uses.
    alpha = 1 / period,  adjust=False  →  same as SMMA / RMA.
    """
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR using Wilder's RMA — matches Pine Script ta.atr()."""
    return rma(true_range(df), period)

# =========================================================
# SUPERTREND  (matches Pine Script supertrend() function)
#
# KEY FIX:  Pine Script passes `close` as _src,  NOT hl2.
#           upperBand = close + factor * atr
#           lowerBand = close - factor * atr
# =========================================================

def calc_supertrend(df: pd.DataFrame, period: int = 11, factor: float = 2.0) -> pd.DataFrame:
    atr_vals = calc_atr(df, period)
    src = df["Close"]                       # ← Pine uses close, not hl2

    upper_raw = src + factor * atr_vals
    lower_raw = src - factor * atr_vals

    # We'll build arrays for speed, then assign back to df
    n             = len(df)
    final_upper   = upper_raw.values.copy().astype(float)
    final_lower   = lower_raw.values.copy().astype(float)
    supertrend    = np.full(n, np.nan)
    direction     = np.zeros(n, dtype=int)

    close_arr = df["Close"].values

    for i in range(1, n):
        # ── Final upper band ──────────────────────────────────────────
        # Pine: upperBand < prevUpperBand OR close[1] > prevUpperBand  → use new value
        if upper_raw.iloc[i] < final_upper[i - 1] or close_arr[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_raw.iloc[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # ── Final lower band ──────────────────────────────────────────
        # Pine: lowerBand > prevLowerBand OR close[1] < prevLowerBand  → use new value
        if lower_raw.iloc[i] > final_lower[i - 1] or close_arr[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_raw.iloc[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # ── Direction ─────────────────────────────────────────────────
        # Pine: if na(atr[1]) → direction = 1
        if np.isnan(atr_vals.iloc[i - 1]):
            direction[i] = 1

        elif np.isnan(supertrend[i - 1]):
            # Still in warm-up region
            direction[i] = 1

        elif supertrend[i - 1] == final_upper[i - 1]:
            # Was on upper band (bearish): flip to bullish if close > upper
            direction[i] = -1 if close_arr[i] > final_upper[i] else 1

        else:
            # Was on lower band (bullish): flip to bearish if close < lower
            direction[i] = 1 if close_arr[i] < final_lower[i] else -1

        # ── Supertrend value ──────────────────────────────────────────
        # direction -1 → bullish → lower band
        # direction  1 → bearish → upper band
        supertrend[i] = final_lower[i] if direction[i] == -1 else final_upper[i]

    df = df.copy()
    df["Supertrend"] = supertrend
    df["Direction"]  = direction
    return df


# =========================================================
# BUY / SELL CROSSOVER SIGNALS
# =========================================================

def calc_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_close = df["Close"].shift(1)
    prev_st    = df["Supertrend"].shift(1)

    # bull = ta.crossover(close, supertrend)
    df["BUY"]  = (prev_close < prev_st) & (df["Close"] > df["Supertrend"])

    # bear = ta.crossunder(close, supertrend)
    df["SELL"] = (prev_close > prev_st) & (df["Close"] < df["Supertrend"])

    df["TREND"] = np.where(df["Close"] > df["Supertrend"], "BULLISH", "BEARISH")

    return df


# =========================================================
# DIAMOND SIGNALS  (Ichimoku-lite breakout in Pine Script)
#
# donchian(len) = avg(lowest(len), highest(len))
# conversionLine = donchian(5)    baseLine = donchian(2)
# leadLine1 = avg(conversion, base)
# leadLine2 = donchian(5)         displaced by 6 bars back
#
# breakup:  lead2 > lead1  AND  green candle  AND  close crosses over lead2
# breakdn:  lead2 < lead1  AND  red candle    AND  close crosses under lead2
# =========================================================

def donchian(df: pd.DataFrame, period: int) -> pd.Series:
    return (df["High"].rolling(period).max() + df["Low"].rolling(period).min()) / 2


def calc_diamond_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conversion = donchian(df, CONVERSION_PERIODS)
    base       = donchian(df, BASE_PERIODS)
    lead1      = (conversion + base) / 2
    lead2      = donchian(df, LAGGING_SPAN2_PERIODS)

    # Displace by DISPLACEMENT-1 bars back (Pine: lead2 = leadLine2[displacement-1])
    d          = DISPLACEMENT - 1
    lead1_disp = lead1.shift(d)
    lead2_disp = lead2.shift(d)

    green_candle = df["Close"] > df["Open"]
    red_candle   = df["Close"] < df["Open"]

    crossup  = (df["Close"].shift(1) < lead2_disp.shift(1)) & (df["Close"] > lead2_disp)
    crossdn  = (df["Close"].shift(1) > lead2_disp.shift(1)) & (df["Close"] < lead2_disp)

    df["DIAMOND_BUY"]  = (lead2_disp > lead1_disp) & green_candle & crossup
    df["DIAMOND_SELL"] = (lead2_disp < lead1_disp) & red_candle   & crossdn

    return df


# =========================================================
# QQE REVERSAL SIGNALS  (Pine Script "reversal signals")
# =========================================================

def calc_qqe_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    src          = df["Close"]
    wilders_per  = RSI_PERIOD * 2 - 1

    rsi          = 100 - (100 / (1 + rma(
        src.diff().clip(lower=0),
        RSI_PERIOD
    ) / rma(
        (-src.diff()).clip(lower=0),
        RSI_PERIOD
    )))
    rsi_ma       = ema(rsi, SF)

    atr_rsi      = rsi_ma.diff().abs()
    ma_atr_rsi   = rma(atr_rsi, wilders_per)
    dar          = rma(ma_atr_rsi, wilders_per) * KQE

    longband  = pd.Series(0.0, index=df.index)
    shortband = pd.Series(0.0, index=df.index)
    trend_qqe = pd.Series(1,   index=df.index)

    for i in range(1, len(df)):
        new_long  = rsi_ma.iloc[i] - dar.iloc[i]
        new_short = rsi_ma.iloc[i] + dar.iloc[i]

        if rsi_ma.iloc[i - 1] > longband.iloc[i - 1] and rsi_ma.iloc[i] > longband.iloc[i - 1]:
            longband.iloc[i] = max(longband.iloc[i - 1], new_long)
        else:
            longband.iloc[i] = new_long

        if rsi_ma.iloc[i - 1] < shortband.iloc[i - 1] and rsi_ma.iloc[i] < shortband.iloc[i - 1]:
            shortband.iloc[i] = min(shortband.iloc[i - 1], new_short)
        else:
            shortband.iloc[i] = new_short

        cross_up   = longband.iloc[i - 1] < rsi_ma.iloc[i]   and longband.iloc[i - 1] >= rsi_ma.iloc[i - 1]
        cross_dn   = shortband.iloc[i - 1] > rsi_ma.iloc[i]  and shortband.iloc[i - 1] <= rsi_ma.iloc[i - 1]

        if cross_up:
            trend_qqe.iloc[i] = 1
        elif cross_dn:
            trend_qqe.iloc[i] = -1
        else:
            trend_qqe.iloc[i] = trend_qqe.iloc[i - 1]

    fast_tl = np.where(trend_qqe == 1, longband, shortband)
    fast_tl = pd.Series(fast_tl, index=df.index)

    # Exlong == 1 / Exshort == 1  → signal fires on the bar the streak starts
    above    = fast_tl < rsi_ma
    exlong   = above  & ~above.shift(1, fill_value=False)
    exshort  = ~above & above.shift(1, fill_value=False)

    df["QQE_BUY"]  = exlong
    df["QQE_SELL"] = exshort

    return df


# =========================================================
# CLOUD  (SMA7 vs SMA13 — used for colour in Pine Script)
# =========================================================

def calc_cloud(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CLOUD_FAST"] = sma(df["Close"], 7)
    df["CLOUD_SLOW"] = sma(df["Close"], 13)
    df["CLOUD_BULL"] = df["CLOUD_FAST"] > df["CLOUD_SLOW"]
    return df


# =========================================================
# MAIN SCANNER
# =========================================================

buy_signals     = []
sell_signals    = []
rev_buy         = []
rev_sell        = []
diamond_buy     = []
diamond_sell    = []

for stock in stocks:
    try:
        print("\n" + "=" * 52)
        print(f"  Checking: {stock}")
        print("=" * 52)

        # ── Download ──────────────────────────────────────────
        df = yf.download(
            stock,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print("  No data returned — skipping.")
            continue

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        print(f"  Rows downloaded   : {len(df)}")
        MIN_BARS = max(ATR_PERIOD, RSI_PERIOD) + 10
        if len(df) < MIN_BARS:
            print(f"  Not enough data (need {MIN_BARS} bars) — skipping.")
            continue

        # ── Indicators ───────────────────────────────────────
        df = calc_supertrend(df, period=ATR_PERIOD, factor=SENSITIVITY)
        df = calc_signals(df)
        df = calc_diamond_signals(df)
        df = calc_qqe_signals(df)
        df = calc_cloud(df)

        # ── Latest state ──────────────────────────────────────
        latest         = df.iloc[-1]
        latest_trend   = latest["TREND"]

        sig_df = df[(df["BUY"]) | (df["SELL"])]
        latest_signal  = "NONE"
        signal_time    = None

        if not sig_df.empty:
            last_sig = sig_df.iloc[-1]
            signal_time = sig_df.index[-1]
            if last_sig["BUY"]:
                latest_signal = "BUY"
                buy_signals.append(stock)
            else:
                latest_signal = "SELL"
                sell_signals.append(stock)

        # QQE reversal
        if latest["QQE_BUY"]:
            rev_buy.append(stock)
        if latest["QQE_SELL"]:
            rev_sell.append(stock)

        # Diamond
        if latest["DIAMOND_BUY"]:
            diamond_buy.append(stock)
        if latest["DIAMOND_SELL"]:
            diamond_sell.append(stock)

        # ── Print summary ─────────────────────────────────────
        print(f"  Current Trend   : {latest_trend}")
        print(f"  Last ST Signal  : {latest_signal}", end="")
        print(f"  @ {signal_time}" if signal_time else "")
        print(f"  QQE Reversal    : {'BUY' if latest['QQE_BUY'] else 'SELL' if latest['QQE_SELL'] else 'NONE'}")
        print(f"  Diamond Signal  : {'BUY' if latest['DIAMOND_BUY'] else 'SELL' if latest['DIAMOND_SELL'] else 'NONE'}")
        print(f"  Cloud           : {'BULLISH' if latest['CLOUD_BULL'] else 'BEARISH'}")

        print("\n  LAST 5 SUPERTREND SIGNALS:")
        if sig_df.empty:
            print("  (none)")
        else:
            for idx, row in sig_df.tail(5).iterrows():
                arrow = "▲ BUY" if row["BUY"] else "▼ SELL"
                print(f"  {idx}  →  {arrow}")

        # ── Chart ─────────────────────────────────────────────
        # Helper: only add a scatter series if it has at least one non-NaN value
        # (mplfinance crashes with "zero-size array" on all-NaN scatter series)
        def safe_scatter(markers, **kwargs):
            if np.any(~np.isnan(markers)):
                apds.append(mpf.make_addplot(markers, type="scatter", **kwargs))

        apds = []
        apds.append(mpf.make_addplot(df["Supertrend"], color="orange", width=1.5))

        buy_markers  = np.where(df["BUY"],  df["Low"]  * 0.995, np.nan)
        sell_markers = np.where(df["SELL"], df["High"] * 1.005, np.nan)
        safe_scatter(buy_markers,  marker="^", markersize=120, color="lime")
        safe_scatter(sell_markers, marker="v", markersize=120, color="red")

        # QQE reversal markers
        qqe_buy_m  = np.where(df["QQE_BUY"],  df["Low"]  * 0.990, np.nan)
        qqe_sell_m = np.where(df["QQE_SELL"], df["High"] * 1.010, np.nan)
        safe_scatter(qqe_buy_m,  marker="^", markersize=60, color="cyan")
        safe_scatter(qqe_sell_m, marker="v", markersize=60, color="magenta")

        # Diamond markers
        dia_buy_m  = np.where(df["DIAMOND_BUY"],  df["Low"]  * 0.993, np.nan)
        dia_sell_m = np.where(df["DIAMOND_SELL"], df["High"] * 1.007, np.nan)
        safe_scatter(dia_buy_m,  marker="D", markersize=50, color="cyan")
        safe_scatter(dia_sell_m, marker="D", markersize=50, color="pink")

        mc = mpf.make_marketcolors(
            up="green", down="red",
            edge="inherit", wick="inherit", volume="inherit"
        )
        style = mpf.make_mpf_style(
            marketcolors=mc, gridstyle="--", facecolor="black"
        )

        title_str = (
            f"{stock} | {latest_trend} | "
            f"ST:{latest_signal} | "
            f"QQE:{'B' if latest['QQE_BUY'] else 'S' if latest['QQE_SELL'] else '-'} | "
            f"Cloud:{'↑' if latest['CLOUD_BULL'] else '↓'}"
        )

        save_path = os.path.join(OUTPUT_FOLDER, f"{stock}.png")

        mpf.plot(
            df,
            type="candle",
            style=style,
            title=title_str,
            volume=True,
            figsize=(16, 9),
            addplot=apds,
            savefig=save_path,
        )

        print(f"\n  Chart saved → {save_path}")

    except Exception as exc:
        print(f"\n  ERROR in {stock}: {exc}")

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 52)
print("  FINAL SCAN RESULTS")
print("=" * 52)

print(f"\n  ▲  Supertrend BUY  : {buy_signals  or ['(none)']}")
print(f"  ▼  Supertrend SELL : {sell_signals or ['(none)']}")
print(f"\n  ▲  QQE Rev BUY     : {rev_buy      or ['(none)']}")
print(f"  ▼  QQE Rev SELL    : {rev_sell     or ['(none)']}")
print(f"\n  ◆  Diamond BUY     : {diamond_buy  or ['(none)']}")
print(f"  ◆  Diamond SELL    : {diamond_sell or ['(none)']}")
print()