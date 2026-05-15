import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

INTERVAL    = "1h"
PERIOD      = "60d"
SENSITIVITY = 2
ATR_PERIOD  = 11
RSI_PERIOD  = 14
SF          = 5
KQE         = 4.238
CONVERSION_PERIODS    = 5
BASE_PERIODS          = 2
LAGGING_SPAN2_PERIODS = 5
DISPLACEMENT          = 6
SR_LEFT  = 20
SR_RIGHT = 10
TOP_N    = 10
OUTPUT_FOLDER = "charts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

NIFTY250_FALLBACK = [
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
    "ABCAPITAL.NS","ABFRL.NS","ALKEM.NS","APLLTD.NS","ASTRAL.NS",
    "AUROPHARMA.NS","BAJAJHLDNG.NS","BALKRISIND.NS","BANDHANBNK.NS",
    "BATAINDIA.NS","BHEL.NS","BIOCON.NS","CAMS.NS","CANFINHOME.NS",
    "CGPOWER.NS","COFORGE.NS","CONCOR.NS","CRISIL.NS","CROMPTON.NS",
    "CUMMINSIND.NS","DALBHARAT.NS","DEEPAKNTR.NS","DIXON.NS","ELGIEQUIP.NS",
    "EMAMILTD.NS","ENDURANCE.NS","ESCORTS.NS","EXIDEIND.NS","FEDERALBNK.NS",
    "FORTIS.NS","GAIL.NS","GLENMARK.NS","GMRINFRA.NS","GRANULES.NS",
    "HAPPSTMNDS.NS","HFCL.NS","HONAUT.NS","IDFCFIRSTB.NS","IEX.NS",
    "INDHOTEL.NS","INDIAMART.NS","INDIANB.NS","INTELLECT.NS","ISEC.NS",
    "JKCEMENT.NS","JSL.NS","JUBLFOOD.NS","KALYANKJIL.NS","KANSAINER.NS",
    "KEI.NS","KIMS.NS","KNRCON.NS","LAURUSLABS.NS",
    "LICHSGFIN.NS","LINDEINDIA.NS","LUPIN.NS","LUXIND.NS","MANAPPURAM.NS",
    "METROPOLIS.NS","MFSL.NS","MRF.NS","NATIONALUM.NS",
    "NAVINFLUOR.NS","NBCC.NS","NCC.NS","NIACL.NS","NLCINDIA.NS",
    "NMDC.NS","OBEROIRLTY.NS","PCBL.NS","PETRONET.NS",
    "PFIZER.NS","PHOENIXLTD.NS","POLYCAB.NS","PNBHOUSING.NS","RADICO.NS",
    "RAILTEL.NS","RAMCOCEM.NS","RITES.NS","RVNL.NS","SAIL.NS",
    "SCHAEFFLER.NS","SOLARINDS.NS","SONACOMS.NS","STARHEALTH.NS","SUNTV.NS",
    "SUPREMEIND.NS","SYNGENE.NS","TATACHEM.NS","TATACOMM.NS","TATAELXSI.NS",
    "TEAMLEASE.NS","THERMAX.NS","TIINDIA.NS","TIMKEN.NS","TORNTPOWER.NS",
    "TRENT.NS","TRIDENT.NS","UBL.NS","UNIONBANK.NS","UPL.NS",
    "UTIAMC.NS","VBL.NS","VEDL.NS","VOLTAS.NS","YESBANK.NS","ZYDUSLIFE.NS",
]

def fetch_nifty250():
    import requests
    url = "https://archives.nseindia.com/content/indices/ind_niftylargemidcap250list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        from io import StringIO
        df_idx = pd.read_csv(StringIO(resp.text))
        symbols = df_idx["Symbol"].dropna().str.strip().tolist()
        tickers = [s + ".NS" for s in symbols]
        print(f"  Fetched {len(tickers)} stocks live from NSE.")
        return tickers
    except Exception as exc:
        print(f"  NSE fetch failed ({exc}). Using hardcoded list.")
        return NIFTY250_FALLBACK

def rma(series, period):
    return series.ewm(alpha=1.0/period, adjust=False).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period).mean()

def true_range(df):
    pc = df["Close"].shift(1)
    return pd.concat([df["High"]-df["Low"],
                      (df["High"]-pc).abs(),
                      (df["Low"]-pc).abs()], axis=1).max(axis=1)

def calc_atr(df, period):
    return rma(true_range(df), period)

def calc_supertrend(df, period=11, factor=2.0):
    atr_vals  = calc_atr(df, period)
    src       = df["Close"]
    upper_raw = src + factor * atr_vals
    lower_raw = src - factor * atr_vals
    n           = len(df)
    final_upper = upper_raw.values.copy().astype(float)
    final_lower = lower_raw.values.copy().astype(float)
    supertrend  = np.full(n, np.nan)
    direction   = np.zeros(n, dtype=int)
    close_arr   = df["Close"].values
    for i in range(1, n):
        if upper_raw.iloc[i] < final_upper[i-1] or close_arr[i-1] > final_upper[i-1]:
            final_upper[i] = upper_raw.iloc[i]
        else:
            final_upper[i] = final_upper[i-1]
        if lower_raw.iloc[i] > final_lower[i-1] or close_arr[i-1] < final_lower[i-1]:
            final_lower[i] = lower_raw.iloc[i]
        else:
            final_lower[i] = final_lower[i-1]
        if np.isnan(atr_vals.iloc[i-1]) or np.isnan(supertrend[i-1]):
            direction[i] = 1
        elif supertrend[i-1] == final_upper[i-1]:
            direction[i] = -1 if close_arr[i] > final_upper[i] else 1
        else:
            direction[i] = 1 if close_arr[i] < final_lower[i] else -1
        supertrend[i] = final_lower[i] if direction[i] == -1 else final_upper[i]
    df = df.copy()
    df["Supertrend"] = supertrend
    df["Direction"]  = direction
    return df

def calc_signals(df):
    df = df.copy()
    pc = df["Close"].shift(1)
    ps = df["Supertrend"].shift(1)
    df["BUY"]   = (pc < ps) & (df["Close"] > df["Supertrend"])
    df["SELL"]  = (pc > ps) & (df["Close"] < df["Supertrend"])
    df["TREND"] = np.where(df["Close"] > df["Supertrend"], "BULLISH", "BEARISH")
    return df

def calc_sr_levels(df, left=SR_LEFT, right=SR_RIGHT):
    close_now = df["Close"].iloc[-1]
    highs, lows = [], []
    for i in range(left, len(df) - right):
        if df["High"].iloc[i] == df["High"].iloc[i-left:i+right+1].max():
            highs.append(float(df["High"].iloc[i]))
        if df["Low"].iloc[i] == df["Low"].iloc[i-left:i+right+1].min():
            lows.append(float(df["Low"].iloc[i]))
    def dedup(levels):
        levels = sorted(set(levels))
        out = []
        for lvl in levels:
            if not out or abs(lvl - out[-1]) / max(out[-1], 1e-9) > 0.003:
                out.append(lvl)
        return out
    res_levels = dedup([h for h in highs if h > close_now])[:4]
    sup_levels = dedup([l for l in lows  if l < close_now])[-4:]
    return res_levels, sup_levels

def donchian(df, period):
    return (df["High"].rolling(period).max() + df["Low"].rolling(period).min()) / 2

def calc_diamond_signals(df):
    df   = df.copy()
    d    = DISPLACEMENT - 1
    conv = donchian(df, CONVERSION_PERIODS)
    base = donchian(df, BASE_PERIODS)
    lead1 = ((conv + base) / 2).shift(d)
    lead2 = donchian(df, LAGGING_SPAN2_PERIODS).shift(d)
    green  = df["Close"] > df["Open"]
    red    = df["Close"] < df["Open"]
    crossup = (df["Close"].shift(1) < lead2.shift(1)) & (df["Close"] > lead2)
    crossdn = (df["Close"].shift(1) > lead2.shift(1)) & (df["Close"] < lead2)
    df["DIAMOND_BUY"]  = (lead2 > lead1) & green & crossup
    df["DIAMOND_SELL"] = (lead2 < lead1) & red   & crossdn
    return df

def calc_qqe_signals(df):
    df          = df.copy()
    src         = df["Close"]
    wilders_per = RSI_PERIOD * 2 - 1
    up   = src.diff().clip(lower=0)
    down = (-src.diff()).clip(lower=0)
    rsi  = 100 - (100 / (1 + rma(up, RSI_PERIOD) / rma(down, RSI_PERIOD).replace(0, np.nan)))
    rsi_ma     = ema(rsi, SF)
    atr_rsi    = rsi_ma.diff().abs()
    ma_atr_rsi = rma(atr_rsi, wilders_per)
    dar        = rma(ma_atr_rsi, wilders_per) * KQE
    n = len(df)
    longband  = np.zeros(n)
    shortband = np.zeros(n)
    trend_q   = np.ones(n, dtype=int)
    rm = rsi_ma.values
    dv = dar.values
    for i in range(1, n):
        nl = rm[i] - dv[i]
        ns = rm[i] + dv[i]
        longband[i]  = max(longband[i-1], nl)  if rm[i-1] > longband[i-1]  and rm[i] > longband[i-1]  else nl
        shortband[i] = min(shortband[i-1], ns) if rm[i-1] < shortband[i-1] and rm[i] < shortband[i-1] else ns
        cup = longband[i-1] < rm[i] and longband[i-1] >= rm[i-1]
        cdn = shortband[i-1] > rm[i] and shortband[i-1] <= rm[i-1]
        trend_q[i] = 1 if cup else (-1 if cdn else trend_q[i-1])
    fast_tl = pd.Series(np.where(trend_q == 1, longband, shortband), index=df.index)
    above   = fast_tl < rsi_ma
    df["QQE_BUY"]  = above  & ~above.shift(1, fill_value=False)
    df["QQE_SELL"] = ~above & above.shift(1, fill_value=False)
    return df

def calc_cloud(df):
    df = df.copy()
    df["CLOUD_FAST"] = sma(df["Close"], 7)
    df["CLOUD_SLOW"] = sma(df["Close"], 13)
    df["CLOUD_BULL"] = df["CLOUD_FAST"] > df["CLOUD_SLOW"]
    return df

def conviction_score(latest_signal, latest, df, sig_df):
    score = 0
    breakdown = {}
    if latest_signal == "NONE":
        return 0, breakdown
    is_buy  = latest_signal == "BUY"

    # 1. Recency
    if not sig_df.empty:
        last_idx = df.index.get_loc(sig_df.index[-1])
        bars_ago = len(df) - 1 - last_idx
        if bars_ago == 0:
            score += 3; breakdown["recency"] = "Fired this bar (+3)"
        elif bars_ago <= 3:
            score += 2; breakdown["recency"] = f"Recent {bars_ago}b ago (+2)"
        elif bars_ago <= 8:
            score += 1; breakdown["recency"] = f"OK {bars_ago}b ago (+1)"
        else:
            breakdown["recency"] = f"Old {bars_ago}b ago (+0)"

    # 2. Cloud alignment
    cloud_ok = (is_buy and latest["CLOUD_BULL"]) or (not is_buy and not latest["CLOUD_BULL"])
    if cloud_ok:
        score += 2; breakdown["cloud"] = "Cloud aligns (+2)"
    else:
        breakdown["cloud"] = "Cloud against (+0)"

    # 3. QQE
    qqe_ok = (is_buy and latest["QQE_BUY"]) or (not is_buy and latest["QQE_SELL"])
    if qqe_ok:
        score += 2; breakdown["qqe"] = "QQE confirms (+2)"
    else:
        breakdown["qqe"] = "QQE neutral (+0)"

    # 4. Diamond
    dia_ok = (is_buy and latest["DIAMOND_BUY"]) or (not is_buy and latest["DIAMOND_SELL"])
    if dia_ok:
        score += 2; breakdown["diamond"] = "Diamond confirms (+2)"
    else:
        breakdown["diamond"] = "Diamond neutral (+0)"

    # 5. Clean trend
    total_sigs = len(sig_df)
    if total_sigs <= 2:
        score += 1; breakdown["clean"] = f"Clean ({total_sigs} flips) (+1)"
    else:
        breakdown["clean"] = f"{total_sigs} flips (+0)"

    return score, breakdown

def save_chart(df, stock, latest_trend, latest_signal, res_levels, sup_levels):
    def safe_scatter(markers, **kw):
        if np.any(~np.isnan(markers)):
            apds.append(mpf.make_addplot(markers, type="scatter", **kw))

    apds = []
    apds.append(mpf.make_addplot(df["Supertrend"], color="orange", width=1.5))

    buy_m  = np.where(df["BUY"],  df["Low"]  * 0.997, np.nan)
    sell_m = np.where(df["SELL"], df["High"] * 1.003, np.nan)
    safe_scatter(buy_m,  marker="^", markersize=100, color="lime")
    safe_scatter(sell_m, marker="v", markersize=100, color="red")

    qqe_buy_m  = np.where(df["QQE_BUY"],  df["Low"]  * 0.992, np.nan)
    qqe_sell_m = np.where(df["QQE_SELL"], df["High"] * 1.008, np.nan)
    safe_scatter(qqe_buy_m,  marker="^", markersize=55, color="cyan")
    safe_scatter(qqe_sell_m, marker="v", markersize=55, color="magenta")

    dia_buy_m  = np.where(df["DIAMOND_BUY"],  df["Low"]  * 0.994, np.nan)
    dia_sell_m = np.where(df["DIAMOND_SELL"], df["High"] * 1.006, np.nan)
    safe_scatter(dia_buy_m,  marker="D", markersize=45, color="cyan")
    safe_scatter(dia_sell_m, marker="D", markersize=45, color="pink")

    mc    = mpf.make_marketcolors(up="green", down="red",
                                   edge="inherit", wick="inherit", volume="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle="--", facecolor="black")

    title_str = (
        f"{stock} | {latest_trend} | ST:{latest_signal} | "
        f"QQE:{'B' if df['QQE_BUY'].iloc[-1] else 'S' if df['QQE_SELL'].iloc[-1] else '-'} | "
        f"Cloud:{'UP' if df['CLOUD_BULL'].iloc[-1] else 'DN'}"
    )

    fig, axes = mpf.plot(
        df, type="candle", style=style, title=title_str,
        volume=True, figsize=(18, 9), addplot=apds, returnfig=True,
    )
    ax = axes[0]

    # --- Buy/Sell text label annotations ---
    for i, (idx, row) in enumerate(df.iterrows()):
        if row["BUY"]:
            ax.annotate(
                "Buy",
                xy=(i, row["Low"] * 0.997), xytext=(i, row["Low"] * 0.988),
                fontsize=7, fontweight="bold", color="white", ha="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#0099BB",
                          edgecolor="white", linewidth=0.5, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="#0099BB", lw=0.8),
            )
        if row["SELL"]:
            ax.annotate(
                "Sell",
                xy=(i, row["High"] * 1.003), xytext=(i, row["High"] * 1.012),
                fontsize=7, fontweight="bold", color="white", ha="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#CC2244",
                          edgecolor="white", linewidth=0.5, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="#CC2244", lw=0.8),
            )

    # --- Support & Resistance zones ---
    for lvl in res_levels:
        ax.axhline(lvl, color="#E93E63", linewidth=1.2, linestyle="--", alpha=0.75)
        ax.axhspan(lvl * 0.998, lvl * 1.002, facecolor="#E93E63", alpha=0.12)
        ax.text(len(df) - 1, lvl, f"R {lvl:.1f}",
                color="#E93E63", fontsize=6.5, va="bottom", ha="right", alpha=0.9)

    for lvl in sup_levels:
        ax.axhline(lvl, color="#00DBFF", linewidth=1.2, linestyle="--", alpha=0.75)
        ax.axhspan(lvl * 0.998, lvl * 1.002, facecolor="#00DBFF", alpha=0.12)
        ax.text(len(df) - 1, lvl, f"S {lvl:.1f}",
                color="#00DBFF", fontsize=6.5, va="top", ha="right", alpha=0.9)

    save_path = os.path.join(OUTPUT_FOLDER, f"{stock}.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)
    return save_path

# =========================================================
# MAIN
# =========================================================

print("\nFetching Nifty 250 constituent list...")
stocks = fetch_nifty250()
print(f"  Total stocks to scan: {len(stocks)}\n")

buy_signals   = []
sell_signals  = []
rev_buy, rev_sell = [], []
diamond_buy, diamond_sell = [], []
conviction_records = []

for stock in stocks:
    try:
        print(f"\n{'='*52}\n  Checking: {stock}\n{'='*52}")

        df = yf.download(stock, period=PERIOD, interval=INTERVAL,
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print("  No data — skipping."); continue

        df = df[["Open","High","Low","Close","Volume"]].dropna()
        print(f"  Rows downloaded   : {len(df)}")

        MIN_BARS = max(ATR_PERIOD, RSI_PERIOD, SR_LEFT + SR_RIGHT) + 10
        if len(df) < MIN_BARS:
            print(f"  Not enough data (need {MIN_BARS}) — skipping."); continue

        df = calc_supertrend(df, period=ATR_PERIOD, factor=SENSITIVITY)
        df = calc_signals(df)
        df = calc_diamond_signals(df)
        df = calc_qqe_signals(df)
        df = calc_cloud(df)

        res_levels, sup_levels = calc_sr_levels(df)

        latest       = df.iloc[-1]
        latest_trend = latest["TREND"]

        sig_df        = df[(df["BUY"]) | (df["SELL"])]
        latest_signal = "NONE"
        signal_time   = None

        if not sig_df.empty:
            last_sig    = sig_df.iloc[-1]
            signal_time = sig_df.index[-1]
            if last_sig["BUY"]:
                latest_signal = "BUY";  buy_signals.append(stock)
            else:
                latest_signal = "SELL"; sell_signals.append(stock)

        if latest["QQE_BUY"]:   rev_buy.append(stock)
        if latest["QQE_SELL"]:  rev_sell.append(stock)
        if latest["DIAMOND_BUY"]:  diamond_buy.append(stock)
        if latest["DIAMOND_SELL"]: diamond_sell.append(stock)

        score, breakdown = conviction_score(latest_signal, latest, df, sig_df)
        if latest_signal != "NONE":
            conviction_records.append({
                "stock": stock, "signal": latest_signal, "score": score,
                "time": signal_time, "trend": latest_trend, "breakdown": breakdown,
            })

        print(f"  Current Trend   : {latest_trend}")
        print(f"  Last ST Signal  : {latest_signal}" + (f"  @ {signal_time}" if signal_time else ""))
        print(f"  QQE Reversal    : {'BUY' if latest['QQE_BUY'] else 'SELL' if latest['QQE_SELL'] else 'NONE'}")
        print(f"  Diamond Signal  : {'BUY' if latest['DIAMOND_BUY'] else 'SELL' if latest['DIAMOND_SELL'] else 'NONE'}")
        print(f"  Cloud           : {'BULLISH' if latest['CLOUD_BULL'] else 'BEARISH'}")
        print(f"  Conviction      : {score}/10")
        print(f"  Resistance      : {[f'{r:.1f}' for r in res_levels]}")
        print(f"  Support         : {[f'{s:.1f}' for s in sup_levels]}")

        print("\n  LAST 5 SUPERTREND SIGNALS:")
        if sig_df.empty:
            print("  (none)")
        else:
            for idx, row in sig_df.tail(5).iterrows():
                print(f"  {idx}  ->  {'BUY' if row['BUY'] else 'SELL'}")

        path = save_chart(df, stock, latest_trend, latest_signal, res_levels, sup_levels)
        print(f"\n  Chart saved -> {path}")

    except Exception as exc:
        print(f"\n  ERROR in {stock}: {exc}")

# =========================================================
# FINAL SUMMARY
# =========================================================

print(f"\n{'='*60}\n  FINAL SCAN RESULTS\n{'='*60}")
print(f"\n  BUY  ({len(buy_signals)}) : {buy_signals  or ['(none)']}")
print(f"  SELL ({len(sell_signals)}) : {sell_signals or ['(none)']}")
print(f"\n  QQE BUY  : {rev_buy    or ['(none)']}")
print(f"  QQE SELL : {rev_sell   or ['(none)']}")
print(f"\n  Diamond BUY  : {diamond_buy  or ['(none)']}")
print(f"  Diamond SELL : {diamond_sell or ['(none)']}")

print(f"\n{'='*60}\n  TOP {TOP_N} HIGH-CONVICTION TRADES\n{'='*60}")

buy_rec  = sorted([r for r in conviction_records if r["signal"]=="BUY"],
                   key=lambda x: x["score"], reverse=True)
sell_rec = sorted([r for r in conviction_records if r["signal"]=="SELL"],
                   key=lambda x: x["score"], reverse=True)

print(f"\n  -- TOP BUY SETUPS --")
for r in buy_rec[:TOP_N]:
    bd = " | ".join(r["breakdown"].values())
    print(f"  [{r['score']}/10] {r['stock']:18s}  {r['time']}  {bd}")

print(f"\n  -- TOP SELL SETUPS --")
for r in sell_rec[:TOP_N]:
    bd = " | ".join(r["breakdown"].values())
    print(f"  [{r['score']}/10] {r['stock']:18s}  {r['time']}  {bd}")

all_rec = sorted(conviction_records, key=lambda x: x["score"], reverse=True)
pd.DataFrame([{"Stock":r["stock"],"Signal":r["signal"],"Score":r["score"],
               "Trend":r["trend"],"SignalTime":r["time"]} for r in all_rec]
             ).to_csv("conviction_summary.csv", index=False)
print(f"\n  Full table saved -> conviction_summary.csv\n")