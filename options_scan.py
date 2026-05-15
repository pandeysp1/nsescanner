import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import date, timedelta
import calendar

warnings.filterwarnings("ignore")

# =========================================================
# SETTINGS
# =========================================================

INTERVAL    = "5m"
PERIOD      = "2d"
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

# Options-specific
MIN_CONVICTION   = 5       # minimum score to qualify for options
VOL_MULTIPLIER   = 1.5     # signal bar volume must be > 1.5x 20-bar avg
MIN_BODY_RATIO   = 0.40    # candle body must be > 40% of ATR(14)
PREMIUM_TP_MULT  = 2.0     # exit when premium = 2x paid
PREMIUM_SL_MULT  = 0.50    # exit when premium = 0.5x paid (50% loss)
TIME_STOP_DAYS   = 2       # exit N days before expiry

OUTPUT_FOLDER         = "charts_options"
OPTIONS_REPORT_CSV    = "options_report.csv"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================================================
# NIFTY 250 FETCH
# =========================================================

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

# =========================================================
# CORE MATH
# =========================================================

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

# =========================================================
# CONVICTION SCORE
# =========================================================

def conviction_score(latest_signal, latest, df, sig_df):
    score = 0
    breakdown = {}
    if latest_signal == "NONE":
        return 0, breakdown
    is_buy = latest_signal == "BUY"

    if not sig_df.empty:
        last_idx = df.index.get_loc(sig_df.index[-1])
        bars_ago = len(df) - 1 - last_idx
        if bars_ago == 0:
            score += 3; breakdown["recency"] = "Fired this bar (+3)"
        elif bars_ago <= 2:
            score += 2; breakdown["recency"] = f"Recent ({bars_ago}b ago) (+2)"
        elif bars_ago <= 8:
            score += 1; breakdown["recency"] = f"OK ({bars_ago}b ago) (+1)"
        else:
            breakdown["recency"] = f"Old ({bars_ago}b ago) (+0)"

    cloud_ok = (is_buy and latest["CLOUD_BULL"]) or (not is_buy and not latest["CLOUD_BULL"])
    if cloud_ok:
        score += 2; breakdown["cloud"] = "Cloud aligns (+2)"
    else:
        breakdown["cloud"] = "Cloud against (+0)"

    qqe_ok = (is_buy and latest["QQE_BUY"]) or (not is_buy and latest["QQE_SELL"])
    if qqe_ok:
        score += 2; breakdown["qqe"] = "QQE confirms (+2)"
    else:
        breakdown["qqe"] = "QQE neutral (+0)"

    dia_ok = (is_buy and latest["DIAMOND_BUY"]) or (not is_buy and latest["DIAMOND_SELL"])
    if dia_ok:
        score += 2; breakdown["diamond"] = "Diamond confirms (+2)"
    else:
        breakdown["diamond"] = "Diamond neutral (+0)"

    total_sigs = len(sig_df)
    if total_sigs <= 2:
        score += 1; breakdown["clean"] = f"Clean ({total_sigs} flips) (+1)"
    else:
        breakdown["clean"] = f"{total_sigs} flips (+0)"

    return score, breakdown

# =========================================================
# OPTIONS HELPERS
# =========================================================

def get_last_thursday(year, month):
    """Return the last Thursday of a given month."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    # weekday(): Monday=0 ... Thursday=3 ... Sunday=6
    offset = (d.weekday() - 3) % 7
    return d - timedelta(days=offset)

def get_expiry_dates():
    """
    Returns (near_expiry, far_expiry) as date objects.
    Near = current month last Thursday if > TIME_STOP_DAYS away, else next month.
    Far  = one month beyond near.
    NOTE: NSE individual stock options are MONTHLY only.
          Weekly options exist only for indices (Nifty, BankNifty, FinNifty, MidcapNifty).
    """
    today = date.today()
    year, month = today.year, today.month

    near = get_last_thursday(year, month)
    # If near expiry is too close (within TIME_STOP_DAYS + 1 buffer), move to next month
    if (near - today).days <= TIME_STOP_DAYS + 1:
        month += 1
        if month > 12:
            month = 1; year += 1
        near = get_last_thursday(year, month)

    # Far = month after near
    far_month = near.month + 1
    far_year  = near.year
    if far_month > 12:
        far_month = 1; far_year += 1
    far = get_last_thursday(far_year, far_month)

    return near, far

def get_strike_interval(price):
    """
    NSE standard strike intervals by price band.
    Reference: NSE F&O contract specifications.
    """
    if price < 50:      return 2.5
    elif price < 100:   return 5
    elif price < 250:   return 10
    elif price < 500:   return 25
    elif price < 1000:  return 50
    elif price < 2500:  return 100
    elif price < 5000:  return 200 # some use 100, conservative
    else:               return 500

def get_atm_strike(price):
    """Round price to nearest NSE strike interval."""
    interval = get_strike_interval(price)
    atm = round(price / interval) * interval
    return atm, interval

def check_options_entry(df, sig_df, latest_signal):
    """
    Returns a dict of entry condition checks for options qualification.
    Stricter than equity — all 4 must pass for a CONFIRMED options entry.
    """
    checks = {}
    if sig_df.empty or latest_signal == "NONE":
        return checks

    # Get the signal bar
    sig_bar_idx  = sig_df.index[-1]
    sig_bar      = df.loc[sig_bar_idx]
    sig_position = df.index.get_loc(sig_bar_idx)
    bars_ago     = len(df) - 1 - sig_position

    # 1. Recency — signal must be <= 2 bars old
    checks["recency_ok"] = bars_ago <= 2
    checks["bars_ago"]   = bars_ago

    # 2. Volume confirmation — signal bar volume > 1.5x rolling 20-bar avg
    avg_vol = df["Volume"].rolling(20).mean().iloc[sig_position]
    sig_vol = float(sig_bar["Volume"])
    vol_ratio = sig_vol / avg_vol if avg_vol > 0 else 0
    checks["volume_ok"]    = vol_ratio >= VOL_MULTIPLIER
    checks["volume_ratio"] = round(vol_ratio, 2)

    # 3. Candle body strength — body > 40% of ATR(14)
    atr14     = calc_atr(df, 14).iloc[sig_position]
    body      = abs(float(sig_bar["Close"]) - float(sig_bar["Open"]))
    body_pct  = body / atr14 if atr14 > 0 else 0
    checks["body_ok"]      = body_pct >= MIN_BODY_RATIO
    checks["body_ratio"]   = round(body_pct, 2)

    # 4. Cloud must align
    checks["cloud_ok"] = bool(latest["CLOUD_BULL"]) if latest_signal == "BUY" else not bool(latest["CLOUD_BULL"])

    # Overall pass — all 4 must be true
    checks["all_pass"] = all([
        checks.get("recency_ok", False),
        checks.get("volume_ok",  False),
        checks.get("body_ok",    False),
        checks.get("cloud_ok",   False),
    ])

    return checks

def build_options_plan(stock, signal, df, sig_df, score, near_expiry, far_expiry):
    """
    Build the complete options trade plan for a qualified stock.
    """
    today        = date.today()
    close_now    = float(df["Close"].iloc[-1])
    st_now       = float(df["Supertrend"].iloc[-1])
    atr14        = float(calc_atr(df, 14).iloc[-1])

    atm_strike, interval = get_atm_strike(close_now)

    is_buy       = signal == "BUY"
    option_type  = "CE (Call)" if is_buy else "PE (Put)"

    # ── Price levels on the underlying stock ──────────────────────────
    sl_price   = st_now                          # supertrend flip = stock SL
    target1    = close_now + (0.7 * atr14) if is_buy else close_now - (0.7 * atr14)
    target2    = close_now + (1.2 * atr14) if is_buy else close_now - (1.2 * atr14)
    sl_dist    = abs(close_now - sl_price)
    sl_pct     = (sl_dist / close_now) * 100

    # ── Expiry selection guidance ─────────────────────────────────────
    near_days  = (near_expiry - today).days
    far_days   = (far_expiry  - today).days

    # Near expiry: prefer only if score >= 9 and signal very fresh
    use_near   = score >= 9 and near_days >= 5
    # Far  expiry: safer for score 7-8, more theta buffer
    recommended_expiry     = near_expiry if use_near else far_expiry
    recommended_expiry_tag = "NEAR" if use_near else "FAR"

    # ── ATR-based rough premium estimate (very rough — real premium needs options chain) ──
    # Approximation: premium ~ 0.5 * ATR * sqrt(DTE/252) * some vol factor
    dte        = near_days if use_near else far_days
    est_premium_rough = round(atr14 * 0.8 * np.sqrt(dte / 252) * 100, 1)

    checks = check_options_entry(df, sig_df, signal)

    plan = {
        "stock":                stock,
        "signal":               signal,
        "option_type":          option_type,
        "score":                score,
        "close_now":            round(close_now, 2),
        "atm_strike":           atm_strike,
        "strike_interval":      interval,

        # Stock price levels
        "entry_price":          round(close_now, 2),
        "sl_price":             round(sl_price, 2),
        "sl_distance":          round(sl_dist, 2),
        "sl_pct":               round(sl_pct, 2),
        "target1":              round(target1, 2),
        "target2":              round(target2, 2),
        "atr14":                round(atr14, 2),

        # Expiry
        "near_expiry":          near_expiry,
        "near_days":            near_days,
        "far_expiry":           far_expiry,
        "far_days":             far_days,
        "recommended_expiry":   recommended_expiry,
        "recommended_tag":      recommended_expiry_tag,
        "est_premium":          est_premium_rough,

        # Entry checks
        "checks":               checks,

        # Exit rules
        "exit_rules": [
            f"EXIT 1 (Profit target): Sell when STOCK hits Target1={round(target1,2)} (50% qty) then Target2={round(target2,2)}",
            f"EXIT 2 (Premium 2x)   : Sell when premium = 2x your entry premium paid",
            f"EXIT 3 (Stop loss)     : Exit if STOCK closes below/above Supertrend ({round(st_now,2)})",
            f"EXIT 4 (Premium SL)    : Exit if premium drops to 50% of entry premium (hard floor)",
            f"EXIT 5 (Time stop)     : Mandatory exit {TIME_STOP_DAYS} days before expiry "
            f"({(recommended_expiry - timedelta(days=TIME_STOP_DAYS)).strftime('%d-%b-%Y')})",
            f"RULE                   : Whichever of EXIT 1-5 triggers FIRST — take it, no exceptions.",
        ],
    }
    return plan

def print_plan(plan):
    c = plan["checks"]
    all_ok = c.get("all_pass", False)
    status = "CONFIRMED ENTRY" if all_ok else "WATCH — entry not confirmed yet"
    star = "★★★" if plan["score"] >= 9 else "★★" if plan["score"] >= 7 else "★"

    print(f"""
{'─'*65}
  {star}  {plan['stock']}  |  {plan['signal']} {plan['option_type']}  |  Score: {plan['score']}/10
  Status: {status}
{'─'*65}
  Current Price : ₹{plan['close_now']}
  ATM Strike    : ₹{plan['atm_strike']}  (interval: ₹{plan['strike_interval']})

  ENTRY CONDITIONS (all 4 required for confirmed entry):
    [{'✓' if c.get('recency_ok') else '✗'}] Signal recency    : {c.get('bars_ago','?')} bars ago  (need <= 2)
    [{'✓' if c.get('volume_ok')  else '✗'}] Volume            : {c.get('volume_ratio','?')}x avg  (need >= {VOL_MULTIPLIER}x)
    [{'✓' if c.get('body_ok')    else '✗'}] Candle body       : {c.get('body_ratio','?')} of ATR  (need >= {MIN_BODY_RATIO})
    [{'✓' if c.get('cloud_ok')   else '✗'}] Cloud alignment   : {'aligned' if c.get('cloud_ok') else 'against trend'}

  STOCK PRICE LEVELS:
    Entry         : ₹{plan['entry_price']}
    Stop Loss     : ₹{plan['sl_price']}  ({plan['sl_pct']}% risk | {plan['sl_distance']} pts from entry)
    Target 1 (50%): ₹{plan['target1']}  (ATR×0.7 = {round(0.7*plan['atr14'],2)} pts)
    Target 2 (50%): ₹{plan['target2']}  (ATR×1.2 = {round(1.2*plan['atr14'],2)} pts)

  OPTIONS DETAILS:
    Action        : BUY  ₹{plan['atm_strike']} {plan['option_type']}
    Near Expiry   : {plan['near_expiry'].strftime('%d-%b-%Y')}  ({plan['near_days']} days)  ← use if score >= 9
    Far  Expiry   : {plan['far_expiry'].strftime('%d-%b-%Y')}  ({plan['far_days']} days)  ← use if score 7-8
    Recommended   : {plan['recommended_expiry'].strftime('%d-%b-%Y')}  [{plan['recommended_tag']}]
    Est. Premium* : ₹{plan['est_premium']} (rough ATR-based estimate — verify on NSE/Sensibull)

  EXIT RULES (whichever fires first):""")
    for rule in plan["exit_rules"]:
        print(f"    {rule}")
    print(f"\n  * Real premium depends on IV. Check Sensibull or NSE options chain before entry.")

def save_options_chart(df, plan, res_levels, sup_levels):
    """Chart zoomed to last 30 bars with all levels annotated."""
    stock  = plan["stock"]
    signal = plan["signal"]
    is_buy = signal == "BUY"

    # Use last 30 bars for a cleaner view
    df_plot = df.iloc[-60:].copy()

    def safe_scatter(markers, **kw):
        if np.any(~np.isnan(markers)):
            apds.append(mpf.make_addplot(markers, type="scatter", **kw))

    apds = []
    apds.append(mpf.make_addplot(df_plot["Supertrend"], color="orange", width=2))

    buy_m  = np.where(df_plot["BUY"],  df_plot["Low"]  * 0.997, np.nan)
    sell_m = np.where(df_plot["SELL"], df_plot["High"] * 1.003, np.nan)
    safe_scatter(buy_m,  marker="^", markersize=120, color="lime")
    safe_scatter(sell_m, marker="v", markersize=120, color="red")

    mc    = mpf.make_marketcolors(up="green", down="red",
                                   edge="inherit", wick="inherit", volume="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle="--", facecolor="#0a0a0a")

    c      = plan["checks"]
    status = "CONFIRMED" if c.get("all_pass") else "WATCH"
    title  = (f"{stock} | {signal} {plan['option_type']} | "
              f"Strike ₹{plan['atm_strike']} | {plan['recommended_expiry'].strftime('%d-%b-%Y')} | "
              f"Score:{plan['score']}/10 | {status}")

    fig, axes = mpf.plot(
        df_plot, type="candle", style=style, title=title,
        volume=True, figsize=(18, 10), addplot=apds, returnfig=True,
    )
    ax = axes[0]

    # Buy/Sell labels
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        if row["BUY"]:
            ax.annotate("Buy", xy=(i, row["Low"]*0.997), xytext=(i, row["Low"]*0.988),
                        fontsize=8, fontweight="bold", color="white", ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0099BB", edgecolor="white", lw=0.5, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color="#0099BB", lw=1))
        if row["SELL"]:
            ax.annotate("Sell", xy=(i, row["High"]*1.003), xytext=(i, row["High"]*1.012),
                        fontsize=8, fontweight="bold", color="white", ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#CC2244", edgecolor="white", lw=0.5, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color="#CC2244", lw=1))

    # Entry, SL, T1, T2 horizontal lines
    levels = [
        (plan["entry_price"], "#FFFFFF", "Entry",   "--", 1.5),
        (plan["sl_price"],    "#FF4444", "SL",      "--", 1.5),
        (plan["target1"],     "#00FF88", "T1 (50%)", "-.", 1.5),
        (plan["target2"],     "#00FF88", "T2 (50%)", "-.", 1.5),
    ]
    for lvl, col, label, ls, lw in levels:
        ax.axhline(lvl, color=col, linewidth=lw, linestyle=ls, alpha=0.85)
        ax.text(0.5, lvl, f" {label}: ₹{lvl}", color=col, fontsize=7.5,
                va="bottom", ha="left", alpha=0.95,
                bbox=dict(facecolor="#0a0a0a", alpha=0.5, pad=1, edgecolor="none"))

    # S&R zones
    for lvl in res_levels:
        ax.axhline(lvl, color="#E93E63", linewidth=1, linestyle="--", alpha=0.6)
        ax.axhspan(lvl*0.998, lvl*1.002, facecolor="#E93E63", alpha=0.1)
        ax.text(len(df_plot)-1, lvl, f"R {lvl:.1f}", color="#E93E63",
                fontsize=6, va="bottom", ha="right", alpha=0.85)

    for lvl in sup_levels:
        ax.axhline(lvl, color="#00DBFF", linewidth=1, linestyle="--", alpha=0.6)
        ax.axhspan(lvl*0.998, lvl*1.002, facecolor="#00DBFF", alpha=0.1)
        ax.text(len(df_plot)-1, lvl, f"S {lvl:.1f}", color="#00DBFF",
                fontsize=6, va="top", ha="right", alpha=0.85)

    # Annotation box — trade summary
    box_text = (
        f"TRADE PLAN\n"
        f"Option : ₹{plan['atm_strike']} {plan['option_type']}\n"
        f"Expiry : {plan['recommended_expiry'].strftime('%d-%b-%Y')}\n"
        f"Entry  : ₹{plan['entry_price']}\n"
        f"SL     : ₹{plan['sl_price']} ({plan['sl_pct']}%)\n"
        f"T1     : ₹{plan['target1']}\n"
        f"T2     : ₹{plan['target2']}\n"
        f"Exit   : First of: T1/T2 | 2x prem | ST flip | 50% prem SL"
    )
    ax.text(0.01, 0.99, box_text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                      edgecolor="#4444AA", linewidth=1, alpha=0.92))

    save_path = os.path.join(OUTPUT_FOLDER, f"OPT_{stock}.png")
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    plt.close(fig)
    return save_path

# =========================================================
# MAIN
# =========================================================

print("\nFetching Nifty 250 constituent list...")
stocks = fetch_nifty250()
print(f"  Total stocks to scan: {len(stocks)}\n")

near_expiry, far_expiry = get_expiry_dates()
today = date.today()
print(f"  Scan date            : {today.strftime('%d-%b-%Y')}")
print(f"  Near expiry (monthly): {near_expiry.strftime('%d-%b-%Y')}  ({(near_expiry-today).days} days)")
print(f"  Far  expiry (monthly): {far_expiry.strftime('%d-%b-%Y')}  ({(far_expiry-today).days} days)")
print(f"\n  NOTE: NSE stock options are MONTHLY only.")
print(f"        Weekly options exist only for Nifty/BankNifty/FinNifty/MidcapNifty indices.\n")

buy_signals, sell_signals = [], []
rev_buy, rev_sell         = [], []
diamond_buy, diamond_sell = [], []
conviction_records        = []
options_plans             = []    # qualified options setups

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
        print(f"  Rows downloaded : {len(df)}")

        MIN_BARS = max(ATR_PERIOD, RSI_PERIOD, SR_LEFT+SR_RIGHT) + 10
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

        if latest["QQE_BUY"]:     rev_buy.append(stock)
        if latest["QQE_SELL"]:    rev_sell.append(stock)
        if latest["DIAMOND_BUY"]:  diamond_buy.append(stock)
        if latest["DIAMOND_SELL"]: diamond_sell.append(stock)

        score, breakdown = conviction_score(latest_signal, latest, df, sig_df)
        if latest_signal != "NONE":
            conviction_records.append({
                "stock": stock, "signal": latest_signal, "score": score,
                "time": signal_time, "trend": latest_trend, "breakdown": breakdown,
            })

        print(f"  Trend    : {latest_trend}  |  Signal: {latest_signal}  |  Score: {score}/10")

        # ── Options qualification ────────────────────────────────────
        if score >= MIN_CONVICTION and latest_signal != "NONE":
            plan = build_options_plan(
                stock, latest_signal, df, sig_df, score, near_expiry, far_expiry)
            options_plans.append((plan, res_levels, sup_levels, df))
            confirmed = plan["checks"].get("all_pass", False)
            print(f"  OPTIONS  : QUALIFIED (score {score})  |  {'CONFIRMED ENTRY' if confirmed else 'WATCH — waiting for entry conditions'}")
        else:
            if score < MIN_CONVICTION:
                print(f"  OPTIONS  : Not qualified (score {score} < {MIN_CONVICTION})")

    except Exception as exc:
        print(f"\n  ERROR in {stock}: {exc}")

# =========================================================
# FINAL SUMMARY — Conviction
# =========================================================

print(f"\n{'='*65}")
print(f"  FINAL SCAN RESULTS")
print(f"{'='*65}")
print(f"\n  BUY  ({len(buy_signals)}) : {buy_signals  or ['(none)']}")
print(f"  SELL ({len(sell_signals)}) : {sell_signals or ['(none)']}")
print(f"  QQE BUY  : {rev_buy    or ['(none)']}")
print(f"  QQE SELL : {rev_sell   or ['(none)']}")
print(f"  Diamond BUY  : {diamond_buy  or ['(none)']}")
print(f"  Diamond SELL : {diamond_sell or ['(none)']}")

# =========================================================
# OPTIONS REPORT
# =========================================================

print(f"\n{'='*65}")
print(f"  OPTIONS TRADING REPORT  —  Score >= {MIN_CONVICTION}")
print(f"  {today.strftime('%d-%b-%Y')}  |  Near: {near_expiry.strftime('%d-%b-%Y')}  |  Far: {far_expiry.strftime('%d-%b-%Y')}")
print(f"{'='*65}")

# Sort: confirmed first, then by score
options_plans.sort(key=lambda x: (x[0]["checks"].get("all_pass", False), x[0]["score"]), reverse=True)

confirmed_plans = [(p,r,s,d) for p,r,s,d in options_plans if p["checks"].get("all_pass")]
watch_plans     = [(p,r,s,d) for p,r,s,d in options_plans if not p["checks"].get("all_pass")]

print(f"\n  CONFIRMED ENTRIES ({len(confirmed_plans)} stocks — enter now):")
for plan, res_levels, sup_levels, df in confirmed_plans:
    print_plan(plan)
    path = save_options_chart(df, plan, res_levels, sup_levels)
    print(f"  Chart: {path}")

print(f"\n  ON WATCH ({len(watch_plans)} stocks — wait for entry conditions):")
for plan, res_levels, sup_levels, df in watch_plans:
    print_plan(plan)
    path = save_options_chart(df, plan, res_levels, sup_levels)
    print(f"  Chart: {path}")

# ── Save CSV ──────────────────────────────────────────────────────────────
rows = []
for plan, _, _, _ in options_plans:
    c = plan["checks"]
    rows.append({
        "Stock":             plan["stock"],
        "Signal":            plan["signal"],
        "OptionType":        plan["option_type"],
        "Score":             plan["score"],
        "ATM_Strike":        plan["atm_strike"],
        "EntryPrice":        plan["entry_price"],
        "StopLoss":          plan["sl_price"],
        "SL_Pct":            plan["sl_pct"],
        "Target1":           plan["target1"],
        "Target2":           plan["target2"],
        "ATR14":             plan["atr14"],
        "NearExpiry":        plan["near_expiry"].strftime("%d-%b-%Y"),
        "NearDays":          plan["near_days"],
        "FarExpiry":         plan["far_expiry"].strftime("%d-%b-%Y"),
        "FarDays":           plan["far_days"],
        "RecommendedExpiry": plan["recommended_expiry"].strftime("%d-%b-%Y"),
        "RecommendedTag":    plan["recommended_tag"],
        "EstPremium_Rough":  plan["est_premium"],
        "Confirmed":         c.get("all_pass", False),
        "RecencyOK":         c.get("recency_ok", False),
        "VolumeOK":          c.get("volume_ok", False),
        "VolumeRatio":       c.get("volume_ratio", 0),
        "BodyOK":            c.get("body_ok", False),
        "BodyRatio":         c.get("body_ratio", 0),
        "CloudOK":           c.get("cloud_ok", False),
    })

df_out = pd.DataFrame(rows).sort_values(["Confirmed","Score"], ascending=[False,False])
df_out.to_csv(OPTIONS_REPORT_CSV, index=False)
print(f"\n{'='*65}")
print(f"  Options report saved -> {OPTIONS_REPORT_CSV}")
print(f"  Options charts saved -> {OUTPUT_FOLDER}/")
print(f"{'='*65}\n")