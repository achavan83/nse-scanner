from flask import Flask, jsonify, request, render_template
import requests
import pandas as pd
import csv
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import time
from datetime import time as dt_time
import numpy as np
import os

# Cache storage
STOCK_CACHE = {}
CACHE_TTL = 60   # seconds (1 minute)
STOCK_LEVELS = {}      # Stores PDH, PDL, ORB
OBR_LEVELS = {}
LTP_CACHE = {}         # Stores latest LTP
LAST_LTP_UPDATE = 0
LTP_REFRESH_INTERVAL = 180   # seconds

orb_candles = 6
samllBreakout = 15
bigBreakout = 25

timeFrame = "5minute"
DAYS = 2
INTERVAL = "5minute"
RESOLUTION = 500

# -------------------------------
# KITE CONNECTION
# -------------------------------

API_KEY = os.environ.get("KITE_API_KEY", "h81rgom2pxekmyx0")
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "W10g62JV5eCLdJqoYdCOYHv3ve8lUzoX")

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)
print("Loading instruments...")
INSTRUMENTS = kite.instruments("NSE")

SYMBOL_TOKEN_MAP = {}

for ins in INSTRUMENTS:
    SYMBOL_TOKEN_MAP[ins["tradingsymbol"]] = ins["instrument_token"]

print("Instrument map ready")


# -------------------------------
# HELPERS
# -------------------------------
def is_strong_candle(o, h, l, c):
    body = abs(c - o)
    range_ = h - l
    if range_ == 0:
        return False
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return (body / range_) > 0.5 and upper_wick < body * 0.3 and lower_wick < body * 0.3


def get_orb(df, start_t, end_t):
    temp = df[(df['date'].dt.time >= start_t) & (df['date'].dt.time <= end_t)]
    if temp.empty:
        return None, None
    return temp['high'].max(), temp['low'].min()


def calculate_vpoc(df, resolution=200):
    high_price = df['high'].max()
    low_price = df['low'].min()
    diff = (high_price - low_price) / resolution
    price_levels = np.array([low_price + diff * i for i in range(resolution)])
    total_vol = np.zeros(resolution)

    for i in range(len(df)):
        h = df.iloc[i]['high']
        l = df.iloc[i]['low']
        v = df.iloc[i]['volume']
        touched = []
        for j in range(resolution):
            price = price_levels[j]
            if l < price < h:
                touched.append(j)
        if touched:
            vol_share = v / len(touched)
            for j in touched:
                total_vol[j] += vol_share

    max_idx = np.argmax(total_vol)
    vpoc = price_levels[max_idx]
    max_vol = np.max(total_vol)
    total_volume = df['volume'].sum()
    vpoc_idx = np.argmax(total_vol)
    vpoc_volume = total_vol[vpoc_idx]
    vpoc_percentage = (vpoc_volume / total_volume) * 100
    top_5_bins = np.sort(total_vol)[-4:]
    concentration = (top_5_bins.sum() / total_volume) * 100
    bar_width = total_vol / max_vol
    threshold = 0.05
    filtered = bar_width[bar_width > threshold]
    avg_width = np.mean(filtered) if len(filtered) > 0 else 0
    return vpoc, concentration, avg_width


def get_previous_day_vpoc(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.date
    unique_days = df['day'].unique()
    if len(unique_days) < 2:
        return None
    prev_day = unique_days[-2]
    prev_df = df[df['day'] == prev_day]
    return calculate_vpoc(prev_df, RESOLUTION)


def calculate_cpr(df_daily):
    df = df_daily.copy()
    df['P'] = (df['high'] + df['low'] + df['close']) / 3
    df['BC'] = (df['high'] + df['low']) / 2
    df['TC'] = (2 * df['P']) - df['BC']
    df['CPR_Width'] = df['TC'] - df['BC']
    return df


def cpr_signal(df):
    df = calculate_cpr(df)
    yday = df.iloc[-2]
    cpr_pct = (yday['CPR_Width'] / yday['close']) * 100
    if cpr_pct < 0.8:
        return "NAR"
    elif cpr_pct > 1.5:
        return "WIDE"
    else:
        return "NOR"


def get_current_day_vpoc(df):
    df['date'] = pd.to_datetime(df['date'])
    today = df['date'].dt.date.iloc[-1]
    curr_df = df[df['date'].dt.date == today]
    return calculate_vpoc(curr_df, RESOLUTION)


def combined_deviation_simple(current_price, poc, vwap):
    combined_price = (poc + vwap) / 2
    deviation_pct = ((current_price - combined_price) / combined_price) * 100
    return round(combined_price, 2), round(deviation_pct, 2)


def combined_deviation(current, poc, vwap, w_poc=0.5, w_vwap=0.5):
    combined_price = (poc * w_poc) + (vwap * w_vwap)
    deviation = ((current - vwap) / vwap) * 100
    return round(combined_price, 2), round(deviation, 2)


def calculate_dpo(df, period=20):
    shift = int(period / 2 + 1)
    sma = df['close'].rolling(window=period).mean()
    df['DPO'] = df['close'] - sma.shift(shift)
    return df


def calculate_macd(df):
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df


def macd_crossover_signal(df):
    df = df.copy()
    recent = df.iloc[-6:]
    crossover = (
        (recent['MACD'].shift(1) <= recent['Signal'].shift(1)) &
        (recent['MACD'] > recent['Signal'])
    )
    last = df.iloc[-1]
    buy = (
        crossover.any() and
        last['MACD'] < 0 and
        (last['MACD'] - last['Signal']) > 0
    )
    sell = (
        ((recent['MACD'].shift(1) >= recent['Signal'].shift(1)) &
         (recent['MACD'] < recent['Signal'])).any() and
        last['MACD'] > 0 and
        (last['MACD'] - last['Signal']) < 0
    )
    return {"buy_signal": bool(buy), "sell_signal": bool(sell)}


def final_signal_engine(df_daily, df_intraday):
    if isinstance(df_daily, list):
        df_daily = pd.DataFrame(df_daily)
    if isinstance(df_intraday, list):
        df_intraday = pd.DataFrame(df_intraday)

    df_daily['P'] = (df_daily['high'] + df_daily['low'] + df_daily['close']) / 3
    df_daily['BC'] = (df_daily['high'] + df_daily['low']) / 2
    df_daily['TC'] = (2 * df_daily['P']) - df_daily['BC']
    df_daily['CPR_Width'] = df_daily['TC'] - df_daily['BC']

    yday = df_daily.iloc[-2]
    cpr_pct = (yday['CPR_Width'] / yday['close']) * 100
    narrow_cpr = cpr_pct < 0.5

    P = yday['P']
    R1 = (2 * P) - yday['low']
    S1 = (2 * P) - yday['high']

    day_high = df_intraday['high'].max()
    day_low = df_intraday['low'].min()
    r1_break = day_high > R1
    s1_break = day_low < S1

    df_intraday['EMA9'] = df_intraday['close'].ewm(span=9, adjust=False).mean()
    df_intraday['EMA21'] = df_intraday['close'].ewm(span=20, adjust=False).mean()
    last = df_intraday.iloc[-1]
    ema_up = last['close'] > last['EMA9'] > last['EMA21']
    ema_down = last['close'] < last['EMA9'] < last['EMA21']

    df_intraday['EMA12'] = df_intraday['close'].ewm(span=12, adjust=False).mean()
    df_intraday['EMA26'] = df_intraday['close'].ewm(span=26, adjust=False).mean()
    df_intraday['MACD'] = df_intraday['EMA12'] - df_intraday['EMA26']
    df_intraday['Signal'] = df_intraday['MACD'].ewm(span=9, adjust=False).mean()
    df_intraday['Hist'] = df_intraday['MACD'] - df_intraday['Signal']

    df_intraday['bull_cross'] = (
        (df_intraday['MACD'].shift(1) <= df_intraday['Signal'].shift(1)) &
        (df_intraday['MACD'] > df_intraday['Signal'])
    )
    df_intraday['bear_cross'] = (
        (df_intraday['MACD'].shift(1) >= df_intraday['Signal'].shift(1)) &
        (df_intraday['MACD'] < df_intraday['Signal'])
    )

    lookback = 6
    recent = df_intraday.tail(lookback)
    last = df_intraday.iloc[-1]
    prev = df_intraday.iloc[-2]

    macd_buy = (
        recent['bull_cross'].any() and
        last['MACD'] > last['Signal'] and
        last['MACD'] < 0 and
        last['MACD'] > prev['MACD']
    )
    macd_sell = (
        recent['bear_cross'].any() and
        last['MACD'] < last['Signal'] and
        last['MACD'] > 0 and
        last['MACD'] < prev['MACD']
    )
    macd_strong_buy = macd_buy and last['Hist'] > prev['Hist']
    macd_strong_sell = macd_sell and last['Hist'] < prev['Hist']

    df_intraday['date'] = pd.to_datetime(df_intraday['date'])
    today = df_intraday['date'].dt.date.max()
    yesterday = sorted(df_intraday['date'].dt.date.unique())[-2]
    t = df_intraday['date'].dt.time
    start_time = pd.to_datetime("09:15").time()
    end_time = min(pd.to_datetime("09:45").time(), pd.Timestamp.now().time())
    today_vol = df_intraday[
        (df_intraday['date'].dt.date == today) & (t >= start_time) & (t <= end_time)
    ]['volume'].sum()
    yday_vol = df_intraday[
        (df_intraday['date'].dt.date == yesterday) & (t >= start_time) & (t <= end_time)
    ]['volume'].sum()
    volume_spike = today_vol > yday_vol

    if narrow_cpr and r1_break and ema_up and macd_strong_buy and volume_spike:
        return "BUY"
    elif narrow_cpr and s1_break and ema_down and macd_strong_sell and volume_spike:
        return "SELL"
    elif not r1_break and not s1_break:
        return "RANGE"
    else:
        return "NO TRADE"


def calculate_adx(df, period=14):
    df = df.copy()
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift()))
    )
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    df['ADX'] = adx
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    return df


def calculate_rsi(df, period=14):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_ema(df):
    df = df.copy()
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    return df


def ema_position_signal(df):
    df = calculate_ema(df)
    last = df.iloc[-1]
    price = last['close']
    ema9 = last['EMA9']
    ema21 = last['EMA21']
    if price > ema9 > ema21:
        return "BUY"
    elif price < ema9 < ema21:
        return "SELL"
    else:
        return "SIDEWAYS"


def supertrend_ok(df, period=10, multiplier=2.0):
    df = df.copy()
    high = df['high']
    low = df['low']
    close = df['close']
    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift()), abs(low - close.shift()))
    )
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    for i in range(1, len(df)):
        if close[i-1] <= final_upper[i-1]:
            final_upper[i] = min(upperband[i], final_upper[i-1])
        else:
            final_upper[i] = upperband[i]
        if close[i-1] >= final_lower[i-1]:
            final_lower[i] = max(lowerband[i], final_lower[i-1])
        else:
            final_lower[i] = lowerband[i]
    supertrend = pd.Series(index=df.index, dtype=float)
    for i in range(1, len(df)):
        if close[i] > final_upper[i-1]:
            supertrend[i] = final_lower[i]
        elif close[i] < final_lower[i-1]:
            supertrend[i] = final_upper[i]
        else:
            supertrend[i] = supertrend[i-1]
    df['Supertrend'] = supertrend
    df['ST_Signal'] = np.where(df['close'] > df['Supertrend'], "BUY", "SELL")
    return df


def calculate_pivots(df):
    df = df.copy()
    df['P'] = (df['high'] + df['low'] + df['close']) / 3
    df['R1'] = (2 * df['P']) - df['low']
    df['S1'] = (2 * df['P']) - df['high']
    df['R2'] = df['P'] + (df['high'] - df['low'])
    df['S2'] = df['P'] - (df['high'] - df['low'])
    df['R3'] = df['high'] + 2 * (df['P'] - df['low'])
    df['S3'] = df['low'] - 2 * (df['high'] - df['P'])
    return df


def pivot_break_from_df(day_high, day_low, R1, S1, R2, S2):
    return {
        "R1_break": bool(day_high > R1),
        "S1_break": bool(day_low < S1),
        "R2_break": bool(day_high > R2),
        "S2_break": bool(day_low < S2)
    }


def volume_ma_break(df, ma_period=20):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df[f'Vol_MA{ma_period}'] = df['volume'].rolling(ma_period).mean()
    prev_vol = df['volume'].shift(1)
    prev_ma = df[f'Vol_MA{ma_period}'].shift(1)
    curr_vol = df['volume']
    curr_ma = df[f'Vol_MA{ma_period}']
    cross = (curr_vol > curr_ma) & (df['date'].dt.date == df['date'].dt.date.shift(1))
    cross_count = int(cross.sum())
    today = df['date'].dt.date.iloc[-1]
    cross_today_mask = cross & (df['date'].dt.date == today)
    cross_times_today = df.loc[cross_today_mask, 'date'].dt.strftime("%H:%M").tolist()
    cross_today = cross[df['date'].dt.date == today]
    cross_count_today = int(cross_today.sum())
    df['vol_above_ma'] = df['volume'] > df[f'Vol_MA{ma_period}']
    first_4 = df[df['date'].dt.date == df['date'].dt.date.iloc[-1]].head(3)
    first_4_above_ma = bool(first_4['vol_above_ma'].all()) if len(first_4) == 3 else False
    today_df = df[df['date'].dt.date == df['date'].dt.date.iloc[-1]].head(6)
    first_6_above_ma = bool(today_df['vol_above_ma'].sum() >= 3) if len(today_df) >= 6 else False
    cross_df = df.loc[cross]
    if cross_df.empty:
        return {"cross": False, "cross_count": 0, "cross_count_today": 0, "time": None, "first_4_above_ma": None, "first_6_above_ma": False}
    last_cross = cross_df.iloc[-1]
    return {
        "cross": True,
        "cross_count": cross_count,
        "cross_count_today": cross_count_today,
        "time": last_cross['date'].strftime("%H:%M"),
        "volume": float(last_cross['volume']),
        "vol_ma": float(last_cross[f'Vol_MA{ma_period}']),
        "cross_times_today": cross_times_today,
        "first_4_above_ma": bool(first_4_above_ma),
        "first_6_above_ma": bool(first_6_above_ma)
    }


state = {}


def scan_market_and_update(symbol, token):
    if not token:
        return None
    today = datetime.now().date()
    from_date = today - timedelta(days=7)
    try:
        data = kite.historical_data(token, from_date, today, "5minute")
        daily = kite.historical_data(token, from_date, today, "day")
    except Exception as e:
        print("Scan error:", e)
        return None

    df = pd.DataFrame(data)
    if df.empty:
        return None

    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.date
    days = sorted(df['day'].unique())

    if today not in days:
        return None

    idx = days.index(today)
    if idx == 0:
        return None

    prev_date = days[idx - 1]
    df_prev = df[df['day'] == prev_date]
    df_curr = df[df['day'] == today]

    if df_prev.empty or df_curr.empty:
        return None

    curr_vpoc, curmax_idx, curwidth = get_current_day_vpoc(df)
    dfNew = calculate_dpo(df, period=20)
    dfNew = calculate_macd(df)
    dfNewRSI = calculate_rsi(df)
    ema = ema_position_signal(df)
    volume20 = volume_ma_break(df, 20)
    dfNewsuper1 = supertrend_ok(df, 10, 1)
    dfNewsuper2 = supertrend_ok(df, 10, 2)
    dfNew = calculate_adx(df)
    dfvolume = first_30min_volume_check(df)
    dfslope = slope_strength(df)
    orbBreakoutWithVol = first_30min_breakout(df, range_minutes=bigBreakout)
    orbBreakoutWithVol_15 = first_30min_breakout(df, range_minutes=samllBreakout)
    dfADX = dfNew[['ADX', '+DI', '-DI']].dropna().iloc[-1]
    macdSignal = macd_crossover_signal(dfNew)

    last_row = dfNew.iloc[-1]
    last_row_RSI = dfNewRSI.iloc[-1]
    dfsuper1 = dfNewsuper1.iloc[-1]
    dfsuper2 = dfNewsuper2.iloc[-1]

    macd_value = last_row['MACD']
    signal_value = last_row['Signal']
    if macd_value > signal_value:
        macdaction = "BUY"
    elif macd_value < signal_value:
        macdaction = "SELL"
    else:
        macdaction = "HOLD"

    dpo_value = last_row['DPO']
    rsi_value = last_row_RSI['RSI']
    dfNewsuper_1 = dfsuper1['ST_Signal']
    dfNewsuper_2 = dfsuper2['ST_Signal']
    dayRange = df_curr.iloc[-1]

    STOCK_LEVELS[symbol]["curr_vpoc"] = float(curr_vpoc)
    STOCK_LEVELS[symbol]["curmax_idx"] = float(curmax_idx)
    STOCK_LEVELS[symbol]["curwidth"] = float(curwidth)
    STOCK_LEVELS[symbol]["df_DPO"] = dpo_value
    STOCK_LEVELS[symbol]["df_MACD"] = macdaction
    STOCK_LEVELS[symbol]["rsi_value"] = rsi_value
    STOCK_LEVELS[symbol]["dfNewsuper_1"] = dfNewsuper_1
    STOCK_LEVELS[symbol]["dfNewsuper_2"] = dfNewsuper_2
    STOCK_LEVELS[symbol]["dfADX"] = dfADX['ADX']
    STOCK_LEVELS[symbol]["dfPlusDI"] = dfADX['+DI']
    STOCK_LEVELS[symbol]["dfMinusDI"] = dfADX['-DI']
    STOCK_LEVELS[symbol]["dfvolume"] = dfvolume
    STOCK_LEVELS[symbol]["dfema"] = ema
    STOCK_LEVELS[symbol]["cdh"] = dayRange["high"]
    STOCK_LEVELS[symbol]["cdl"] = dayRange["low"]
    STOCK_LEVELS[symbol]["macdSignal"] = macdSignal
    STOCK_LEVELS[symbol]["orbBreakoutWithVol"] = orbBreakoutWithVol
    STOCK_LEVELS[symbol]["orbBreakoutWithVol_15"] = orbBreakoutWithVol_15
    STOCK_LEVELS[symbol]["volume20"] = volume20
    STOCK_LEVELS[symbol]["dfslope"] = dfslope

    pdh_30, pdl_30 = get_orb(df_prev, dt_time(9, 15), dt_time(9, 45))
    cdh_30, cdl_30 = get_orb(df_curr, dt_time(9, 15), dt_time(9, 45))

    if not pdh_30 or not cdh_30:
        return None

    df_curr = df_curr.sort_values("date")
    df_curr['cum_vol'] = df_curr['volume'].cumsum()
    df_curr['cum_vol_price'] = (df_curr['close'] * df_curr['volume']).cumsum()
    df_curr['vwap'] = df_curr['cum_vol_price'] / df_curr['cum_vol']

    if symbol not in state:
        state[symbol] = {"pd_break": None, "invalid": False, "alert": False, "entry": None}

    s = state[symbol]

    for i in range(len(df_curr)):
        row = df_curr.iloc[i]
        current_time = row['date'].time()
        if current_time <= dt_time(9, 45):
            continue

        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        vwap = row['vwap']

        if s["alert"]:
            return {"signal": s["entry"]["signal"], "price": s["entry"]["price"], "time": s["entry"]["time"]}

        if s["pd_break"] is None:
            if (c > cdh_30 and is_strong_candle(o, h, l, c)) or (c < cdl_30 and is_strong_candle(o, h, l, c)):
                s["invalid"] = True

        if s["invalid"]:
            return None

        if s["pd_break"] is None:
            if c > pdh_30 and is_strong_candle(o, h, l, c):
                s["pd_break"] = "BUY"
            elif c < pdl_30 and is_strong_candle(o, h, l, c):
                s["pd_break"] = "SELL"
        else:
            if s["pd_break"] == "BUY":
                if c > cdh_30 and is_strong_candle(o, h, l, c) and l > vwap:
                    s["alert"] = True
                    s["entry"] = {"signal": "BUY", "price": c, "time": str(row['date'])}
            elif s["pd_break"] == "SELL":
                if c < cdl_30 and is_strong_candle(o, h, l, c) and h < vwap:
                    s["alert"] = True
                    s["entry"] = {"signal": "SELL", "price": c, "time": str(row['date'])}

    if s.get("entry"):
        return s["entry"]
    return None


def check_pdh_pdl_break(symbol):
    now = time.time()
    if symbol in STOCK_CACHE:
        cached_data = STOCK_CACHE[symbol]
        if now - cached_data["timestamp"] < CACHE_TTL:
            return cached_data["data"]

    try:
        instrument = f"NSE:{symbol}"
        ltp_data = kite.ltp([instrument])
        if instrument not in ltp_data:
            return None

        ltp = ltp_data[instrument]["last_price"]
        instrument_token = ltp_data[instrument]["instrument_token"]
        today = datetime.now().date()
        from_date = today - timedelta(days=5)
        candles = kite.historical_data(instrument_token, from_date, today, "day")

        if len(candles) < 2:
            return None

        previous_day = candles[-2]
        pdh = previous_day["high"]
        pdl = previous_day["low"]
        df = calculate_pivots(candles)
        yday = df.iloc[-2]

        signal = None
        if ltp > pdh:
            signal = "BUY"
        elif ltp < pdl:
            signal = "SELL"

        result = {
            "pdh": pdh, "pdl": pdl, "ltp": ltp, "signal": signal,
            "P": float(yday['P']), "R1": float(yday['R1']), "R2": float(yday['R2']),
            "R3": float(yday['R3']), "S1": float(yday['S1']), "S2": float(yday['S2']), "S3": float(yday['S3'])
        }

        STOCK_CACHE[symbol] = {"timestamp": now, "data": result}
        return result

    except Exception as e:
        print("Error:", e)
        return None


def prepare_stock_levels(symbol):
            return False

        first_6 = intraday[:orb_candles]

        if len(first_6) < orb_candles:
            return False

        orb_high = max(c['high'] for c in first_6)
        orb_low = min(c['low'] for c in first_6)

        day_high = max(c['high'] for c in intraday)
        day_low = min(c['low'] for c in intraday)

        last_close = float(df.iloc[-1]['close'])

        dfday = pd.DataFrame(daily)
        privousdf = calculate_pivots(dfday)

        yday = privousdf.iloc[-2]

        cprsignal = cpr_signal(dfday)

        STOCK_LEVELS[symbol] = {
            "token": token,
            "pdh": pdh,
            "pdl": pdl,
            "cdh": cdh,
            "cdl": cdl,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "day_high": day_high,
            "day_low": day_low,
            "pdVol": pdVol,
            "last_close": last_close,
            "prev_vpoc": float(prev_vpoc) if prev_vpoc else 0,
            "curr_vpoc": float(curr_vpoc),
            "premax_idx": float(premax_idx) if premax_idx else 0,
            "curmax_idx": float(curmax_idx),
            "curwidth": float(curwidth),
            "prewidth": float(prewidth) if prewidth else 0,
            "df_DPO": float(dpo_value),
            "df_MACD": macdaction,
            "dfADX": float(dfADX['ADX']),
            "dfPlusDI": float(dfADX['+DI']),
            "dfMinusDI": float(dfADX['-DI']),
            "cprsignal": cprsignal,
            "rsi_value": float(rsi_value),
            "dfNewsuper_1": dfNewsuper_1,
            "dfNewsuper_2": dfNewsuper_2,
            "dfvolume": bool(dfvolume),
            "dfema": ema,
            "P": float(yday['P']),
            "R1": float(yday['R1']),
            "R2": float(yday['R2']),
            "R3": float(yday['R3']),
            "S1": float(yday['S1']),
            "S2": float(yday['S2']),
            "S3": float(yday['S3']),
            "macdSignal": macdSignal,
            "orbBreakoutWithVol": orbBreakoutWithVol,
            "orbBreakoutWithVol_15": orbBreakoutWithVol_15,
            "volume20": volume20,
            "dfslope": dfslope
        }

        return True

    except Exception as e:
        print(f"prepare_stock_levels ERROR {symbol}: {e}")
        return False


def check_breakout(last_close, curr_vpoc, vwap, cpr, adx, adxdiplus, adxminus, dpo, macd,
                   deviation, vwapdiff, rsi, supertrend1, supertrend2, volume, ema,
                   macdSignal, orbBreakoutWithVol, volume20, first_6_above_ma, overall_slope, symbol):
    signal = None
    if (last_close > vwap and cpr == "NAR" and adx > 25 and adxdiplus > 20 and rsi > 50
            and supertrend1 == "BUY" and supertrend2 == "BUY" and volume == True
            and ema == "BUY" and orbBreakoutWithVol['buy'] == True
            and volume20 == True and first_6_above_ma == True and overall_slope == True):
        signal = "BULL"
    elif (last_close < vwap and cpr == "NAR" and adx > 25 and adxminus > 20 and rsi < 40
          and supertrend1 == "SELL" and supertrend2 == "SELL" and volume == True
          and ema == "SELL" and orbBreakoutWithVol['sell'] == True
          and volume20 == True and first_6_above_ma == True and overall_slope == True):
        signal = "BEAR"
    return signal


def get_volume_change_percent(new_value, old_value):
    old_vol = float(old_value)
    new_vol = float(new_value)
    if old_vol == 0:
        return "0.00"
    change = ((new_vol - old_vol) / old_vol) * 100
    return "{:.2f}".format(change)


def first_30min_volume_check(df):
    df['date'] = pd.to_datetime(df['date'])
    today = df['date'].dt.date.max()
    yesterday = sorted(df['date'].dt.date.unique())[-2]
    now = datetime.now().time()
    start_time = pd.to_datetime("09:15").time()
    end_time = now
    t = df['date'].dt.time
    today_vol = df[(df['date'].dt.date == today) & (t >= start_time) & (t <= end_time)]['volume'].sum()
    yday_vol = df[(df['date'].dt.date == yesterday) & (t >= start_time) & (t <= end_time)]['volume'].sum()
    return bool(today_vol > yday_vol)


def slope_strength(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    today = df['date'].dt.date.max()
    df = df[df['date'].dt.date == today]

    if len(df) < 5:
        return {
            "ema_up": False,
            "ema_down": False,
            "ema9_up": False,
            "ema9_down": False,
            "vwap_up": False,
            "vwap_down": False,
            "ema_strength": 0,
            "ema9_strength": 0,
            "vwap_strength": 0,
            "ema_distance_pct": 0,
            "vwap_distance_pct": 0,
            "ema_gap": 0
        }

    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['TPV'] = df['TP'] * df['volume']
    df['Cum_TPV'] = df['TPV'].cumsum()
    df['Cum_Vol'] = df['volume'].cumsum()
    df['VWAP'] = df['Cum_TPV'] / df['Cum_Vol']

    ema_slope = df['EMA20'].iloc[-1] - df['EMA20'].iloc[0]
    ema9_slope = df['EMA9'].iloc[-1] - df['EMA9'].iloc[0]
    vwap_slope = df['VWAP'].iloc[-1] - df['VWAP'].iloc[0]

    buffer = 0.001
    candle = df.iloc[-1]

    if ema_slope > 0:
        no_touch_ema9 = candle['low'] > candle['EMA9'] * (1 + buffer)
        no_touch_ema20 = candle['low'] > candle['EMA20'] * (1 + buffer)
        no_touch_vwap = candle['low'] > candle['VWAP'] * (1 + buffer)
        ema_gap = ((candle['EMA9'] - candle['EMA20']) / candle['EMA20']) * 100
    else:
        no_touch_ema9 = candle['high'] < candle['EMA9'] * (1 - buffer)
        no_touch_ema20 = candle['high'] < candle['EMA20'] * (1 - buffer)
        no_touch_vwap = candle['high'] < candle['VWAP'] * (1 - buffer)
        ema_gap = ((candle['EMA20'] - candle['EMA9']) / candle['EMA20']) * 100

    price = df['close'].iloc[-1]
    ema_dist_pct = ((price - df['EMA20'].iloc[-1]) / df['EMA20'].iloc[-1]) * 100
    vwap_dist_pct = ((price - df['VWAP'].iloc[-1]) / df['VWAP'].iloc[-1]) * 100

    return {
        "ema_up": bool(ema_slope > 0 and candle['low'] > candle['EMA20']),
        "ema_down": bool(ema_slope < 0 and candle['high'] < candle['EMA20']),
        "ema9_up": bool(ema9_slope > 0 and candle['low'] > candle['EMA9']),
        "ema9_down": bool(ema9_slope < 0 and candle['high'] < candle['EMA9']),
        "vwap_up": bool(vwap_slope > 0 and candle['low'] > candle['VWAP']),
        "vwap_down": bool(vwap_slope < 0 and candle['high'] < candle['VWAP']),
        "ema_strength": float(ema_slope),
        "ema9_strength": float(ema9_slope),
        "vwap_strength": float(vwap_slope),
        "ema_distance_pct": float(ema_dist_pct),
        "vwap_distance_pct": float(vwap_dist_pct),
        "ema_gap": float(ema_gap)
    }


def update_ltp(symbols):
    global LAST_LTP_UPDATE
    now = time.time()
    if now - LAST_LTP_UPDATE < LTP_REFRESH_INTERVAL:
        return
    instruments = [f"NSE:{s}" for s in symbols]
    ltp_data = kite.ltp(instruments)
    for s in symbols:
        key = f"NSE:{s}"
        if key in ltp_data:
            LTP_CACHE[s] = ltp_data[key]["last_price"]
    LAST_LTP_UPDATE = now


def get_signal(symbol, symbol_ltp):
    levels = STOCK_LEVELS.get(symbol)
    ltp = int(float(symbol_ltp))
    if levels is None or ltp is None:
        return None
    pdh = levels.get("pdh")
    pdl = levels.get("pdl")
    orb_high = levels.get("orb_high")
    orb_low = levels.get("orb_low")
    if None in (pdh, pdl, orb_high, orb_low):
        return None
    if ltp > pdh and ltp > orb_high:
        return "BUY"
    if ltp < pdl and ltp < orb_low:
        return "SELL"
    return None


def first_30min_breakout(df, range_minutes=30):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    today = df['date'].dt.date.max()
    df_today = df[df['date'].dt.date == today]
    start = pd.to_datetime("09:15")
    range_end = (start + pd.Timedelta(minutes=range_minutes)).time()

    range_df = df_today[
        (df_today['date'].dt.time >= start.time()) &
        (df_today['date'].dt.time <= range_end)
    ]

    if len(range_df) < 3:
        return {"buy": False, "sell": False, "time": None}

    range_high = range_df['high'].max()
    range_low = range_df['low'].min()
    post_df = df_today[df_today['date'].dt.time > range_end].reset_index(drop=True)

    if len(post_df) < 4:
        return {"buy": False, "sell": False, "time": None}

    for i in range(1, len(post_df)):
        candle = post_df.iloc[i]
        prev_candle = post_df.iloc[i - 1]
        body = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        is_bullish = candle['close'] > candle['open']
        is_bearish = candle['close'] < candle['open']
        if candle_range == 0:
            continue
        healthy = body > (0.4 * candle_range)
        vol_strong = candle['volume'] > (0.1 * prev_candle["volume"])
        close_near_high = (candle['high'] - candle['close']) < (0.50 * candle_range)
        close_near_low = (candle['close'] - candle['low']) < (0.50 * candle_range)

        if candle['close'] > range_high and healthy and is_bullish and vol_strong and close_near_high:
            return {"buy": True, "sell": False, "time": candle['date'].strftime("%H:%M")}
        if candle['close'] < range_low and healthy and is_bearish and vol_strong and close_near_low:
            return {"buy": False, "sell": True, "time": candle['date'].strftime("%H:%M")}

    return {"buy": False, "sell": False, "time": None}


# Load allowed stocks from CSV
ALLOWED_STOCKS = set()


def load_allowed_stocks():
    global ALLOWED_STOCKS

    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nse_derivative_stocks_FullList.csv"
    )

    if not os.path.exists(csv_path):
        print(f"CSV FILE NOT FOUND: {csv_path}")
        return

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            if not reader.fieldnames:
                print("CSV has no headers")
                return

            first_column = reader.fieldnames[0]

            for row in reader:
                value = str(row[first_column]).strip().upper()

                if value:
                    ALLOWED_STOCKS.add(value)

        print(f"Loaded {len(ALLOWED_STOCKS)} stocks")

    except Exception as e:
        print("CSV Load Error:", e)


load_allowed_stocks()

app = Flask(__name__)
session = requests.Session()

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/"
}


def initialize_session():
    session.get("https://www.nseindia.com", headers=BASE_HEADERS)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/api/indices')
def get_indices():
    try:
        index_type = request.args.get('type', 'Sectoral Indices')
        initialize_session()
        url = f"https://www.nseindia.com/api/heatmap-index?type={index_type}"
        response = session.get(url, headers=BASE_HEADERS, timeout=10)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/stocks')
def get_stocks():
    try:
        index = request.args.get('index')
        index_type = request.args.get('type', 'Sectoral Indices')
        breakout_only = request.args.get('breakout_only', 'false').lower() == 'true'

        initialize_session()
        url = f"https://www.nseindia.com/api/heatmap-symbols?type={index_type}&indices={index}"
        response = session.get(url, headers=BASE_HEADERS, timeout=10)
        if response.status_code != 200:
            return jsonify({
                "error": f"NSE API failed with status {response.status_code}"
            }), 500
        
        try:
            stocks = response.json()
        except Exception:
            return jsonify({
                "error": "NSE returned invalid JSON"
            }), 500

        symbols = []
        for stock in stocks:
            symbol = stock["symbol"].strip().upper()
            if symbol in ALLOWED_STOCKS:
                symbols.append(symbol)

        valid_symbols = []

        for s in symbols:
            try:
                if s not in STOCK_LEVELS:
                    success = prepare_stock_levels(s)
        
                    if success:
                        valid_symbols.append(s)
                else:
                    valid_symbols.append(s)
        
            except Exception as e:
                print(f"Error preparing {s}: {e}")

        filtered_stocks = []

        for stock in stocks:
            symbol = stock["symbol"].strip().upper()
            if symbol in valid_symbols and symbol in STOCK_LEVELS:
                signal = get_signal(symbol, stock['lastPrice'])
                if signal:
                    scan_result = scan_market_and_update(symbol, STOCK_LEVELS[symbol]["token"])
                    if scan_result:
                        STOCK_LEVELS[symbol]["scanner_time"] = scan_result["time"]

                combined_price, deviation = combined_deviation_simple(
                    float(stock['lastPrice']), STOCK_LEVELS[symbol]['curr_vpoc'], float(stock['vwap']))
                combined, dev = combined_deviation(
                    float(stock['lastPrice']), STOCK_LEVELS[symbol]['curr_vpoc'], float(stock['vwap']))
                pivotrangeBr = pivot_break_from_df(
                    STOCK_LEVELS[symbol]["cdh"], STOCK_LEVELS[symbol]["cdl"],
                    STOCK_LEVELS[symbol]["R1"], STOCK_LEVELS[symbol]["S1"],
                    STOCK_LEVELS[symbol]["R2"], STOCK_LEVELS[symbol]["S2"])

                overall_slope = False
                if (STOCK_LEVELS[symbol]["dfslope"]["ema_up"] and
                        STOCK_LEVELS[symbol]["dfslope"]["ema9_up"] and
                        STOCK_LEVELS[symbol]["dfslope"]["vwap_up"]):
                    overall_slope = True
                elif (STOCK_LEVELS[symbol]["dfslope"]["ema_down"] and
                      STOCK_LEVELS[symbol]["dfslope"]["ema9_down"] and
                      STOCK_LEVELS[symbol]["dfslope"]["vwap_down"]):
                    overall_slope = True

                signalVPOC = check_breakout(
                    STOCK_LEVELS[symbol]['last_close'], STOCK_LEVELS[symbol]['curr_vpoc'],
                    float(stock['vwap']), STOCK_LEVELS[symbol]['cprsignal'],
                    STOCK_LEVELS[symbol]["dfADX"], STOCK_LEVELS[symbol]["dfPlusDI"],
                    STOCK_LEVELS[symbol]["dfMinusDI"], STOCK_LEVELS[symbol]["df_DPO"],
                    STOCK_LEVELS[symbol]["df_MACD"], deviation, dev,
                    STOCK_LEVELS[symbol]["rsi_value"], STOCK_LEVELS[symbol]["dfNewsuper_1"],
                    STOCK_LEVELS[symbol]["dfNewsuper_2"], STOCK_LEVELS[symbol]["dfvolume"],
                    STOCK_LEVELS[symbol]["dfema"], STOCK_LEVELS[symbol]["macdSignal"],
                    STOCK_LEVELS[symbol]["orbBreakoutWithVol"],
                    STOCK_LEVELS[symbol]["volume20"]['cross'],
                    STOCK_LEVELS[symbol]["volume20"]['first_6_above_ma'],
                    overall_slope, symbol)

                stock["token"] = STOCK_LEVELS[symbol]["token"]
                stock["pdh"] = STOCK_LEVELS[symbol]["pdh"]
                stock["pdl"] = STOCK_LEVELS[symbol]["pdl"]
                stock["orb_high"] = STOCK_LEVELS[symbol]["orb_high"]
                stock["orb_low"] = STOCK_LEVELS[symbol]["orb_low"]
                stock["pdVol"] = STOCK_LEVELS[symbol]["pdVol"]
                stock["signal"] = signal
                stock["scanner_time"] = STOCK_LEVELS[symbol].get("scanner_time")
                stock["last_close"] = STOCK_LEVELS[symbol]["last_close"]
                stock["prev_vpoc"] = STOCK_LEVELS[symbol]["prev_vpoc"]
                stock["curr_vpoc"] = STOCK_LEVELS[symbol]["curr_vpoc"]
                stock["curmax_idx"] = STOCK_LEVELS[symbol]["curmax_idx"]
                stock["premax_idx"] = STOCK_LEVELS[symbol]["premax_idx"]
                stock["curwidth"] = STOCK_LEVELS[symbol]["curwidth"]
                stock["df_DPO"] = STOCK_LEVELS[symbol]["df_DPO"]
                stock["df_MACD"] = STOCK_LEVELS[symbol]["df_MACD"]
                stock["signalVPOC"] = signalVPOC
                stock["deviation"] = deviation
                stock["dfADX"] = STOCK_LEVELS[symbol]["dfADX"]
                stock["dfPlusDI"] = STOCK_LEVELS[symbol]["dfPlusDI"]
                stock["dfMinusDI"] = STOCK_LEVELS[symbol]["dfMinusDI"]
                stock["dev"] = dev
                stock["cpr"] = STOCK_LEVELS[symbol]['cprsignal']
                stock["rsi"] = STOCK_LEVELS[symbol]["rsi_value"]
                stock["VolumeIncr"] = STOCK_LEVELS[symbol]["dfvolume"]
                stock["R1_break"] = pivotrangeBr['R1_break']
                stock["R2_break"] = pivotrangeBr['R2_break']
                stock["S1_break"] = pivotrangeBr['S1_break']
                stock["S2_break"] = pivotrangeBr['S2_break']
                stock["macdSignalBuy"] = STOCK_LEVELS[symbol]["macdSignal"]['buy_signal']
                stock["macdSignalSell"] = STOCK_LEVELS[symbol]["macdSignal"]['sell_signal']
                stock["breakOT"] = STOCK_LEVELS[symbol]["orbBreakoutWithVol"]['time']
                stock["breakOT_15"] = STOCK_LEVELS[symbol]["orbBreakoutWithVol_15"]['time']
                stock["volume20"] = STOCK_LEVELS[symbol]["volume20"]['cross']
                stock["volume20Time"] = STOCK_LEVELS[symbol]["volume20"]['time']
                stock["volume20Count"] = STOCK_LEVELS[symbol]["volume20"]['cross_count']
                stock["cross_count_today"] = STOCK_LEVELS[symbol]["volume20"]['cross_count_today']
                stock["cross_times_today"] = STOCK_LEVELS[symbol]["volume20"]['cross_times_today']
                stock["first_4_above_ma"] = STOCK_LEVELS[symbol]["volume20"]['first_4_above_ma']
                stock["first_6_above_ma"] = STOCK_LEVELS[symbol]["volume20"]['first_6_above_ma']
                stock["ema9_strength"] = STOCK_LEVELS[symbol]["dfslope"]['ema9_strength']
                stock["ema20_strength"] = STOCK_LEVELS[symbol]["dfslope"]['ema_strength']
                stock["vwap_strength"] = STOCK_LEVELS[symbol]["dfslope"]['vwap_strength']
                stock["ema_distance_pct"] = STOCK_LEVELS[symbol]["dfslope"]['ema_distance_pct']
                stock["vwap_distance_pct"] = STOCK_LEVELS[symbol]["dfslope"]['vwap_distance_pct']
                stock["ema_gap"] = STOCK_LEVELS[symbol]["dfslope"]['ema_gap']
                stock["overall_slope"] = overall_slope

                if breakout_only:
                    if signal is not None and STOCK_LEVELS[symbol]['cprsignal'] == "NAR":
                        filtered_stocks.append(stock)
                else:
                    filtered_stocks.append(stock)

        return jsonify(filtered_stocks)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.errorhandler(Exception)
def handle_exception(e):
    print("GLOBAL ERROR:", str(e))
    return jsonify({
        "error": str(e)
    }), 500
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
