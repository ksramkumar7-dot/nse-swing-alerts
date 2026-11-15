import argparse
import html
import io
import os
import warnings
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IST = timezone(timedelta(hours=5, minutes=30))

INDEX_URL_PRIMARY = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
INDEX_URL_BACKUP = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

HISTORY_DIR = "history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "bot_signals_log.csv")

MIN_PRICE = 60.0
MIN_AVG_TURNOVER_CR = 15.0
BASE_MIN_SCORE = 62
MIN_RISK_REWARD = 1.5
# 5y ensures >= 60 Monthly bars & >= 260 Weekly bars for accurate 14-period MTF RSI
LOOKBACK_PERIOD = "5y"

FALLBACK_NSE_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "M&M", "TATAMOTORS", "TATASTEEL", "POWERGRID", "NTPC", "ONGC", "COALINDIA",
    "JSWSTEEL", "ADANIPORTS", "ADANIENT", "BAJAJFINSV", "HCLTECH", "TECHM",
    "HINDALCO", "GRASIM", "CIPLA", "DRREDDY", "DIVISLAB", "EICHERMOT", "HEROMOTOCO",
    "APOLLOHOSP", "BRITANNIA", "INDUSINDBK", "BPCL", "IOC", "BEL", "HAL", "TRENT",
    "ZOMATO", "DLF", "CHOLAFIN", "BANKBARODA", "PNB", "CANBK", "PFC", "RECLTD",
    "BHEL", "SIEMENS", "ABB", "CUMMINSIND", "CGPOWER", "PERSISTENT", "COFORGE",
    "LTIM", "MPHASIS", "POLYCAB", "KEI", "DIXON", "VBL", "PIDILITIND", "BERGEPAINT",
    "HAVELLS", "VOLTAS", "BLUESTARCO", "AUROPHARMA", "LUPIN", "BIOCON", "MAXHEALTH",
    "INDIGO", "IRCTC", "CONCOR", "JINDALSTEL", "SAIL", "NMDC", "VEDL", "TATAPOWER",
    "NHPC", "SUZLON", "ASHOKLEY", "TVSMOTOR", "BHARATFORG", "MOTHERSON", "EXIDEIND",
    "ABCAPITAL", "MUTHOOTFIN", "SBILIFE", "HDFCLIFE", "ICICIPRULI", "MARICO",
    "DABUR", "GODREJCP", "COLPAL", "TATACONSUM", "PAGEIND", "KPITTECH", "CDSL",
    "MCX", "BSE", "KAYNES", "TATATECH"
]


# ==============================================================================
# INDICATOR & MTF HELPERS
# ==============================================================================
def calc_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    return rsi


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df_clean = df.copy()
    if getattr(df_clean.index, "tz", None) is not None:
        df_clean.index = df_clean.index.tz_localize(None)
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    try:
        return df_clean.resample(rule).agg(agg).dropna()
    except ValueError:
        fallback = {"ME": "M", "W-FRI": "W"}.get(rule, rule)
        return df_clean.resample(fallback).agg(agg).dropna()


def compute_mtf_metrics(df: pd.DataFrame) -> dict:
    """Computes Weekly (1W) and Monthly (1M) RSI & Trend Status from daily OHLCV."""
    out = {
        "rsi_1w": np.nan, "w_above_ema20": False, "w_above_ema50": False,
        "rsi_1m": np.nan, "m_above_ema12": False,
    }

    df_w = resample_ohlcv(df, "W-FRI")
    if len(df_w) >= 16:
        w_rsi = calc_rsi_series(df_w["Close"], 14)
        w_ema20 = df_w["Close"].ewm(span=20, adjust=False).mean()
        w_ema50 = df_w["Close"].ewm(span=50, adjust=False).mean()
        out["rsi_1w"] = float(w_rsi.iloc[-1]) if pd.notna(w_rsi.iloc[-1]) else np.nan
        out["w_above_ema20"] = bool(df_w["Close"].iloc[-1] > w_ema20.iloc[-1])
        out["w_above_ema50"] = bool(df_w["Close"].iloc[-1] > w_ema50.iloc[-1])

    df_m = resample_ohlcv(df, "ME")
    if len(df_m) >= 15:
        m_rsi = calc_rsi_series(df_m["Close"], 14)
        m_ema12 = df_m["Close"].ewm(span=12, adjust=False).mean()
        out["rsi_1m"] = float(m_rsi.iloc[-1]) if pd.notna(m_rsi.iloc[-1]) else np.nan
        out["m_above_ema12"] = bool(df_m["Close"].iloc[-1] > m_ema12.iloc[-1])

    return out


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()
    df["VOL_SMA20"] = v.rolling(20).mean()
    df["HIGH_52W"] = h.rolling(252, min_periods=60).max()
    df["HIGH_20D"] = h.rolling(20).max()

    df["RSI"] = calc_rsi_series(c, 14)

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # ADX(14), +DI, -DI
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100.0 * (plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / tr14)
    minus_di = 100.0 * (minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / tr14)
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    df["ADX"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["MACD_HIST"] = macd - macd.ewm(span=9, adjust=False).mean()

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_WIDTH"] = (4 * std20) / sma20.replace(0, np.nan)
    df["BB_WIDTH_PCTL"] = df["BB_WIDTH"].rolling(60).rank(pct=True)
    df["RANGE"] = h - l
    return df


# ==============================================================================
# PATTERN & CONFLUENCE DETECTORS
# ==============================================================================
def detect_zigzag_pivots(df: pd.DataFrame, reversal_pct: float = 0.045):
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    if n < 30:
        return []

    pivots = []
    init_hi, init_lo = int(np.argmax(highs[:10])), int(np.argmin(lows[:10]))
    if init_lo < init_hi:
        pivots.append((init_lo, float(lows[init_lo]), "L"))
        trend, last_idx, last_p = 1, init_hi, float(highs[init_hi])
    else:
        pivots.append((init_hi, float(highs[init_hi]), "H"))
        trend, last_idx, last_p = -1, init_lo, float(lows[init_lo])

    for i in range(max(init_hi, init_lo) + 1, n):
        hi, lo = float(highs[i]), float(lows[i])
        if trend == 1:
            if hi > last_p:
                last_p, last_idx = hi, i
            elif (last_p - lo) / last_p >= reversal_pct:
                pivots.append((last_idx, last_p, "H"))
                trend, last_idx, last_p = -1, i, lo
        else:
            if lo < last_p:
                last_p, last_idx = lo, i
            elif (hi - last_p) / last_p >= reversal_pct:
                pivots.append((last_idx, last_p, "L"))
                trend, last_idx, last_p = 1, i, hi

    pivots.append((last_idx, last_p, "H" if trend == 1 else "L"))
    return pivots


def check_half_bat(df: pd.DataFrame, pivots: list):
    n = len(df)
    cmp, prev_c = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])

    for offset in [0, 1]:
        if len(pivots) < 4 + offset:
            continue
        win = pivots[-(4 + offset): len(pivots) - offset]
        (_, x_p, x_t), (_, a_p, a_t), (_, b_p, b_t), (c_i, c_p, c_t) = win
        if (x_t, a_t, b_t, c_t) != ("H", "L", "H", "L"):
            continue

        xa, ab = x_p - a_p, b_p - a_p
        if xa <= 0 or (xa / x_p) < 0.06 or ab <= 0:
            continue

        r_ab, r_bc = ab / xa, (b_p - c_p) / ab
        days_c = (n - 1) - c_i

        if (0.35 <= r_ab <= 0.65) and (0.70 <= r_bc <= 0.92) and (c_p >= a_p * 0.99):
            if prev_c <= b_p < cmp and cmp <= b_p * 1.04:
                return True, f"4Pt HalfBat BO (AB:{r_ab:.2f}|BC:{r_bc:.2f})", 24, x_p, x_p + 0.272 * xa, c_p
            if days_c <= 4 and cmp <= c_p * 1.05:
                fib_err = abs(r_ab - 0.50) + abs(r_bc - 0.886)
                pts = max(15, int(24 - fib_err * 40))
                return True, f"4Pt HalfBat C-PRZ (AB:{r_ab:.2f}|BC:{r_bc:.2f})", pts, b_p, a_p + 0.886 * xa, min(c_p, a_p)

    if len(pivots) >= 3:
        (_, x_p, x_t), (_, a_p, a_t), (b_i, b_p, b_t) = pivots[-3:]
        if (x_t, a_t, b_t) == ("L", "H", "L"):
            xa = a_p - x_p
            if xa > 0 and (xa / x_p) >= 0.08 and b_p > x_p:
                r_ab = (a_p - b_p) / xa
                days_b = (n - 1) - b_i
                in_mid = 0.48 <= r_ab <= 0.64
                in_deep = 0.76 <= r_ab <= 0.90
                if (in_mid or in_deep) and days_b <= 4 and cmp <= b_p * 1.05:
                    ideal = 0.618 if in_mid else 0.886
                    pts = max(14, int(21 - abs(r_ab - ideal) * 80))
                    return True, f"3Pt HalfBat ({r_ab:.3f})", pts, a_p, a_p + 0.272 * xa, b_p

    return False, "", 0, None, None, None


def check_rsi_divergence(df: pd.DataFrame):
    sub = df.tail(40).copy().reset_index(drop=True)
    if len(sub) < 25:
        return False, "", 0

    lows, rsis = sub["Low"].values, sub["RSI"].values
    n = len(sub)
    swing_lows = []
    for i in range(2, n - 1):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and lows[i] <= lows[i + 1] and (i == n - 2 or lows[i] <= lows[i + 2]):
            if not np.isnan(rsis[i]):
                swing_lows.append(i)

    if len(swing_lows) < 2:
        return False, "", 0

    idx2 = swing_lows[-1]
    if (n - 1) - idx2 > 5:
        return False, "", 0

    p2, r2 = float(lows[idx2]), float(rsis[idx2])
    curr_close, ema50 = float(sub["Close"].iloc[-1]), float(sub["EMA50"].iloc[-1])
    # Ensure price didn't break below the second swing low and momentum is turning up
    if float(sub["Low"].iloc[-1]) < p2 * 0.995:
        return False, "", 0
    if float(sub["MACD_HIST"].iloc[-1]) <= float(sub["MACD_HIST"].iloc[-2]) and curr_close <= float(sub["EMA9"].iloc[-1]):
        return False, "", 0

    for idx1 in reversed(swing_lows[:-1]):
        if idx2 - idx1 < 4:
            continue
        p1, r1 = float(lows[idx1]), float(rsis[idx1])
        if (p2 <= p1 * 1.005) and (r2 >= r1 + 2.5) and (r1 < 46) and (float(rsis[-1]) > r2):
            return True, f"Reg Bullish RSI Div ({r1:.0f}->{r2:.0f})", 22
        if (curr_close > ema50) and (p2 > p1 * 1.005) and (r2 <= r1 - 2.0) and (float(rsis[-1]) > r2):
            return True, f"Hidden Bullish RSI Div ({r1:.0f}->{r2:.0f})", 20

    return False, "", 0


def check_compression_breakout(df: pd.DataFrame):
    curr, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    tags, pts = [], 0
    if (prev["High"] <= prev2["High"]) and (prev["Low"] >= prev2["Low"]) and (curr["Close"] > prev["High"]):
        tags.append("InsideBar BO")
        pts += 8
    ranges_7 = df["RANGE"].iloc[-8:-1]
    if len(ranges_7) == 7 and prev["RANGE"] <= ranges_7.min() and curr["Close"] > prev["High"]:
        tags.append("NR7 Expansion")
        pts += 8
    if pd.notna(prev["BB_WIDTH_PCTL"]) and prev["BB_WIDTH_PCTL"] <= 0.25 and curr["Close"] > prev["High"]:
        tags.append("BB Squeeze")
        pts += 7
    # Volume Dry-up before breakout (VCP characteristic)
    vol_dry = df["Volume"].iloc[-4:-1].min() < (df["VOL_SMA20"].iloc[-1] * 0.85)
    if tags and vol_dry:
        tags.append("VolDryUp")
        pts += 4
    return ", ".join(tags), min(15, pts)


# ==============================================================================
# MARKET CONTEXT & CORE STOCK ANALYZER
# ==============================================================================
def get_nifty_context():
    try:
        nifty = yf.download("^NSEI", period=LOOKBACK_PERIOD, interval="1d", auto_adjust=True, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty = compute_indicators(nifty.dropna())
        mtf = compute_mtf_metrics(nifty)
        c = float(nifty["Close"].iloc[-1])
        e20, e50 = float(nifty["EMA20"].iloc[-1]), float(nifty["EMA50"].iloc[-1])
        ret_20d = float((c / nifty["Close"].iloc[-21] - 1) * 100)
        rsi_1d = round(float(nifty["RSI"].iloc[-1]), 1)
        rsi_1w = round(mtf["rsi_1w"], 1) if pd.notna(mtf["rsi_1w"]) else None
        rsi_1m = round(mtf["rsi_1m"], 1) if pd.notna(mtf["rsi_1m"]) else None

        if c > e20 and e20 > e50:
            regime, min_score = "BULLISH (100% Size)", BASE_MIN_SCORE
        elif c > e50:
            regime, min_score = "CAUTIOUS (50-75% Size)", BASE_MIN_SCORE + 4
        else:
            regime, min_score = "BEARISH (25-50% Size)", BASE_MIN_SCORE + 10

        return {
            "close": round(c, 1),
            "ret_20d": ret_20d,
            "regime": regime,
            "min_score": min_score,
            "rsi_1d": rsi_1d,
            "rsi_1w": rsi_1w,
            "rsi_1m": rsi_1m,
        }
    except Exception:
        return {
            "close": 0, "ret_20d": 0.0, "regime": "UNKNOWN",
            "min_score": BASE_MIN_SCORE, "rsi_1d": None, "rsi_1w": None, "rsi_1m": None
        }


def analyze_stock(symbol: str, df: pd.DataFrame, nifty_ctx: dict, mode: str = "daily"):
    if df is None or len(df) < 100:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 100:
        return None

    # Use last ~1.5y for fast daily indicator computation, full df for MTF Weekly/Monthly
    df_daily = compute_indicators(df.tail(380))
    curr, prev = df_daily.iloc[-1], df_daily.iloc[-2]
    cmp = float(curr["Close"])
    if cmp < MIN_PRICE or pd.isna(curr["ATR"]) or pd.isna(curr["RSI"]) or pd.isna(curr["ADX"]):
        return None

    # 1. Liquidity Check
    avg_turnover_cr = float((df_daily["Close"].tail(20) * df_daily["Volume"].tail(20)).mean() / 1e7)
    if avg_turnover_cr < MIN_AVG_TURNOVER_CR:
        return None

    # 2. ATR% Volatility Normalization (avoid sluggish <1% or hyper-erratic >6.5%)
    atr = float(curr["ATR"])
    atr_pct = (atr / cmp) * 100.0
    if not (1.0 <= atr_pct <= 6.5):
        return None

    # 3. Candle Quality & Volume Confirmation
    c_range = float(curr["High"] - curr["Low"])
    if c_range <= 0:
        return None
    close_pos = (cmp - float(curr["Low"])) / c_range
    body_ratio = (cmp - float(curr["Open"])) / c_range
    if not (cmp > float(curr["Open"]) and close_pos >= 0.55 and body_ratio >= 0.35):
        return None

    vol_ratio = float(curr["Volume"] / curr["VOL_SMA20"]) if curr["VOL_SMA20"] > 0 else 1.0
    if vol_ratio < 1.05:
        return None
    # Reject erratic blow-off spike with weak body
    if vol_ratio > 6.5 and body_ratio < 0.60:
        return None

    # 4. Anti-Chasing Extension Check (not overstretched above EMA20)
    ema20 = float(curr["EMA20"])
    dist_ema20_pct = ((cmp - ema20) / ema20) * 100.0
    if dist_ema20_pct > 12.0 or (cmp - ema20) > 2.8 * atr:
        return None

    # 5. RSI (1D / 1W / 1M) & Overbought Guard
    rsi_1d = float(curr["RSI"])
    if rsi_1d > 76.0 or rsi_1d < 30.0:
        return None

    mtf = compute_mtf_metrics(df)
    rsi_1w = mtf["rsi_1w"]
    rsi_1m = mtf["rsi_1m"]
    if pd.notna(rsi_1w) and rsi_1w > 80.0:
        return None

    # 6. Core Strategy Triggers (Half-Bat / RSI Divergence + Compression)
    pivots = detect_zigzag_pivots(df_daily, reversal_pct=0.045)
    hb_ok, hb_tag, hb_pts, t1_hb, t2_hb, sl_hb = check_half_bat(df_daily, pivots)
    div_ok, div_tag, div_pts = check_rsi_divergence(df_daily)
    comp_tag, comp_pts = check_compression_breakout(df_daily)

    if not (hb_ok or div_ok):
        return None

    # 7. ADX & Directional Bias Check
    adx = float(curr["ADX"])
    plus_di = float(curr["PLUS_DI"])
    minus_di = float(curr["MINUS_DI"])
    if adx < 13.0:
        return None
    if (minus_di > plus_di + 6.0) and not div_ok:
        return None

    # 8. Relative Strength vs Nifty
    stock_20d = float((cmp / df_daily["Close"].iloc[-21] - 1) * 100)
    rs_diff = stock_20d - nifty_ctx["ret_20d"]
    if "BEARISH" in nifty_ctx["regime"] and rs_diff < 0:
        return None
    rs_pts = 8 if rs_diff >= 3.0 else (5 if rs_diff >= 0 else 0)

    # 9. MTF Alignment Scoring (1D, 1W, 1M)
    mtf_pts = 0
    mtf_flags = []
    if pd.notna(rsi_1w):
        if 48 <= rsi_1w <= 72 and mtf["w_above_ema20"]:
            mtf_pts += 7
            mtf_flags.append("1W-Bull")
        elif rsi_1w >= 42:
            mtf_pts += 3
    if pd.notna(rsi_1m):
        if rsi_1m >= 50 and mtf["m_above_ema12"]:
            mtf_pts += 6
            mtf_flags.append("1M-Bull")
        elif rsi_1m >= 45:
            mtf_pts += 3

    # In weekly/monthly report mode, enforce stronger higher-timeframe alignment
    if mode == "weekly" and pd.notna(rsi_1w) and rsi_1w < 46:
        return None
    if mode == "monthly" and ((pd.notna(rsi_1m) and rsi_1m < 50) or (pd.notna(rsi_1w) and rsi_1w < 50)):
        return None

    # 10. Trend, Momentum, ADX & 52W High Proximity Scoring
    ema200 = float(curr["EMA200"])
    ema50 = float(curr["EMA50"])
    trend_pts = (5 if cmp > ema200 else 0) + (4 if ema20 > ema50 else 0)
    if cmp > float(curr["EMA9"]) and float(curr["MACD_HIST"]) > float(prev["MACD_HIST"]):
        trend_pts += 3
    if plus_di > minus_di and adx >= 18:
        trend_pts += 4
    high_52w = float(curr["HIGH_52W"]) if pd.notna(curr["HIGH_52W"]) else cmp
    near_52w_pct = ((high_52w - cmp) / high_52w) * 100.0
    if near_52w_pct <= 15.0:
        trend_pts += 3

    vol_close_pts = (
        (7 if close_pos >= 0.75 else (4 if close_pos >= 0.65 else 0)) +
        (7 if vol_ratio >= 1.5 else (4 if vol_ratio >= 1.2 else 0))
    )
    confluence_bonus = 8 if (hb_ok and div_ok) else 0

    total_score = min(
        100,
        hb_pts + div_pts + comp_pts + rs_pts + mtf_pts + trend_pts + vol_close_pts + confluence_bonus
    )

    if total_score < nifty_ctx.get("min_score", BASE_MIN_SCORE):
        return None

    # 11. Trade Levels & Overhead Resistance / Clean Runway Check
    tick = lambda x: round(round(x * 20) / 20, 2)
    buy_above = tick(float(curr["High"]) * 1.0005)
    max_buy = tick(buy_above * 1.02)
    recent_low = float(df_daily["Low"].tail(7).min())
    sl = tick(min(sl_hb if sl_hb is not None else recent_low, recent_low) - 0.35 * atr)
    sl = max(sl, tick(buy_above - 2.3 * atr))
    sl = min(sl, tick(float(curr["Low"]) - 0.1 * atr))

    risk = buy_above - sl
    if risk <= 0:
        return None

    if t1_hb and t1_hb > buy_above + 1.5 * risk:
        t1 = tick(t1_hb)
        t2 = tick(t2_hb) if t2_hb else tick(buy_above + 2.8 * risk)
    else:
        t1 = tick(buy_above + 1.8 * risk)
        t2 = tick(buy_above + 3.0 * risk)

    # Overhead Resistance Guard: reject if EMA200 sits directly between entry & T1 blocking runway
    if cmp < ema200 and (ema200 - buy_above) < (1.0 * risk):
        return None

    rr = round((t1 - buy_above) / risk, 2)
    if rr < MIN_RISK_REWARD:
        return None

    signals = [s for s in [hb_tag, div_tag, comp_tag] if s]
    if rs_diff >= 3.0:
        signals.append(f"RS+{rs_diff:.1f}%")
    if mtf_flags:
        signals.append("/".join(mtf_flags))

    return {
        "Symbol": symbol,
        "Score": total_score,
        "CMP": round(cmp, 2),
        "Buy_Above": buy_above,
        "Max_Chase": max_buy,
        "SL": sl,
        "Target_1": t1,
        "Target_2": t2,
        "RR": rr,
        "RSI_1D": round(rsi_1d, 1),
        "RSI_1W": round(rsi_1w, 1) if pd.notna(rsi_1w) else np.nan,
        "RSI_1M": round(rsi_1m, 1) if pd.notna(rsi_1m) else np.nan,
        "ADX": round(adx, 1),
        "ATR%": round(atr_pct, 2),
        "Vol_x": round(vol_ratio, 2),
        "ClosePos%": int(close_pos * 100),
        "Turnover_Cr": round(avg_turnover_cr, 1),
        "Confluence_Triggers": " + ".join(signals),
    }


# ==============================================================================
# DATA DOWNLOADER & SCANNER
# ==============================================================================
def get_nse_symbols():
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    for url in [INDEX_URL_PRIMARY, INDEX_URL_BACKUP]:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200 and "Symbol" in r.text:
                df_i = pd.read_csv(io.StringIO(r.text))
                syms = [s.strip() for s in df_i["Symbol"].dropna().astype(str) if s.isascii() and "DUMMY" not in s.upper()]
                if len(syms) >= 50:
                    return syms
        except Exception:
            pass
    return FALLBACK_NSE_SYMBOLS


def extract_ticker_df(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker]
        return pd.DataFrame()
    return data


def run_scan(symbols_subset=None, mode: str = "daily"):
    nifty_ctx = get_nifty_context()
    symbols = symbols_subset if symbols_subset else get_nse_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    results = []

    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            data = yf.download(
                batch, period=LOOKBACK_PERIOD, interval="1d",
                group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
        except Exception:
            continue
        for t in batch:
            try:
                df_s = extract_ticker_df(data, t)
                res = analyze_stock(t.replace(".NS", ""), df_s, nifty_ctx, mode=mode)
                if res:
                    results.append(res)
            except Exception:
                continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by=["Score", "RR", "Vol_x"], ascending=[False, False, False]).reset_index(drop=True)
    return res_df, nifty_ctx


# ==============================================================================
# SIGNAL LOGGING & WEEKLY / MONTHLY BOT SUGGESTION PERFORMANCE REPORTS
# ==============================================================================
def save_signal_history(res_df: pd.DataFrame, nifty_ctx: dict, signal_date: str):
    if res_df.empty:
        return
    os.makedirs(HISTORY_DIR, exist_ok=True)
    log_df = res_df.copy()
    log_df.insert(0, "Signal_Date", signal_date)
    log_df["Nifty_Regime"] = nifty_ctx["regime"]

    cols = [
        "Signal_Date", "Symbol", "Score", "CMP", "Buy_Above", "Max_Chase",
        "SL", "Target_1", "Target_2", "RR", "RSI_1D", "RSI_1W", "RSI_1M",
        "ADX", "ATR%", "Nifty_Regime", "Confluence_Triggers"
    ]
    log_df = log_df[cols]

    if os.path.exists(HISTORY_FILE):
        existing = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([existing, log_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Signal_Date", "Symbol"], keep="last")
        combined = combined.sort_values(by=["Signal_Date", "Score"], ascending=[True, False])
    else:
        combined = log_df

    combined.to_csv(HISTORY_FILE, index=False)
    print(f"[HISTORY] Logged {len(log_df)} signals for {signal_date} into {HISTORY_FILE}")


def evaluate_bot_suggestions(lookback_days: int) -> tuple[pd.DataFrame, dict]:
    """Audits logged bot suggestions over the past `lookback_days` using real OHLC data."""
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(), {"note": "No historical signals logged yet. Logging starts on 4:00 PM final runs."}

    hist = pd.read_csv(HISTORY_FILE)
    if hist.empty:
        return pd.DataFrame(), {"note": "Signal history file is empty."}

    hist["Signal_Date_dt"] = pd.to_datetime(hist["Signal_Date"], errors="coerce")
    cutoff = pd.Timestamp(datetime.now(IST).date() - timedelta(days=lookback_days))
    period_df = hist[hist["Signal_Date_dt"] >= cutoff].copy()

    if period_df.empty:
        return pd.DataFrame(), {"note": f"No logged signals in the past {lookback_days} days."}

    symbols = sorted(period_df["Symbol"].unique().tolist())
    tickers = [f"{s}.NS" for s in symbols]
    ohlc_map = {}

    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            data = yf.download(
                batch, period=f"{lookback_days + 15}d", interval="1d",
                group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
            for t in batch:
                df_t = extract_ticker_df(data, t).dropna()
                if not df_t.empty:
                    if getattr(df_t.index, "tz", None) is not None:
                        df_t.index = df_t.index.tz_localize(None)
                    ohlc_map[t.replace(".NS", "")] = df_t
        except Exception:
            continue

    eval_rows = []
    for _, row in period_df.iterrows():
        sym = row["Symbol"]
        df_s = ohlc_map.get(sym)
        sig_dt = row["Signal_Date_dt"]
        buy_above = float(row["Buy_Above"])
        max_chase = float(row["Max_Chase"])
        sl = float(row["SL"])
        t1 = float(row["Target_1"])
        t2 = float(row["Target_2"])

        if df_s is None or df_s.empty:
            continue

        after = df_s[df_s.index > sig_dt]
        if after.empty:
            eval_rows.append({**row.to_dict(), "Status": "PENDING_NEXT_DAY", "Entry": np.nan, "Exit_Or_CMP": np.nan, "Return%": 0.0, "Max_Gain%": 0.0})
            continue

        # Check entry within next 2 trading sessions per bot rule
        triggered = False
        entry_price = np.nan
        entry_idx = None
        for idx_pos, (bar_dt, bar) in enumerate(after.head(2).iterrows()):
            o, h, l = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
            if o > max_chase:
                break
            if h >= buy_above:
                triggered = True
                entry_price = max(o, buy_above)
                entry_idx = idx_pos
                break

        if not triggered:
            eval_rows.append({
                "Signal_Date": row["Signal_Date"], "Symbol": sym, "Score": row["Score"],
                "RSI_1D": row.get("RSI_1D", np.nan), "RSI_1W": row.get("RSI_1W", np.nan), "RSI_1M": row.get("RSI_1M", np.nan),
                "Status": "NOT_TRIGGERED", "Entry": np.nan,
                "Exit_Or_CMP": round(float(after["Close"].iloc[-1]), 2), "Return%": 0.0, "Max_Gain%": 0.0
            })
            continue

        trade_bars = after.iloc[entry_idx:]
        status = "OPEN"
        exit_price = float(trade_bars["Close"].iloc[-1])
        max_high = entry_price
        t1_hit = False

        for k, (_, bar) in enumerate(trade_bars.iterrows()):
            h, l = float(bar["High"]), float(bar["Low"])
            max_high = max(max_high, h)
            # Conservative intrabar check: SL checked before T1/T2
            if l <= sl:
                status = "T1_THEN_SL" if t1_hit else "SL_HIT"
                exit_price = sl
                break
            if h >= t2:
                status = "T2_HIT"
                exit_price = t2
                break
            if h >= t1:
                t1_hit = True
                status = "T1_HIT"
                exit_price = t1

        ret_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
        max_gain_pct = round(((max_high - entry_price) / entry_price) * 100.0, 2)

        eval_rows.append({
            "Signal_Date": row["Signal_Date"], "Symbol": sym, "Score": row["Score"],
            "RSI_1D": row.get("RSI_1D", np.nan), "RSI_1W": row.get("RSI_1W", np.nan), "RSI_1M": row.get("RSI_1M", np.nan),
            "Status": status, "Entry": round(entry_price, 2),
            "Exit_Or_CMP": round(exit_price, 2), "Return%": ret_pct, "Max_Gain%": max_gain_pct
        })

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        return eval_df, {"note": "Could not evaluate signals due to missing market data."}

    trig = eval_df[~eval_df["Status"].isin(["NOT_TRIGGERED", "PENDING_NEXT_DAY"])]
    wins = trig[trig["Status"].isin(["T1_HIT", "T2_HIT", "T1_THEN_SL"])]
    losses = trig[trig["Status"] == "SL_HIT"]
    closed = trig[trig["Status"].isin(["T1_HIT", "T2_HIT", "T1_THEN_SL", "SL_HIT"])]

    summary = {
        "total_signals": int(len(eval_df)),
        "triggered": int(len(trig)),
        "not_triggered": int((eval_df["Status"] == "NOT_TRIGGERED").sum()),
        "t2_hits": int((trig["Status"] == "T2_HIT").sum()),
        "t1_hits": int(trig["Status"].isin(["T1_HIT", "T1_THEN_SL"]).sum()),
        "sl_hits": int(len(losses)),
        "open_trades": int((trig["Status"] == "OPEN").sum()),
        "win_rate": round((len(wins) / len(closed)) * 100.0, 1) if len(closed) > 0 else 0.0,
        "avg_return": round(float(trig["Return%"].mean()), 2) if len(trig) > 0 else 0.0,
        "best_pick": f"{trig.loc[trig['Return%'].idxmax(), 'Symbol']} ({trig['Return%'].max():+.1f}%)" if len(trig) > 0 else "N/A",
        "worst_pick": f"{trig.loc[trig['Return%'].idxmin(), 'Symbol']} ({trig['Return%'].min():+.1f}%)" if len(trig) > 0 else "N/A",
    }
    return eval_df.sort_values(by=["Signal_Date", "Return%"], ascending=[False, False]), summary


# ==============================================================================
# TELEGRAM & REPORT EXPORT
# ==============================================================================
def fmt_rsi(val) -> str:
    return f"{val:.1f}" if pd.notna(val) else "N/A"


def send_telegram_html(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Skipping Telegram alert.")
        return

    # Auto-chunk messages to stay safely under Telegram's 4096-character limit
    chunks, current = [], []
    curr_len = 0
    for line in msg.split("\n"):
        if curr_len + len(line) + 1 > 3800:
            chunks.append("\n".join(current))
            current, curr_len = [line], len(line) + 1
        else:
            current.append(line)
            curr_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for idx, chunk in enumerate(chunks, 1):
        try:
            resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"}, timeout=15)
            if resp.status_code == 200:
                print(f"[TELEGRAM] Alert part {idx}/{len(chunks)} sent successfully!")
            else:
                print(f"[TELEGRAM ERROR] {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[TELEGRAM ERROR] {e}")


def export_daily_report(res_df: pd.DataFrame, nifty_ctx: dict, is_preclose: bool):
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")

    header_title = (
        f"⏳ <b>3:15 PM PRE-CLOSE CAPTURE ({today_str})</b>"
        if is_preclose
        else f"🚀 <b>4:00 PM FINAL: TOMORROW'S BUY ({today_str})</b>"
    )
    nifty_rsi_line = (
        f"📊 Nifty RSI (1D/1W/1M): <code>{fmt_rsi(nifty_ctx['rsi_1d'])}</code> / "
        f"<code>{fmt_rsi(nifty_ctx['rsi_1w'])}</code> / <code>{fmt_rsi(nifty_ctx['rsi_1m'])}</code>"
    )

    if res_df.empty:
        print("[RESULT] No A+ setups matched today.")
        send_telegram_html(
            f"{header_title}\n"
            f"📈 Nifty: <code>{nifty_ctx['close']}</code> | <code>{html.escape(nifty_ctx['regime'])}</code>\n"
            f"{nifty_rsi_line}\n\n"
            f"No A+ HalfBat / RSI Divergence setups passed all filters today. Cash is also a position!"
        )
        return

    if not is_preclose:
        save_signal_history(res_df, nifty_ctx, today_str)

    csv_file = f"tmrw_buy_watchlist_{today_str}.csv"
    html_file = f"tmrw_buy_report_{today_str}.html"
    res_df.to_csv(csv_file, index=False)

    top_html = res_df.head(20).to_html(index=False, border=0)
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(
            f"<html><body style='font-family:Arial,sans-serif;'>"
            f"<h2>Swing Watchlist {today_str}</h2>"
            f"<p><b>Nifty:</b> {nifty_ctx['close']} | {nifty_ctx['regime']} | "
            f"RSI (1D/1W/1M): {fmt_rsi(nifty_ctx['rsi_1d'])} / {fmt_rsi(nifty_ctx['rsi_1w'])} / {fmt_rsi(nifty_ctx['rsi_1m'])}</p>"
            f"{top_html}</body></html>"
        )

    print(res_df.head(15).to_string(index=False))

    lines = [
        header_title,
        f"📈 Nifty: <code>{nifty_ctx['close']}</code> | <code>{html.escape(nifty_ctx['regime'])}</code>",
        nifty_rsi_line,
        "<i>Rule: Buy tomorrow after 9:20 AM only above Buy_Above (skip if &gt; Max_Chase)</i>",
        ""
    ]
    for _, row in res_df.head(8).iterrows():
        sym = html.escape(str(row["Symbol"]))
        trig = html.escape(str(row["Confluence_Triggers"]))
        lines.append(
            f"<b>{sym}</b> | Score: <code>{row['Score']}</code> | RR: <code>1:{row['RR']}</code> | ADX: <code>{row['ADX']}</code>\n"
            f"📊 RSI (1D/1W/1M): <code>{fmt_rsi(row['RSI_1D'])}</code> / <code>{fmt_rsi(row['RSI_1W'])}</code> / <code>{fmt_rsi(row['RSI_1M'])}</code>\n"
            f"Buy &gt; <code>{row['Buy_Above']}</code> (Max <code>{row['Max_Chase']}</code>) | SL: <code>{row['SL']}</code>\n"
            f"T1: <code>{row['Target_1']}</code> | T2: <code>{row['Target_2']}</code>\n"
            f"<i>{trig}</i>\n"
        )
    send_telegram_html("\n".join(lines))


def export_periodic_report(mode: str):
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    lookback_days = 7 if mode == "weekly" else 30
    label = "WEEKLY" if mode == "weekly" else "MONTHLY"

    eval_df, summary = evaluate_bot_suggestions(lookback_days)
    res_df, nifty_ctx = run_scan(mode=mode)

    # Save CSV & HTML Artifacts
    html_file = f"{mode}_report_{today_str}.html"
    perf_csv = f"{mode}_bot_performance_{today_str}.csv"
    picks_csv = f"{mode}_mtf_picks_{today_str}.csv"

    if not eval_df.empty:
        eval_df.to_csv(perf_csv, index=False)
    if not res_df.empty:
        res_df.to_csv(picks_csv, index=False)

    perf_html = eval_df.to_html(index=False, border=0) if not eval_df.empty else f"<p>{summary.get('note', 'No historical data.')}</p>"
    picks_html = res_df.head(15).to_html(index=False, border=0) if not res_df.empty else "<p>No MTF setups matched.</p>"

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(
            f"<html><body style='font-family:Arial,sans-serif;'>"
            f"<h2>NSE Swing Master - {label} Report ({today_str})</h2>"
            f"<p><b>Nifty:</b> {nifty_ctx['close']} | {nifty_ctx['regime']} | "
            f"RSI (1D/1W/1M): {fmt_rsi(nifty_ctx['rsi_1d'])} / {fmt_rsi(nifty_ctx['rsi_1w'])} / {fmt_rsi(nifty_ctx['rsi_1m'])}</p>"
            f"<h3>1. Bot Suggestion Performance (Last {lookback_days} Days)</h3>"
            f"<p>{summary}</p>{perf_html}"
            f"<h3>2. Top {label} Multi-Timeframe Swing Picks</h3>"
            f"{picks_html}</body></html>"
        )

    # Telegram Summary
    lines = [
        f"📅 <b>NSE SWING MASTER — {label} REPORT ({today_str})</b>",
        f"📈 Nifty: <code>{nifty_ctx['close']}</code> | <code>{html.escape(nifty_ctx['regime'])}</code>",
        f"📊 Nifty RSI (1D/1W/1M): <code>{fmt_rsi(nifty_ctx['rsi_1d'])}</code> / <code>{fmt_rsi(nifty_ctx['rsi_1w'])}</code> / <code>{fmt_rsi(nifty_ctx['rsi_1m'])}</code>",
        "",
        f"🎯 <b>Bot Suggestion Audit (Last {lookback_days} Days)</b>"
    ]

    if "note" in summary:
        lines.append(f"<i>{html.escape(summary['note'])}</i>")
    else:
        lines.extend([
            f"Signals: <code>{summary['total_signals']}</code> | Triggered: <code>{summary['triggered']}</code> | Skipped: <code>{summary['not_triggered']}</code>",
            f"Wins (T1/T2): <code>{summary['t1_hits'] + summary['t2_hits']}</code> (T2: <code>{summary['t2_hits']}</code>) | SL: <code>{summary['sl_hits']}</code> | Open: <code>{summary['open_trades']}</code>",
            f"Win Rate: <code>{summary['win_rate']}%</code> | Avg Return: <code>{summary['avg_return']:+.2f}%</code>",
            f"🏆 Best: <code>{html.escape(summary['best_pick'])}</code> | ⚠️ Worst: <code>{html.escape(summary['worst_pick'])}</code>",
        ])
        print(f"\n=== {label} BOT PERFORMANCE ===")
        print(eval_df.head(15).to_string(index=False))

    lines.extend(["", f"🔥 <b>Top {label} MTF Aligned Setups</b>"])
    if res_df.empty:
        lines.append("<i>No stocks passed strict MTF filters in this cycle.</i>")
    else:
        print(f"\n=== {label} MTF PICKS ===")
        print(res_df.head(10).to_string(index=False))
        for _, row in res_df.head(6).iterrows():
            lines.append(
                f"<b>{html.escape(str(row['Symbol']))}</b> | Score: <code>{row['Score']}</code> | RR: <code>1:{row['RR']}</code>\n"
                f"RSI (1D/1W/1M): <code>{fmt_rsi(row['RSI_1D'])}</code> / <code>{fmt_rsi(row['RSI_1W'])}</code> / <code>{fmt_rsi(row['RSI_1M'])}</code>\n"
                f"Buy &gt; <code>{row['Buy_Above']}</code> | SL: <code>{row['SL']}</code> | T1: <code>{row['Target_1']}</code>\n"
            )

    send_telegram_html("\n".join(lines))


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run immediately")
    parser.add_argument(
        "--mode",
        choices=["auto", "daily", "weekly", "monthly"],
        default="auto",
        help="Report mode: daily scan, weekly report, monthly report, or auto-detect"
    )
    args = parser.parse_args()

    now_ist = datetime.now(IST)
    mode = args.mode
    if mode == "auto":
        mode = "daily"

    if mode == "daily":
        is_preclose = (now_ist.hour == 15 and now_ist.minute < 40)
        res_df, nifty_ctx = run_scan(mode="daily")
        export_daily_report(res_df, nifty_ctx, is_preclose=is_preclose)
    else:
        export_periodic_report(mode=mode)


if __name__ == "__main__":
    main()