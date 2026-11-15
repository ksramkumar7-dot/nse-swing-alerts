import argparse
import html
import io
import os
import time
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

# Reads from GitHub Secrets automatically (or paste locally for offline tests)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

MIN_PRICE = 60.0
MIN_AVG_TURNOVER_CR = 15.0
MIN_SCORE = 62
MIN_RISK_REWARD = 1.5
LOOKBACK_PERIOD = "9mo"

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


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()
    df["VOL_SMA20"] = v.rolling(20).mean()

    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

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
                return True, f"4Pt HalfBat BO (AB:{r_ab:.2f}|BC:{r_bc:.2f})", 25, x_p, x_p + 0.272 * xa, c_p
            if days_c <= 4 and cmp <= c_p * 1.05:
                fib_err = abs(r_ab - 0.50) + abs(r_bc - 0.886)
                pts = max(16, int(25 - fib_err * 40))
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
                    pts = max(14, int(22 - abs(r_ab - ideal) * 80))
                    return True, f"3Pt HalfBat ({r_ab:.3f})", pts, a_p, a_p + 0.272 * xa, b_p

    return False, "", 0, None, None, None


def check_rsi_divergence(df: pd.DataFrame):
    sub = df.tail(38).copy().reset_index(drop=True)
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

    for idx1 in reversed(swing_lows[:-1]):
        if idx2 - idx1 < 4:
            continue
        p1, r1 = float(lows[idx1]), float(rsis[idx1])
        if (p2 <= p1 * 1.005) and (r2 >= r1 + 2.5) and (r1 < 46) and (float(rsis[-1]) > r2):
            return True, f"Reg Bullish RSI Div ({r1:.0f}->{r2:.0f})", 24
        if (curr_close > ema50) and (p2 > p1 * 1.005) and (r2 <= r1 - 2.0) and (float(rsis[-1]) > r2):
            return True, f"Hidden Bullish RSI Div ({r1:.0f}->{r2:.0f})", 22

    return False, "", 0


def check_compression_breakout(df: pd.DataFrame):
    curr, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    tags, pts = [], 0
    if (prev["High"] <= prev2["High"]) and (prev["Low"] >= prev2["Low"]) and (curr["Close"] > prev["High"]):
        tags.append("InsideBar BO")
        pts += 10
    ranges_7 = df["RANGE"].iloc[-8:-1]
    if len(ranges_7) == 7 and prev["RANGE"] <= ranges_7.min() and curr["Close"] > prev["High"]:
        tags.append("NR7 Expansion")
        pts += 10
    if pd.notna(prev["BB_WIDTH_PCTL"]) and prev["BB_WIDTH_PCTL"] <= 0.25 and curr["Close"] > prev["High"]:
        tags.append("BB Squeeze")
        pts += 8
    return ", ".join(tags), min(15, pts)


def get_nifty_context():
    try:
        nifty = yf.download("^NSEI", period=LOOKBACK_PERIOD, interval="1d", auto_adjust=True, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty = compute_indicators(nifty.dropna())
        c, e20, e50 = float(nifty["Close"].iloc[-1]), float(nifty["EMA20"].iloc[-1]), float(nifty["EMA50"].iloc[-1])
        ret_20d = float((c / nifty["Close"].iloc[-21] - 1) * 100)
        if c > e20 and e20 > e50:
            regime = "BULLISH (100% Size)"
        elif c > e50:
            regime = "CAUTIOUS (50-75% Size)"
        else:
            regime = "BEARISH (25-50% Size)"
        return {"close": round(c, 1), "ret_20d": ret_20d, "regime": regime}
    except Exception:
        return {"close": 0, "ret_20d": 0.0, "regime": "UNKNOWN"}


def analyze_stock(symbol: str, df: pd.DataFrame, nifty_ctx: dict):
    if df is None or len(df) < 90:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 90:
        return None

    df = compute_indicators(df)
    curr, prev = df.iloc[-1], df.iloc[-2]
    cmp = float(curr["Close"])
    if cmp < MIN_PRICE or pd.isna(curr["ATR"]) or pd.isna(curr["RSI"]):
        return None

    avg_turnover_cr = float((df["Close"].tail(20) * df["Volume"].tail(20)).mean() / 1e7)
    if avg_turnover_cr < MIN_AVG_TURNOVER_CR:
        return None

    c_range = float(curr["High"] - curr["Low"])
    if c_range <= 0:
        return None
    close_pos = (cmp - float(curr["Low"])) / c_range
    if not (cmp > float(curr["Open"]) and close_pos >= 0.55):
        return None

    vol_ratio = float(curr["Volume"] / curr["VOL_SMA20"]) if curr["VOL_SMA20"] > 0 else 1.0
    if vol_ratio < 1.05:
        return None

    atr, rsi = float(curr["ATR"]), float(curr["RSI"])
    pivots = detect_zigzag_pivots(df, reversal_pct=0.045)
    hb_ok, hb_tag, hb_pts, t1_hb, t2_hb, sl_hb = check_half_bat(df, pivots)
    div_ok, div_tag, div_pts = check_rsi_divergence(df)
    comp_tag, comp_pts = check_compression_breakout(df)

    if not (hb_ok or div_ok):
        return None

    stock_20d = float((cmp / df["Close"].iloc[-21] - 1) * 100)
    rs_diff = stock_20d - nifty_ctx["ret_20d"]
    rs_pts = 10 if rs_diff >= 3.0 else (6 if rs_diff >= 0 else 0)

    trend_pts = (5 if cmp > float(curr["EMA200"]) else 0) + (4 if float(curr["EMA20"]) > float(curr["EMA50"]) else 0)
    if cmp > float(curr["EMA9"]) and float(curr["MACD_HIST"]) > float(prev["MACD_HIST"]):
        trend_pts += 4

    vol_close_pts = (8 if close_pos >= 0.75 else (5 if close_pos >= 0.65 else 0)) + (8 if vol_ratio >= 1.5 else (5 if vol_ratio >= 1.2 else 0))
    confluence_bonus = 10 if (hb_ok and div_ok) else 0
    total_score = min(100, hb_pts + div_pts + comp_pts + rs_pts + trend_pts + vol_close_pts + confluence_bonus)

    if total_score < MIN_SCORE:
        return None

    tick = lambda x: round(round(x * 20) / 20, 2)
    buy_above = tick(float(curr["High"]) * 1.0005)
    max_buy = tick(buy_above * 1.02)
    recent_low = float(df["Low"].tail(7).min())
    sl = tick(min(sl_hb if sl_hb is not None else recent_low, recent_low) - 0.35 * atr)
    sl = max(sl, tick(buy_above - 2.3 * atr))

    risk = buy_above - sl
    if risk <= 0:
        return None

    if t1_hb and t1_hb > buy_above + 1.5 * risk:
        t1 = tick(t1_hb)
        t2 = tick(t2_hb) if t2_hb else tick(buy_above + 2.8 * risk)
    else:
        t1 = tick(buy_above + 1.8 * risk)
        t2 = tick(buy_above + 3.0 * risk)

    rr = round((t1 - buy_above) / risk, 2)
    if rr < MIN_RISK_REWARD:
        return None

    signals = [s for s in [hb_tag, div_tag, comp_tag] if s]
    if rs_diff >= 3.0:
        signals.append(f"RS+{rs_diff:.1f}%")

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
        "RSI": round(rsi, 1),
        "Vol_x": round(vol_ratio, 2),
        "ClosePos%": int(close_pos * 100),
        "Turnover_Cr": round(avg_turnover_cr, 1),
        "Confluence_Triggers": " + ".join(signals),
    }


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


def run_scan(symbols_subset=None):
    nifty_ctx = get_nifty_context()
    symbols = symbols_subset if symbols_subset else get_nse_symbols()
    tickers = [f"{s}.NS" for s in symbols]
    results = []

    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            data = yf.download(batch, period=LOOKBACK_PERIOD, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
        except Exception:
            continue
        for t in batch:
            try:
                df_s = data if len(batch) == 1 else data[t]
                res = analyze_stock(t.replace(".NS", ""), df_s, nifty_ctx)
                if res:
                    results.append(res)
            except Exception:
                continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by=["Score", "RR", "Vol_x"], ascending=[False, False, False]).reset_index(drop=True)
    return res_df, nifty_ctx


def send_telegram_html(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Skipping Telegram alert.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
        if resp.status_code == 200:
            print("[TELEGRAM] Alert sent successfully!")
        else:
            print(f"[TELEGRAM ERROR] {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")


def export_and_share(res_df: pd.DataFrame, nifty_ctx: dict):
    now_ist = datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    is_preclose = (now_ist.hour == 15 and now_ist.minute < 40)

    header_title = (
        f"⏳ <b>3:15 PM PRE-CLOSE CAPTURE ({today_str})</b>"
        if is_preclose
        else f"🚀 <b>4:00 PM FINAL: TOMORROW'S BUY ({today_str})</b>"
    )

    if res_df.empty:
        print("[RESULT] No A+ setups matched today.")
        send_telegram_html(
            f"{header_title}\n"
            f"📈 Nifty: <code>{nifty_ctx['close']}</code> | <code>{html.escape(nifty_ctx['regime'])}</code>\n\n"
            f"No A+ HalfBat / RSI Divergence setups passed all filters today. Cash is also a position!"
        )
        return

    csv_file = f"tmrw_buy_watchlist_{today_str}.csv"
    html_file = f"tmrw_buy_report_{today_str}.html"
    res_df.to_csv(csv_file, index=False)

    top_html = res_df.head(15).to_html(index=False, border=0)
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(f"<html><body><h2>Swing Watchlist {today_str}</h2><p>Nifty: {nifty_ctx['close']} | {nifty_ctx['regime']}</p>{top_html}</body></html>")

    print(res_df.head(15).to_string(index=False))

    lines = [
        header_title,
        f"📈 Nifty: <code>{nifty_ctx['close']}</code> | <code>{html.escape(nifty_ctx['regime'])}</code>",
        "<i>Rule: Buy tomorrow after 9:20 AM only above Buy_Above (skip if &gt; Max_Chase)</i>",
        ""
    ]
    for _, row in res_df.head(8).iterrows():
        sym = html.escape(str(row["Symbol"]))
        trig = html.escape(str(row["Confluence_Triggers"]))
        lines.append(
            f"<b>{sym}</b> | Score: <code>{row['Score']}</code> | RR: <code>1:{row['RR']}</code>\n"
            f"Buy &gt; <code>{row['Buy_Above']}</code> (Max <code>{row['Max_Chase']}</code>) | SL: <code>{row['SL']}</code>\n"
            f"T1: <code>{row['Target_1']}</code> | T2: <code>{row['Target_2']}</code>\n"
            f"<i>{trig}</i>\n"
        )
    send_telegram_html("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Run immediately (used by Cloud/GitHub Actions)")
    args = parser.parse_args()

    res_df, nifty_ctx = run_scan()
    export_and_share(res_df, nifty_ctx)


if __name__ == "__main__":
    main()