"""
ICT (Inner Circle Trader) Forex Dashboard
==========================================
Real-time dashboard for EUR/USD and GBP/USD with ICT concepts:
- Daily Power of 3 (Midnight Open, 8:30 AM Open, Premium/Discount)
- Killzone Tracker (Asian, London, NY)
- Fair Value Gaps (FVG)
- Liquidity Sweeps (Asian Range, PDH/PDL)
- SMT Divergence (EUR/USD vs GBP/USD)
- Directional Bias Engine (H4 swing structure analysis)
- Draw on Liquidity (DOL – distance to PDH/PDL target)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time as dtime
import pytz

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

NY_TZ = pytz.timezone("America/New_York")

YFINANCE_SYMBOLS = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"}
TWELVEDATA_SYMBOLS = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD"}

TIMEFRAME_MAP_YF = {"5m": "5m", "15m": "15m", "1h": "60m", "4h": "60m"}
TIMEFRAME_MAP_TD = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}

# How many calendar days of history to request for each timeframe
PERIOD_DAYS = {"5m": 5, "15m": 14, "1h": 30, "4h": 60}


def fetch_data_yfinance(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch OHLC data via yfinance and convert to New York time."""
    import yfinance as yf

    ticker = YFINANCE_SYMBOLS[symbol]
    interval = TIMEFRAME_MAP_YF[timeframe]
    days = PERIOD_DAYS[timeframe]
    end = datetime.now(tz=NY_TZ)
    start = end - timedelta(days=days)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
    )
    if df.empty:
        return df

    # yfinance may return MultiIndex columns for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close"]].dropna()
    df.index.name = "datetime"
    return df


def fetch_data_twelvedata(symbol: str, timeframe: str, api_key: str) -> pd.DataFrame:
    """Fetch OHLC data via TwelveData and convert to New York time."""
    from twelvedata import TDClient

    td = TDClient(apikey=api_key)
    ts = (
        td.time_series(
            symbol=TWELVEDATA_SYMBOLS[symbol],
            interval=TIMEFRAME_MAP_TD[timeframe],
            outputsize=500,
            timezone="America/New_York",
        )
        .as_pandas()
    )
    if ts is None or ts.empty:
        return pd.DataFrame()

    ts.index = pd.to_datetime(ts.index)
    if ts.index.tz is None:
        ts.index = ts.index.tz_localize(NY_TZ)
    else:
        ts.index = ts.index.tz_convert(NY_TZ)

    ts = ts.rename(columns=str.lower)
    for col in ["open", "high", "low", "close"]:
        ts[col] = pd.to_numeric(ts[col], errors="coerce")
    ts = ts[["open", "high", "low", "close"]].dropna().sort_index()
    ts.index.name = "datetime"
    return ts


def _resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLC data to 4h bars."""
    if df.empty:
        return df
    resampled = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()
    resampled.index.name = "datetime"
    return resampled


def fetch_data(symbol: str, timeframe: str, source: str, api_key: str = "") -> pd.DataFrame:
    if source == "TwelveData":
        return fetch_data_twelvedata(symbol, timeframe, api_key)
    if timeframe == "4h":
        # yfinance has no native 4h interval; fetch 1h and resample
        df_1h = fetch_data_yfinance(symbol, "4h")  # uses 60m interval with 60-day window
        return _resample_to_4h(df_1h)
    return fetch_data_yfinance(symbol, timeframe)


# ---------------------------------------------------------------------------
# ICT Logic – Daily Power of 3
# ---------------------------------------------------------------------------

def get_midnight_open(df: pd.DataFrame) -> float | None:
    """Return the open price of the candle closest to 00:00 EST today."""
    now_ny = datetime.now(tz=NY_TZ)
    midnight = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    window = df.loc[midnight - timedelta(minutes=15): midnight + timedelta(minutes=15)]
    if window.empty:
        # Fallback: use previous trading day's midnight
        prev_midnight = midnight - timedelta(days=1)
        window = df.loc[prev_midnight - timedelta(minutes=15): prev_midnight + timedelta(minutes=15)]
    if window.empty:
        return None
    return float(window.iloc[0]["open"])


def get_830_open(df: pd.DataFrame) -> float | None:
    """Return the open price of the candle closest to 08:30 EST today."""
    now_ny = datetime.now(tz=NY_TZ)
    target = now_ny.replace(hour=8, minute=30, second=0, microsecond=0)
    if now_ny < target:
        target -= timedelta(days=1)
    window = df.loc[target - timedelta(minutes=15): target + timedelta(minutes=15)]
    if window.empty:
        return None
    return float(window.iloc[0]["open"])


def premium_discount_label(price: float, reference: float) -> str:
    if price > reference:
        return "Premium"
    elif price < reference:
        return "Discount"
    return "At Open"


# ---------------------------------------------------------------------------
# ICT Logic – Killzone Tracker
# ---------------------------------------------------------------------------

KILLZONES = {
    "Asian Range": (dtime(20, 0), dtime(0, 0)),
    "London Killzone": (dtime(2, 0), dtime(5, 0)),
    "NY Killzone": (dtime(7, 0), dtime(10, 0)),
}


def current_killzone() -> str | None:
    now = datetime.now(tz=NY_TZ).time()
    for name, (start, end) in KILLZONES.items():
        if start > end:  # wraps midnight
            if now >= start or now < end:
                return name
        else:
            if start <= now < end:
                return name
    return None


def killzone_status() -> dict[str, str]:
    """Return status for each killzone: Active / Upcoming / Closed."""
    now = datetime.now(tz=NY_TZ).time()
    statuses = {}
    for name, (start, end) in KILLZONES.items():
        if start > end:
            active = now >= start or now < end
        else:
            active = start <= now < end
        if active:
            statuses[name] = "Active"
        else:
            statuses[name] = "Closed"
    return statuses


# ---------------------------------------------------------------------------
# ICT Logic – Fair Value Gaps
# ---------------------------------------------------------------------------

def detect_fvg(df: pd.DataFrame) -> list[dict]:
    """
    Detect 3-candle Fair Value Gaps.
    Bullish FVG: candle[i-2].high < candle[i].low  (gap up)
    Bearish FVG: candle[i-2].low  > candle[i].high (gap down)
    """
    fvgs = []
    if len(df) < 3:
        return fvgs

    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index

    for i in range(2, len(df)):
        # Bullish FVG – gap between candle i-2 high and candle i low
        if highs[i - 2] < lows[i]:
            fvgs.append({
                "type": "Bullish FVG",
                "top": float(lows[i]),
                "bottom": float(highs[i - 2]),
                "start": timestamps[i - 2],
                "end": timestamps[i],
            })
        # Bearish FVG – gap between candle i-2 low and candle i high
        elif lows[i - 2] > highs[i]:
            fvgs.append({
                "type": "Bearish FVG",
                "top": float(lows[i - 2]),
                "bottom": float(highs[i]),
                "start": timestamps[i - 2],
                "end": timestamps[i],
            })
    return fvgs


# ---------------------------------------------------------------------------
# ICT Logic – Liquidity Levels (PDH/PDL & Asian Range)
# ---------------------------------------------------------------------------

def get_previous_daily_hl(df: pd.DataFrame) -> dict:
    """Compute Previous Daily High / Low from the last full trading day."""
    daily = df.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    if len(daily) < 2:
        return {"pdh": None, "pdl": None}
    prev = daily.iloc[-2]
    return {"pdh": float(prev["high"]), "pdl": float(prev["low"])}


def get_asian_range(df: pd.DataFrame) -> dict:
    """Get the most recent Asian session (20:00 – 00:00 EST) high/low."""
    now_ny = datetime.now(tz=NY_TZ)
    today_start = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    asian_start = today_start - timedelta(hours=4)  # 20:00 previous day
    asian_end = today_start

    asia = df.loc[asian_start:asian_end]
    if asia.empty:
        # Try one more day back
        asian_start -= timedelta(days=1)
        asian_end -= timedelta(days=1)
        asia = df.loc[asian_start:asian_end]
    if asia.empty:
        return {"asian_high": None, "asian_low": None}
    return {
        "asian_high": float(asia["high"].max()),
        "asian_low": float(asia["low"].min()),
    }


def liquidity_distances(price: float, levels: dict) -> list[dict]:
    """Calculate distance in pips to each liquidity level."""
    results = []
    label_map = {
        "pdh": "Previous Daily High",
        "pdl": "Previous Daily Low",
        "asian_high": "Asian High",
        "asian_low": "Asian Low",
    }
    for key, label in label_map.items():
        level = levels.get(key)
        if level is None:
            continue
        dist_pips = round((price - level) * 10_000, 1)
        results.append({"Level": label, "Price": f"{level:.5f}", "Distance (pips)": dist_pips})
    return results


# ---------------------------------------------------------------------------
# ICT Logic – Liquidity Sweeps
# ---------------------------------------------------------------------------

def detect_liquidity_sweeps(df: pd.DataFrame, levels: dict) -> list[dict]:
    """Check if price has swept any key liquidity level in recent candles."""
    sweeps = []
    if df.empty:
        return sweeps

    recent = df.tail(20)
    label_map = {
        "pdh": "Previous Daily High",
        "pdl": "Previous Daily Low",
        "asian_high": "Asian High",
        "asian_low": "Asian Low",
    }
    for key, label in label_map.items():
        level = levels.get(key)
        if level is None:
            continue
        for ts, row in recent.iterrows():
            # Sweep high-side level: wick above then close below
            if key in ("pdh", "asian_high"):
                if row["high"] > level and row["close"] < level:
                    sweeps.append({"time": ts, "level": label, "type": "Bearish Sweep", "price": f"{level:.5f}"})
            # Sweep low-side level: wick below then close above
            if key in ("pdl", "asian_low"):
                if row["low"] < level and row["close"] > level:
                    sweeps.append({"time": ts, "level": label, "type": "Bullish Sweep", "price": f"{level:.5f}"})

    # Deduplicate to most recent per level
    seen = set()
    unique = []
    for s in reversed(sweeps):
        if s["level"] not in seen:
            seen.add(s["level"])
            unique.append(s)
    return list(reversed(unique))


# ---------------------------------------------------------------------------
# ICT Logic – Market Structure Shift (MSS)
# ---------------------------------------------------------------------------

def detect_mss(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
    """
    Detect Market Structure Shifts (break of recent swing high/low).
    Bullish MSS: price breaks above a recent swing high after making lower lows.
    Bearish MSS: price breaks below a recent swing low after making higher highs.
    """
    shifts = []
    if len(df) < lookback:
        return shifts

    recent = df.tail(lookback)
    highs = recent["high"].values
    lows = recent["low"].values
    timestamps = recent.index

    for i in range(4, len(recent)):
        # Swing high at i-2
        if highs[i - 2] > highs[i - 3] and highs[i - 2] > highs[i - 1]:
            # Bearish MSS: current candle breaks below the swing low
            swing_low = min(lows[i - 3], lows[i - 2], lows[i - 1])
            if lows[i] < swing_low:
                shifts.append({
                    "time": timestamps[i],
                    "type": "Bearish MSS",
                    "price": f"{swing_low:.5f}",
                })
        # Swing low at i-2
        if lows[i - 2] < lows[i - 3] and lows[i - 2] < lows[i - 1]:
            # Bullish MSS: current candle breaks above the swing high
            swing_high = max(highs[i - 3], highs[i - 2], highs[i - 1])
            if highs[i] > swing_high:
                shifts.append({
                    "time": timestamps[i],
                    "type": "Bullish MSS",
                    "price": f"{swing_high:.5f}",
                })

    # Keep most recent 10
    return shifts[-10:]


# ---------------------------------------------------------------------------
# ICT Logic – SMT Divergence
# ---------------------------------------------------------------------------

def detect_smt_divergence(
    df_eur: pd.DataFrame, df_gbp: pd.DataFrame, lookback: int = 30
) -> list[dict]:
    """
    Smart Money Technique (SMT) Divergence:
    Compare swing highs/lows of EUR/USD and GBP/USD.
    Alert when one makes a new extreme and the other fails to confirm.
    """
    divergences = []
    if df_eur.empty or df_gbp.empty:
        return divergences

    # Align on common timestamps
    common = df_eur.index.intersection(df_gbp.index)
    if len(common) < lookback:
        return divergences
    common = common.sort_values()[-lookback:]

    eur = df_eur.loc[common]
    gbp = df_gbp.loc[common]

    for i in range(2, len(common)):
        ts = common[i]
        # Bearish SMT: EUR makes Higher High but GBP does NOT
        eur_hh = eur["high"].iloc[i] > eur["high"].iloc[i - 1]
        gbp_hh = gbp["high"].iloc[i] > gbp["high"].iloc[i - 1]
        if eur_hh and not gbp_hh:
            divergences.append({
                "time": ts,
                "type": "Bearish SMT",
                "detail": "EUR/USD Higher High, GBP/USD failed to confirm",
            })
        elif gbp_hh and not eur_hh:
            divergences.append({
                "time": ts,
                "type": "Bearish SMT",
                "detail": "GBP/USD Higher High, EUR/USD failed to confirm",
            })

        # Bullish SMT: EUR makes Lower Low but GBP does NOT
        eur_ll = eur["low"].iloc[i] < eur["low"].iloc[i - 1]
        gbp_ll = gbp["low"].iloc[i] < gbp["low"].iloc[i - 1]
        if eur_ll and not gbp_ll:
            divergences.append({
                "time": ts,
                "type": "Bullish SMT",
                "detail": "EUR/USD Lower Low, GBP/USD failed to confirm",
            })
        elif gbp_ll and not eur_ll:
            divergences.append({
                "time": ts,
                "type": "Bullish SMT",
                "detail": "GBP/USD Lower Low, EUR/USD failed to confirm",
            })

    # Keep most recent 10
    return divergences[-10:]


# ---------------------------------------------------------------------------
# ICT Logic – Directional Bias Engine (H4)
# ---------------------------------------------------------------------------

def _find_swing_points(df: pd.DataFrame, order: int = 3) -> tuple[list[dict], list[dict]]:
    """
    Identify swing highs and swing lows using a rolling window comparison.
    A swing high at index i requires the high to be the max of
    [i-order .. i+order]. Similarly for swing lows.
    Returns (swing_highs, swing_lows) each as list of {index, price, time}.
    """
    swing_highs = []
    swing_lows = []
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df.index

    for i in range(order, len(df) - order):
        # Swing high: highest in the window
        window_highs = highs[i - order: i + order + 1]
        if highs[i] == window_highs.max() and np.sum(window_highs == highs[i]) == 1:
            swing_highs.append({"idx": i, "price": float(highs[i]), "time": timestamps[i]})
        # Swing low: lowest in the window
        window_lows = lows[i - order: i + order + 1]
        if lows[i] == window_lows.min() and np.sum(window_lows == lows[i]) == 1:
            swing_lows.append({"idx": i, "price": float(lows[i]), "time": timestamps[i]})

    return swing_highs, swing_lows


def determine_h4_bias(df_h4: pd.DataFrame) -> dict:
    """
    Analyse the H4 chart to determine directional bias.

    Logic:
    - Find the two most recent swing highs and two most recent swing lows.
    - If the latest swing broke above the prior swing high → BULLISH.
    - If the latest swing broke below the prior swing low  → BEARISH.
    - Whichever event is more recent wins.

    Returns {"bias": "BULLISH"|"BEARISH"|"NEUTRAL",
             "detail": str,
             "swing_high": float|None,
             "swing_low": float|None}
    """
    if df_h4.empty or len(df_h4) < 10:
        return {"bias": "NEUTRAL", "detail": "Insufficient H4 data", "swing_high": None, "swing_low": None}

    swing_highs, swing_lows = _find_swing_points(df_h4, order=3)

    bullish_break_time = None
    bearish_break_time = None

    # Check for bullish break of structure (higher high)
    if len(swing_highs) >= 2:
        prev_sh, last_sh = swing_highs[-2], swing_highs[-1]
        if last_sh["price"] > prev_sh["price"]:
            bullish_break_time = last_sh["time"]

    # Check for bearish break of structure (lower low)
    if len(swing_lows) >= 2:
        prev_sl, last_sl = swing_lows[-2], swing_lows[-1]
        if last_sl["price"] < prev_sl["price"]:
            bearish_break_time = last_sl["time"]

    latest_sh = swing_highs[-1]["price"] if swing_highs else None
    latest_sl = swing_lows[-1]["price"] if swing_lows else None

    # Determine which break is more recent
    if bullish_break_time and bearish_break_time:
        if bullish_break_time >= bearish_break_time:
            return {
                "bias": "BULLISH",
                "detail": f"H4 Higher High @ {swing_highs[-1]['price']:.5f} ({swing_highs[-1]['time'].strftime('%m-%d %H:%M')})",
                "swing_high": latest_sh,
                "swing_low": latest_sl,
            }
        else:
            return {
                "bias": "BEARISH",
                "detail": f"H4 Lower Low @ {swing_lows[-1]['price']:.5f} ({swing_lows[-1]['time'].strftime('%m-%d %H:%M')})",
                "swing_high": latest_sh,
                "swing_low": latest_sl,
            }
    elif bullish_break_time:
        return {
            "bias": "BULLISH",
            "detail": f"H4 Higher High @ {swing_highs[-1]['price']:.5f} ({swing_highs[-1]['time'].strftime('%m-%d %H:%M')})",
            "swing_high": latest_sh,
            "swing_low": latest_sl,
        }
    elif bearish_break_time:
        return {
            "bias": "BEARISH",
            "detail": f"H4 Lower Low @ {swing_lows[-1]['price']:.5f} ({swing_lows[-1]['time'].strftime('%m-%d %H:%M')})",
            "swing_high": latest_sh,
            "swing_low": latest_sl,
        }

    return {"bias": "NEUTRAL", "detail": "No clear H4 structure break", "swing_high": latest_sh, "swing_low": latest_sl}


# ---------------------------------------------------------------------------
# ICT Logic – Draw on Liquidity
# ---------------------------------------------------------------------------

def draw_on_liquidity(price: float, pdh: float | None, pdl: float | None, bias: str) -> dict:
    """
    Calculate the Draw on Liquidity (DOL) target and distance.

    - BULLISH bias  → price is drawn toward the PDH (buy-side liquidity).
    - BEARISH bias  → price is drawn toward the PDL (sell-side liquidity).
    - Also reports distance to both levels regardless of bias.
    """
    result = {
        "target": None,
        "target_label": None,
        "target_price": None,
        "distance_pips": None,
        "pdh_pips": None,
        "pdl_pips": None,
    }

    if pdh is not None:
        result["pdh_pips"] = round((pdh - price) * 10_000, 1)
    if pdl is not None:
        result["pdl_pips"] = round((pdl - price) * 10_000, 1)

    if bias == "BULLISH" and pdh is not None:
        result["target"] = "PDH (Buy-side Liquidity)"
        result["target_label"] = "Previous Daily High"
        result["target_price"] = pdh
        result["distance_pips"] = result["pdh_pips"]
    elif bias == "BEARISH" and pdl is not None:
        result["target"] = "PDL (Sell-side Liquidity)"
        result["target_label"] = "Previous Daily Low"
        result["target_price"] = pdl
        result["distance_pips"] = result["pdl_pips"]

    return result


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def build_candlestick_chart(
    df: pd.DataFrame,
    fvgs: list[dict],
    levels: dict,
    symbol: str,
    timeframe: str,
) -> go.Figure:
    """Build a Plotly candlestick chart with FVG rectangles and liquidity lines."""
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Shade FVGs (limit to last 20 for readability)
    for fvg in fvgs[-20:]:
        color = "rgba(0,200,83,0.15)" if "Bullish" in fvg["type"] else "rgba(255,82,82,0.15)"
        border = "rgba(0,200,83,0.6)" if "Bullish" in fvg["type"] else "rgba(255,82,82,0.6)"
        fig.add_shape(
            type="rect",
            x0=fvg["start"], x1=fvg["end"],
            y0=fvg["bottom"], y1=fvg["top"],
            fillcolor=color,
            line=dict(color=border, width=1),
            layer="below",
        )

    # Liquidity lines
    line_styles = {
        "pdh": ("Previous Daily High", "#ff9800", "dash"),
        "pdl": ("Previous Daily Low", "#ff9800", "dot"),
        "asian_high": ("Asian High", "#9c27b0", "dash"),
        "asian_low": ("Asian Low", "#9c27b0", "dot"),
    }
    for key, (label, color, dash) in line_styles.items():
        val = levels.get(key)
        if val is not None:
            fig.add_hline(
                y=val,
                line_color=color,
                line_dash=dash,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_color=color,
            )

    fig.update_layout(
        title=f"{symbol} – {timeframe} (New York Time)",
        xaxis_title="Time (EST/EDT)",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="ICT Forex Dashboard", layout="wide")

    # ---- Sidebar --------------------------------------------------------
    st.sidebar.title("ICT Forex Dashboard")
    st.sidebar.markdown("---")

    data_source = st.sidebar.radio("Data Source", ["yfinance", "TwelveData"], index=0)
    api_key = ""
    if data_source == "TwelveData":
        api_key = st.sidebar.text_input("TwelveData API Key", type="password")
        if not api_key:
            st.sidebar.warning("Enter your TwelveData API key to fetch data.")

    pair = st.sidebar.selectbox("Pair", ["EURUSD", "GBPUSD"])
    timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h"])

    auto_refresh = st.sidebar.checkbox("Auto-refresh (60 s)", value=False)
    if auto_refresh:
        st.sidebar.caption("Dashboard will re-run every 60 seconds.")

    st.sidebar.markdown("---")
    st.sidebar.caption("All times shown in New York (EST/EDT)")

    # ---- Fetch data -----------------------------------------------------
    if data_source == "TwelveData" and not api_key:
        st.info("Please enter your TwelveData API key in the sidebar to get started.")
        st.stop()

    with st.spinner("Fetching market data..."):
        try:
            df = fetch_data(pair, timeframe, data_source, api_key)
        except Exception as e:
            st.error(f"Error fetching data for {pair}: {e}")
            st.stop()

    if df.empty:
        st.warning(f"No data returned for {pair} ({timeframe}). Market may be closed.")
        st.stop()

    # Also fetch the other pair for SMT divergence
    other_pair = "GBPUSD" if pair == "EURUSD" else "EURUSD"
    try:
        df_other = fetch_data(other_pair, timeframe, data_source, api_key)
    except Exception:
        df_other = pd.DataFrame()

    # Fetch H4 data for Directional Bias Engine
    try:
        df_h4 = fetch_data(pair, "4h", data_source, api_key)
    except Exception:
        df_h4 = pd.DataFrame()

    # ---- Compute ICT metrics --------------------------------------------
    last_price = float(df["close"].iloc[-1])
    midnight_open = get_midnight_open(df)
    open_830 = get_830_open(df)
    kz_status = killzone_status()
    active_kz = current_killzone()

    fvgs = detect_fvg(df)
    pdhl = get_previous_daily_hl(df)
    asian = get_asian_range(df)
    levels = {**pdhl, **asian}

    sweeps = detect_liquidity_sweeps(df, levels)
    mss_list = detect_mss(df)
    smt_list = detect_smt_divergence(df, df_other) if not df_other.empty else []

    # Directional Bias Engine (H4)
    bias_info = determine_h4_bias(df_h4)
    bias = bias_info["bias"]

    # Draw on Liquidity
    dol = draw_on_liquidity(last_price, levels.get("pdh"), levels.get("pdl"), bias)

    # ---- Header with Bias Color -----------------------------------------
    if bias == "BULLISH":
        header_color = "#00c853"  # green
    elif bias == "BEARISH":
        header_color = "#ff1744"  # red
    else:
        header_color = "#ffc107"  # amber/neutral

    st.markdown(
        f'<h1 style="color:{header_color}; margin-bottom:0;">ICT Dashboard – {pair}'
        f'<span style="font-size:0.5em; margin-left:1em; padding:4px 12px;'
        f' border-radius:6px; background:{header_color}; color:#fff;">'
        f'{bias}</span></h1>',
        unsafe_allow_html=True,
    )
    st.caption(f"H4 Bias: {bias_info['detail']}")

    # ---- Row 1: Live Status + Killzones ---------------------------------
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        st.subheader("Live Status")
        st.metric("Current Price", f"{last_price:.5f}")
        if active_kz:
            st.success(f"Active Killzone: **{active_kz}**")
        else:
            st.info("No active Killzone")

    with col2:
        st.subheader("Power of 3")
        if midnight_open:
            po3_label = premium_discount_label(last_price, midnight_open)
            delta = round((last_price - midnight_open) * 10_000, 1)
            st.metric("Midnight Open", f"{midnight_open:.5f}", delta=f"{delta} pips")
            st.caption(f"Zone: **{po3_label}**")
        else:
            st.caption("Midnight Open not available")
        if open_830:
            label_830 = premium_discount_label(last_price, open_830)
            delta_830 = round((last_price - open_830) * 10_000, 1)
            st.metric("8:30 AM Open", f"{open_830:.5f}", delta=f"{delta_830} pips")
            st.caption(f"Zone: **{label_830}**")
        else:
            st.caption("8:30 AM Open not available")

    with col3:
        st.subheader("Killzone Tracker")
        for name, status in kz_status.items():
            start, end = KILLZONES[name]
            time_label = f"{start.strftime('%H:%M')} – {end.strftime('%H:%M')} EST"
            if status == "Active":
                st.success(f"**{name}** ({time_label})")
            else:
                st.caption(f"{name} ({time_label}) – {status}")

    st.markdown("---")

    # ---- Row 2: Chart ---------------------------------------------------
    st.subheader("Candlestick Chart with Fair Value Gaps")
    fig = build_candlestick_chart(df, fvgs, levels, pair, timeframe)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Row 3: Directional Bias + Draw on Liquidity --------------------
    bias_col, dol_col = st.columns(2)

    with bias_col:
        st.subheader("Directional Bias Engine (H4)")
        if bias == "BULLISH":
            st.success(f"**{bias}** — Market structure broke to the upside")
        elif bias == "BEARISH":
            st.error(f"**{bias}** — Market structure broke to the downside")
        else:
            st.warning(f"**{bias}** — No clear directional break")
        st.caption(bias_info["detail"])
        if bias_info["swing_high"] is not None:
            st.metric("H4 Swing High", f"{bias_info['swing_high']:.5f}")
        if bias_info["swing_low"] is not None:
            st.metric("H4 Swing Low", f"{bias_info['swing_low']:.5f}")

    with dol_col:
        st.subheader("Draw on Liquidity")
        if dol["target"]:
            if bias == "BULLISH":
                st.success(f"Target: **{dol['target']}**")
            else:
                st.error(f"Target: **{dol['target']}**")
            st.metric(
                dol["target_label"],
                f"{dol['target_price']:.5f}",
                delta=f"{dol['distance_pips']} pips remaining",
            )
        else:
            st.info("No DOL target — PDH/PDL data unavailable or bias is NEUTRAL")

        st.markdown("**Distance to Daily Levels**")
        dol_rows = []
        if dol["pdh_pips"] is not None:
            direction = "above" if dol["pdh_pips"] > 0 else "below"
            dol_rows.append({
                "Level": "Previous Daily High",
                "Price": f"{levels['pdh']:.5f}",
                "Distance": f"{abs(dol['pdh_pips'])} pips {direction}",
            })
        if dol["pdl_pips"] is not None:
            direction = "above" if dol["pdl_pips"] > 0 else "below"
            dol_rows.append({
                "Level": "Previous Daily Low",
                "Price": f"{levels['pdl']:.5f}",
                "Distance": f"{abs(dol['pdl_pips'])} pips {direction}",
            })
        if dol_rows:
            st.dataframe(pd.DataFrame(dol_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Daily level data not available")

    st.markdown("---")

    # ---- Row 4: Signals + Liquidity Radar -------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Signals Table")

        signals = []
        for m in mss_list:
            signals.append({
                "Time (EST)": m["time"].strftime("%Y-%m-%d %H:%M"),
                "Signal": m["type"],
                "Detail": f"@ {m['price']}",
            })
        for s in smt_list:
            signals.append({
                "Time (EST)": s["time"].strftime("%Y-%m-%d %H:%M"),
                "Signal": s["type"],
                "Detail": s["detail"],
            })
        for sw in sweeps:
            signals.append({
                "Time (EST)": sw["time"].strftime("%Y-%m-%d %H:%M"),
                "Signal": sw["type"],
                "Detail": f"{sw['level']} @ {sw['price']}",
            })

        if signals:
            sig_df = pd.DataFrame(signals).sort_values("Time (EST)", ascending=False).head(20)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals detected in the current data window.")

    with col_right:
        st.subheader("Liquidity Radar")
        liq = liquidity_distances(last_price, levels)
        if liq:
            liq_df = pd.DataFrame(liq)
            st.dataframe(liq_df, use_container_width=True, hide_index=True)
        else:
            st.info("Liquidity levels not available.")

        # FVG summary
        st.subheader("Recent Fair Value Gaps")
        if fvgs:
            fvg_rows = []
            for f in fvgs[-10:]:
                fvg_rows.append({
                    "Time": f["start"].strftime("%Y-%m-%d %H:%M"),
                    "Type": f["type"],
                    "Top": f"{f['top']:.5f}",
                    "Bottom": f"{f['bottom']:.5f}",
                    "Size (pips)": round((f["top"] - f["bottom"]) * 10_000, 1),
                })
            st.dataframe(pd.DataFrame(fvg_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No FVGs detected.")

    # ---- Auto-refresh ---------------------------------------------------
    if auto_refresh:
        import time as _time
        _time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
