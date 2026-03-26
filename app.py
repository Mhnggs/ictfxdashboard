"""
ICT (Inner Circle Trader) Forex Dashboard
==========================================
Real-time dashboard for EUR/USD and GBP/USD with ICT concepts:
- Daily Power of 3 (Midnight Open, 8:30 AM Open, Premium/Discount)
- Killzone Tracker (Asian, London, NY)
- Fair Value Gaps (FVG)
- Liquidity Sweeps (Asian Range, PDH/PDL)
- SMT Divergence (EUR/USD vs GBP/USD)
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

TIMEFRAME_MAP_YF = {"5m": "5m", "15m": "15m", "1h": "60m"}
TIMEFRAME_MAP_TD = {"5m": "5min", "15m": "15min", "1h": "1h"}

# How many calendar days of history to request for each timeframe
PERIOD_DAYS = {"5m": 5, "15m": 14, "1h": 30}


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


def fetch_data(symbol: str, timeframe: str, source: str, api_key: str = "") -> pd.DataFrame:
    if source == "TwelveData":
        return fetch_data_twelvedata(symbol, timeframe, api_key)
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

    # ---- Header ---------------------------------------------------------
    st.title(f"ICT Dashboard – {pair}")

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

    # ---- Row 3: Signals + Liquidity Radar -------------------------------
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
