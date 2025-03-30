import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import requests

# ------------------------ CONFIG ------------------------
TWELVE_DATA_API_KEY = '14ecff712573433ab2697593d22d5247'
TELEGRAM_BOT_TOKEN = '7564845823:AAHWWYtt2d-qaWsTxMbAxpzpwQVcYluGF1Q'
TELEGRAM_CHAT_ID = '6522301339'

ASSETS = {
    'XAU/USD': 'GC=F',
    'XAG/USD': 'SI=F',
    'GBP/USD': 'GBPUSD=X',
    'USD/CHF': 'USDCHF=X',
    'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X',
    'EUR/USD': 'EURUSD=X',
    'FTSE 100': '^FTMC',
    'S&P 500': '^GSPC'
}

TIMEFRAME_OPTIONS = {
    "30 days": 30,
    "90 days": 90,
    "180 days": 180
}

LOG_FILE = "zone_signals_log.csv"

# ------------------------ FUNCTIONS ------------------------
def get_yf_data(symbol, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        return None
    df = df[['Close']].rename(columns={'Close': 'close'}).reset_index()
    df['RSI'] = calculate_rsi(df['close'])
    df['ATR'] = calculate_atr(symbol, days)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(symbol, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        return None
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr = tr.rolling(window=14).mean()
    return atr.iloc[-1]

def calculate_zones(df):
    max_price = float(df['close'].max())
    min_price = float(df['close'].min())
    current_price = float(df['close'].iloc[-1])
    price_range = max_price - min_price
    upper_threshold = max_price - (price_range * 0.10)
    lower_threshold = min_price + (price_range * 0.10)

    if current_price >= upper_threshold:
        zone = 'Sell'
    elif current_price <= lower_threshold:
        zone = 'Buy'
    else:
        zone = 'Neutral'

    return {
        'Current Price': current_price,
        'High': max_price,
        'Low': min_price,
        'RSI': round(df['RSI'].iloc[-1], 2) if 'RSI' in df.columns else None,
        'ATR': round(df['ATR'].iloc[-1], 2) if 'ATR' in df.columns else None,
        'Zone': zone,
        '% From High': f"{-((max_price - current_price) / max_price * 100):.2f}%",
        '% From Low': f"{((current_price - min_price) / min_price * 100):.2f}%"
    }

def generate_chart_base64(df):
    fig, ax = plt.subplots(figsize=(2, 0.8))
    ax.plot(df['close'], linewidth=1.5)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" width="100"/>'

def style_zone(zone):
    if zone == "Buy":
        return '<span style="background-color:#28a745; color:white; padding:4px 10px; border-radius:8px; font-weight:bold;">Buy</span>'
    elif zone == "Sell":
        return '<span style="background-color:#dc3545; color:white; padding:4px 10px; border-radius:8px; font-weight:bold;">Sell</span>'
    else:
        return '<span style="background-color:#e2e3e5; color:#383d41; padding:4px 8px; border-radius:8px;">Neutral</span>'

def log_signals(df, timeframe):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signals = df[(df['Zone'] == 'Buy') | (df['Zone'] == 'Sell')].copy()
    if not signals.empty:
        signals['Timeframe'] = timeframe
        signals['Timestamp'] = timestamp
        signals.to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
    return signals

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram Error:", e)

# ------------------------ UI ------------------------
st.set_page_config(page_title="Trading Zone Dashboard", layout="wide")
st.title("📈 Trading Pair Zone Dashboard")

st.markdown("This dashboard shows whether your selected assets are in Buy, Sell, or Neutral zones based on their selected timeframe.")

col1, col2 = st.columns([1, 1])
with col1:
    selected_timeframe = st.selectbox("📅 Select Timeframe:", list(TIMEFRAME_OPTIONS.keys()), index=1)
with col2:
    show_only_signals = st.checkbox("🔎 Show only Buy/Sell zones", value=False)

days_lookback = TIMEFRAME_OPTIONS[selected_timeframe]

results = []
signal_alerts = []

for label, symbol in ASSETS.items():
    data = get_yf_data(symbol, days_lookback)

    if data is None or len(data) < 30:
        results.append({"Symbol": label, "Error": "Data unavailable"})
        continue

    analysis = calculate_zones(data)
    chart = generate_chart_base64(data)
    zone = analysis['Zone']
    rsi = analysis['RSI']

    if zone in ['Buy', 'Sell']:
        alert_msg = f"🚨 *{label}* entered *{zone}* zone (RSI: {rsi}) on *{selected_timeframe}*"
        signal_alerts.append(alert_msg)
        send_telegram_message(alert_msg)

    results.append({"Symbol": f"{label}<br>{chart}", "Error": "None", **analysis})

df_results = pd.DataFrame(results)

if not df_results.empty and 'Zone' in df_results.columns:
    df_to_log = df_results.copy()
    df_to_log['Symbol'] = df_to_log['Symbol'].str.extract(r'^(.*?)<br>')[0]
    logged_signals = log_signals(df_to_log, selected_timeframe)

df_display = df_results.copy()

if not df_display.empty and 'Zone' in df_display.columns:
    df_display['Zone'] = df_display['Zone'].apply(lambda z: style_zone(z) if z in ['Buy', 'Sell', 'Neutral'] else z)
    if show_only_signals:
        df_display = df_display[df_display['Zone'].str.contains('Buy|Sell')]

st.markdown("### ✅ Zone Overview with Highlights")
st.write(
    df_display.to_html(escape=False, index=False),
    unsafe_allow_html=True
)

if signal_alerts:
    st.markdown("---")
    for alert in signal_alerts:
        st.markdown(f"<div style='padding:10px; background:#fff3cd; border-left:5px solid #ffc107; border-radius:6px;'>{alert}</div>", unsafe_allow_html=True)
    st.audio("https://www.soundjay.com/buttons/sounds/beep-07.mp3", format="audio/mp3")

csv = df_results.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", csv, "trading_zones.csv", "text/csv")

st.caption("Data from Yahoo Finance (via yfinance). Not financial advice. 😉")





