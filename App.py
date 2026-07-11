# ══════════════════════════════════════════════════════════════════
# app.py — Mshauri wa Malipo ya Kigeni | Tanzania  (v3 – Premium UI)
# ══════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import warnings
from datetime import timedelta, date
warnings.filterwarnings('ignore')

import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mshauri wa Malipo ya Kigeni",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="auto"
)

# ─────────────────────────────────────────────────────────────────
# PATHS + EXTERNAL DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

def load_css(path):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(os.path.join(BASE_DIR, "assets", "style.css"))

PURPOSES = {
    "💳 Netflix / Spotify / huduma za mtandaoni": "streaming",
    "📦 Bidhaa / mizigo kutoka nje (China, UAE…)": "bidhaa",
    "✈️  Tiketi / safari nje ya nchi":             "safari",
    "📤 Kutuma pesa kwa familia / rafiki":         "kutuma",
    "🎓 Ada ya masomo nje ya nchi":                "masomo",
    "💊 Dawa / huduma za afya kutoka nje":         "afya",
    "🔄 Malipo mengine ya kigeni":                 "nyingine",
}

CURRENCIES = {
    "USD 🇺🇸 — Dola ya Marekani": ("USD", "usd"),
    "EUR 🇪🇺 — Euro ya Ulaya":    ("EUR", "eur"),
    "CNY 🇨🇳 — Yuan ya China":    ("CNY", "cny"),
}

TIMEFRAMES = {
    "Wiki 1 (Siku 7)":    7,
    "Wiki 2 (Siku 14)":  14,
    "Mwezi 1 (Siku 30)": 30,
}

# ─────────────────────────────────────────────────────────────────
# SIMPLE TRANSLATION / I18N (Chinese, English, Kiswahili)
TRANSLATIONS = {
    'en': {
        'sidebar_title': '⚙️ Tell me more',
        'purpose_label': 'What are you paying for?',
        'currency_label': 'Payment currency',
        'timeframe_label': 'Check horizon',
        'refresh_button': '🔄 Refresh data & models',
        'payment_title': '🧮 Payment amount',
        'download_csv': '⤓ Download forecast CSV',
        'tab_advice': '💬 My advice',
        'tab_trend': '📈 Price trend',
        'tab_analysis': '🔬 Deep analysis',
        'compare_title': '🌍 Current currency comparison',
        'header_title': '💱 Foreign Payment Advisor',
        'header_sub': 'Helps you decide when to pay in USD, EUR, or CNY',
    },
    'sw': {
        'sidebar_title': '⚙️ Niambie Zaidi',
        'purpose_label': 'Unataka kulipa nini?',
        'currency_label': 'Sarafu ya malipo',
        'timeframe_label': 'Angalia mwenendo wa muda gani?',
        'refresh_button': '🔄 Rudisha Data & Modeli',
        'payment_title': '🧮 Kiasi cha Malipo',
        'download_csv': '⤓ Pakua utabiri kama CSV',
        'tab_advice': '💬 Ushauri Wangu',
        'tab_trend': '📈 Mwenendo wa Bei',
        'tab_analysis': '🔬 Uchambuzi wa Kina',
        'compare_title': '🌍 Kulinganisha Sarafu za Sasa',
        'header_title': '💱 Mshauri wa Malipo ya Kigeni',
        'header_sub': 'Inakusaidia kujua wakati mzuri wa kulipa kwa USD, EUR, au CNY — bila kuhitaji ujuzi wa fedha',
    },
    'zh': {
        'sidebar_title': '⚙️ 告诉我更多',
        'purpose_label': '你要支付什么？',
        'currency_label': '支付货币',
        'timeframe_label': '查看时间范围',
        'refresh_button': '🔄 刷新数据和模型',
        'payment_title': '🧮 支付金额',
        'download_csv': '⤓ 下载预测 CSV',
        'tab_advice': '💬 我的建议',
        'tab_trend': '📈 价格趋势',
        'tab_analysis': '🔬 深度分析',
        'compare_title': '🌍 当前货币比较',
        'header_title': '💱 外汇支付顾问',
        'header_sub': '帮助您决定何时以 USD、EUR 或 CNY 支付',
    }
}

def translate(key):
    lang = globals().get('LANG', 'sw')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

PRESETS = {
    "streaming": [("Netflix 15",  15.0),  ("Spotify 10",  10.0),  ("YouTube 14",  14.0)],
    "bidhaa":    [("Ndogo 100",  100.0),  ("Kati 500",   500.0),  ("Kubwa 2000", 2000.0)],
    "safari":    [("Ndege 300",  300.0),  ("Kati 800",   800.0),  ("Luxury 2k", 2000.0)],
    "kutuma":    [("Kidogo 50",   50.0),  ("Kati 200",   200.0),  ("Kubwa 500",  500.0)],
    "masomo":    [("Sem 1500",  1500.0),  ("Mwaka 3k", 3000.0),  ("Masters 6k",6000.0)],
    "afya":      [("Dawa 30",    30.0),   ("Tiba 200",   200.0),  ("Upasuaji 2k",2000.0)],
    "nyingine":  [("Kidogo 50",  50.0),   ("Kati 500",   500.0),  ("Kubwa 2k", 2000.0)],
}


# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    dfs = {}
    for cur, name in [('usd','USD'), ('eur','EUR'), ('cny','CNY')]:
        path = os.path.join(DATA_DIR, f'{cur}_tzs_clean.csv')
        df   = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        df   = df.sort_values('date').reset_index(drop=True)
        dfs[name] = df
    return dfs


@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    load_errors = {}   # collect what actually failed, per currency/model
    for cur in ['usd', 'eur', 'cny']:
        models[cur] = {'arima': None, 'prophet': None,
                       'xgboost': None, 'lstm': None, 'scaler': None}
        load_errors[cur] = {}
        try:
            models[cur]['arima'] = joblib.load(
                os.path.join(MODELS_DIR, f'arima_{cur}_model.pkl'))
        except Exception as e:
            load_errors[cur]['arima'] = str(e)
        try:
            models[cur]['prophet'] = joblib.load(
                os.path.join(MODELS_DIR, f'prophet_{cur}_model.pkl'))
        except Exception as e:
            load_errors[cur]['prophet'] = str(e)
        try:
            models[cur]['xgboost'] = joblib.load(
                os.path.join(MODELS_DIR, f'xgboost_{cur}_model_direct.pkl'))
        except Exception as e:
            load_errors[cur]['xgboost'] = str(e)
        try:
            import tensorflow as tf
            models[cur]['lstm'] = tf.keras.models.load_model(
                os.path.join(MODELS_DIR, f'lstm_{cur}_model.h5'), compile=False)
            models[cur]['scaler'] = joblib.load(
                os.path.join(MODELS_DIR, f'scaler_{cur}.pkl'))
        except Exception as e:
            load_errors[cur]['lstm'] = str(e)

        # Print to server-side logs (visible in Render "Logs" tab) so failures
        # are no longer silent.
        for name, err in load_errors[cur].items():
            print(f"[MODEL LOAD FAILED] currency={cur} model={name} error={err}")

    models['_load_errors'] = load_errors   # stashed here so UI code can show it
    return models


# ─────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────
def pred_arima(model, steps):
    try:
        return model.forecast(steps=steps).values
    except Exception:
        return None

def pred_prophet(model, steps):
    try:
        future   = model.make_future_dataframe(periods=steps)
        forecast = model.predict(future)
        return forecast['yhat'].tail(steps).values
    except Exception:
        return None

def _xgboost_features(df):
    """Must exactly match the feature engineering used in retrain_models.py"""
    vals = df['mean'].values
    i = len(vals) - 1
    l1  = vals[i]
    l3  = vals[i-2]  if i >= 2  else l1
    l7  = vals[i-6]  if i >= 6  else l1
    l14 = vals[i-13] if i >= 13 else l1
    l30 = vals[i-29] if i >= 29 else l1
    ma7  = vals[max(0, i-6):i+1].mean()
    ma30 = vals[max(0, i-29):i+1].mean()
    std7 = vals[max(0, i-6):i+1].std()
    d = df['date'].iloc[i]
    return np.array([[l1, l3, l7, l14, l30, ma7, ma30, std7,
                      d.month, d.weekday(), (d.month - 1) // 3 + 1]])

def pred_xgboost(model, df, steps):
    try:
        feat = _xgboost_features(df)
        pred = model.predict(feat)[0]   # shape: (30,) — one shot, all days at once
        return np.array(pred[:steps])
    except Exception:
        return None

def pred_lstm(model, scaler, df, steps, seq=60):
    try:
        if len(df) < seq:
            return None
        scaled = scaler.transform(df['mean'].values[-seq:].reshape(-1, 1)).flatten()
        X = scaled.reshape(1, seq, 1)
        pred_scaled = model.predict(X, verbose=0)[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return pred[:steps]
    except Exception:
        return None

def get_ensemble(models_dict, df, cur_key, steps):
    forecasts = {}

    p = pred_xgboost(models_dict[cur_key]['xgboost'], df, steps)
    if p is not None: forecasts['XGBoost'] = p

    if models_dict[cur_key]['lstm'] and models_dict[cur_key]['scaler']:
        p = pred_lstm(models_dict[cur_key]['lstm'], models_dict[cur_key]['scaler'], df, steps)
        if p is not None: forecasts['LSTM'] = p

    p = pred_arima(models_dict[cur_key]['arima'], steps)
    if p is not None: forecasts['ARIMA'] = p

    p = pred_prophet(models_dict[cur_key]['prophet'], steps)
    if p is not None: forecasts['Prophet'] = p

    if not forecasts:
        return None, {}
    ensemble = np.mean(list(forecasts.values()), axis=0)
    return ensemble, forecasts

def get_confidence_band(all_forecasts, ensemble):
    if len(all_forecasts) < 2:
        spread = ensemble * 0.005
        return ensemble - spread, ensemble + spread
    arr = np.array(list(all_forecasts.values()))
    std = np.std(arr, axis=0)
    return ensemble - std, ensemble + std


# ─────────────────────────────────────────────────────────────────
# BEST-DAY ANALYSIS
# ─────────────────────────────────────────────────────────────────
def best_day_analysis(df):
    temp = df.copy()
    temp['weekday']       = temp['date'].dt.day_name()
    temp['week_of_month'] = (temp['date'].dt.day - 1) // 7 + 1
    wd  = temp.groupby('weekday')['mean'].mean()
    wom = temp.groupby('week_of_month')['mean'].mean()
    best_wd  = wd.idxmin()
    best_wom = int(wom.idxmin())
    day_sw   = {'Monday':'Jumatatu','Tuesday':'Jumanne','Wednesday':'Jumatano',
                'Thursday':'Alhamisi','Friday':'Ijumaa',
                'Saturday':'Jumamosi','Sunday':'Jumapili'}
    wom_lbl  = {1:"Wiki ya 1",2:"Wiki ya 2",3:"Wiki ya 3",4:"Wiki ya 4"}
    return {
        'best_weekday': day_sw.get(best_wd, best_wd),
        'best_wom':     wom_lbl.get(best_wom, f"Wiki {best_wom}"),
        'avg_saving_wd': float(wd.max() - wd.min()),
    }


# ─────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────
def get_advice(current_rate, ensemble, payment_amount, cur_name, purpose_key, steps):
    avg_future = float(np.mean(ensemble))
    change_pct = (avg_future - current_rate) / current_rate * 100
    tzs_now    = payment_amount * current_rate
    tzs_later  = payment_amount * avg_future
    tzs_save   = tzs_now - tzs_later

    if change_pct > 1.0:
        action = "LIPA SASA"; rec_style = "rec-go"; icon = "⚡"
        headline = (f"Bei ya {cur_name} inatarajiwa KUPANDA kwa {change_pct:.1f}% "
                    f"katika siku {steps} zijazo. Ukilipa sasa utalipa TZS chache zaidi.")
        saving_direction = "hasara"; saving_amount = abs(tzs_save)
    elif change_pct < -1.0:
        action = "SUBIRI"; rec_style = "rec-wait"; icon = "⏳"
        headline = (f"Bei ya {cur_name} inatarajiwa KUSHUKA kwa {abs(change_pct):.1f}% "
                    f"katika siku {steps} zijazo. Ukisubiri utalipa TZS chache zaidi!")
        saving_direction = "okoa"; saving_amount = abs(tzs_save)
    else:
        action = "ANGALIA"; rec_style = "rec-watch"; icon = "👀"
        headline = (f"Bei ya {cur_name} iko STABLE sasa hivi "
                    f"(mabadiliko ya {change_pct:.1f}% tu).")
        saving_direction = "stable"; saving_amount = 0.0

    if abs(change_pct) >= 2.0:
        urg_level = "Juu";   urg_class = "urg-high"; urg_note = "Mabadiliko ya bei ni makubwa — chukua hatua haraka."
    elif abs(change_pct) >= 1.0:
        urg_level = "Kati";  urg_class = "urg-mid";  urg_note = "Mabadiliko ni wastani — angalia kabla ya kulipa."
    else:
        urg_level = "Ndogo"; urg_class = "urg-low";  urg_note = "Mabadiliko ni madogo — hatari ni ndogo kwa sasa."

    purpose_msgs = {
        "streaming": {"LIPA SASA":f"Lipia subscription sasa — bei itapanda na utapoteza zaidi ukisubiri. 🎬","SUBIRI":f"Ukisubiri siku {steps}, subscription itakugharimu TZS chache zaidi. 🎬","ANGALIA":f"Bei iko stable — unaweza kulipa wakati wowote. 🎬"},
        "bidhaa":    {"LIPA SASA":f"Lipia invoice ya {cur_name} {payment_amount:,.0f} sasa kabla bei haijapanda. 📦","SUBIRI":f"Subiri — bei itashuka na utaokoa. 📦","ANGALIA":f"Bei iko stable. Unaweza kulipa wakati wowote. 📦"},
        "safari":    {"LIPA SASA":f"Nunua {cur_name} kwa safari sasa — bei itapanda. ✈️","SUBIRI":f"Subiri — bei itashuka na utapata thamani nzuri. ✈️","ANGALIA":f"Bei iko stable. Unaweza kubadilisha pesa ya safari wakati wowote. ✈️"},
        "kutuma":    {"LIPA SASA":f"Tuma pesa sasa! Bei itapanda — mpokeaji atapata kidogo zaidi. 📤","SUBIRI":f"Subiri siku {steps} — bei itashuka. 📤","ANGALIA":f"Bei iko stable. Unaweza kutuma pesa wakati wowote. 📤"},
        "masomo":    {"LIPA SASA":f"Lipia ada sasa — bei itapanda na ada itakuwa ghali zaidi. 🎓","SUBIRI":f"Kama tarehe inaruhusu, subiri — bei itashuka. 🎓","ANGALIA":f"Bei iko stable — panga malipo kulingana na tarehe yako. 🎓"},
        "afya":      {"LIPA SASA":f"Lipia sasa — bei itapanda. Afya yako ni muhimu! 💊","SUBIRI":f"Kama hali inaruhusu, subiri — bei itashuka. 💊","ANGALIA":f"Bei iko stable. Unaweza kulipa wakati wowote. 💊"},
        "nyingine":  {"LIPA SASA":f"Fanya malipo sasa — bei itapanda.","SUBIRI":f"Subiri — bei itashuka.","ANGALIA":f"Bei iko stable. Unaweza kufanya malipo wakati wowote."},
    }
    purpose_text = purpose_msgs.get(purpose_key, purpose_msgs["nyingine"])[action]

    return {
        "action": action, "rec_style": rec_style, "icon": icon,
        "headline": headline, "purpose_text": purpose_text,
        "change_pct": change_pct,
        "tzs_now": tzs_now, "tzs_later": tzs_later, "tzs_save": tzs_save,
        "saving_direction": saving_direction, "saving_amount": saving_amount,
        "avg_future": avg_future,
        "urgency_level": urg_level, "urgency_class": urg_class, "urgency_note": urg_note,
    }


# ─────────────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────────────
all_data = {}; all_models = {}
with st.spinner("Inapakia mfumo..."):
    try:
        all_data   = load_data()
        all_models = load_models()
    except Exception as e:
        st.error(f"❌ Hitilafu ya kupakia: {e}"); st.stop()

if not all_data:
    st.error("❌ Data haikupatikana. Angalia folder ya 'data/'"); st.stop()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 0.8rem 0">
        <span style="font-family:'Fraunces',serif;font-size:1.2rem;font-weight:700;
              background:linear-gradient(135deg,#0E7C7B,#E8A33D);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            💱 Mshauri
        </span>
    </div>""", unsafe_allow_html=True)
    # Language selector
    lang_choice = st.selectbox("Language / Lugha / 语言", ["Kiswahili (sw)", "English (en)", "中文 (zh)"], index=0)
    if lang_choice.endswith("(sw)"):
        LANG = 'sw'
    elif lang_choice.endswith("(en)"):
        LANG = 'en'
    else:
        LANG = 'zh'

    st.markdown(f'<div class="sb-title">{translate("sidebar_title")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">🎯 Lengo la Malipo</div>', unsafe_allow_html=True)
    purpose_label = st.selectbox(translate('purpose_label'), list(PURPOSES.keys()), label_visibility="collapsed")
    purpose_key   = PURPOSES[purpose_label]

    st.markdown('<div class="sb-title">💱 Sarafu &amp; Muda</div>', unsafe_allow_html=True)
    currency_label = st.selectbox(translate('currency_label'), list(CURRENCIES.keys()), label_visibility="collapsed")
    cur_display, cur_key = CURRENCIES[currency_label]
    timeframe_label = st.selectbox(translate('timeframe_label'), list(TIMEFRAMES.keys()))
    steps = TIMEFRAMES[timeframe_label]

    st.markdown("**— au —** chagua tarehe maalum:")
    use_date = st.checkbox("Weka tarehe yangu mwenyewe")
    if use_date:
        target_date = st.date_input(
            "Tarehe ya kulipa",
            value=date.today() + timedelta(days=7),
            min_value=date.today() + timedelta(days=1),
            max_value=date.today() + timedelta(days=90),
        )
        steps = max(1, (target_date - date.today()).days)
        st.caption(f"📅 Siku {steps} hadi {target_date.strftime('%d/%m/%Y')}")

    st.markdown(f'<div class="sb-title">{translate("payment_title")}</div>', unsafe_allow_html=True)
    presets = PRESETS.get(purpose_key, PRESETS["nyingine"])
    st.markdown('<p class="preset-lbl">Chaguo la haraka:</p>', unsafe_allow_html=True)
    pc = st.columns(3)
    preset_value = None
    for i, (lbl, val) in enumerate(presets):
        with pc[i]:
            if st.button(lbl, key=f"pre_{i}", use_container_width=True):
                preset_value = val

    if 'payment_amount' not in st.session_state:
        st.session_state['payment_amount'] = 15.0 if purpose_key == "streaming" else 1000.0
    if preset_value is not None:
        st.session_state['payment_amount'] = preset_value

    payment_amount = st.number_input(
        f"Kiasi ({cur_display})", min_value=0.01, max_value=10_000_000.0,
        value=float(st.session_state['payment_amount']), step=1.0,
    )
    st.session_state['payment_amount'] = payment_amount

    st.markdown("---")
    if st.button(translate('refresh_button'), use_container_width=True):
        load_data.clear(); load_models.clear(); st.rerun()

    st.markdown("""<p style='font-size:0.73rem;color:#5C7488;line-height:1.6;margin-top:0.8rem;'>
    ⚠️ Ushauri kulingana na data ya kihistoria. Thibitisha na
    <a href="https://www.bot.go.tz" target="_blank" style="color:#0E7C7B">BOT</a>
    au benki yako.</p>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FETCH DATA & RUN FORECAST
# ─────────────────────────────────────────────────────────────────
df           = all_data[cur_display]
current_rate = float(df['mean'].iloc[-1])
prev_rate    = float(df['mean'].iloc[-2])
rate_delta   = current_rate - prev_rate
rate_delta_p = rate_delta / prev_rate * 100
last_date    = df['date'].iloc[-1]
days_old     = (pd.Timestamp.now() - last_date).days

with st.spinner(f"Inachambua {cur_display}/TZS..."):
    ensemble, all_forecasts = get_ensemble(all_models, df, cur_key, steps)

if ensemble is None:
    st.error("❌ Modeli hazikuweza kutoa utabiri."); st.stop()

lower_band, upper_band = get_confidence_band(all_forecasts, ensemble)
advice = get_advice(current_rate, ensemble, payment_amount, cur_display, purpose_key, steps)
bda    = best_day_analysis(df)

# Plain-language "how confident should I be" indicator, based on how closely
# the underlying forecasts agree with each other — no model names/jargon
# shown to the end user.
def get_confidence_label(all_forecasts):
    if len(all_forecasts) < 2:
        return "Kati", "conf-mid", "Data haitoshi kulinganisha"
    arr = np.array(list(all_forecasts.values()))
    spread_pct = float(np.std(arr[:, -1]) / np.mean(arr[:, -1]) * 100)
    if spread_pct < 0.5:
        return "Juu", "conf-high", "Uchambuzi mbalimbali unakubaliana"
    elif spread_pct < 2.0:
        return "Kati", "conf-mid", "Uchambuzi unakubaliana kiasi"
    else:
        return "Chini", "conf-low", "Uchambuzi haukubaliani — angalia kwa makini"

conf_label, conf_class, conf_note = get_confidence_label(all_forecasts)

future_dates = [df['date'].iloc[-1] + timedelta(days=i+1) for i in range(steps)]

# ─────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────
arrow  = "▲" if rate_delta >= 0 else "▼"
acolor_css = "color:#D94E1F" if rate_delta >= 0 else "color:#0E9F6E"

col_hdr, col_pill = st.columns([2, 1])
with col_hdr:
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">{translate('header_title')}</div>
        <p class="hero-sub">{translate('header_sub')}</p>
    </div>
    """, unsafe_allow_html=True)

with col_pill:
    st.markdown(f"""
    <div style="height:100%;display:flex;flex-direction:column;justify-content:center;gap:10px;padding-top:0.2rem">
        <div class="glass-card g-indigo" style="margin-bottom:0">
            <div class="gc-label">Bei ya Sasa · {cur_display}/TZS</div>
            <div class="gc-value">TZS {current_rate:,.2f}</div>
            <div class="gc-note" style="{acolor_css};font-weight:600">{arrow} {abs(rate_delta):.2f} ({abs(rate_delta_p):.2f}%) tangu jana</div>
            <div class="gc-icon">💱</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Data warning
if days_old > 2:
    st.markdown(f"""
    <div class="data-warn-3d">
        📅 Bei ni ya <b>{last_date.strftime('%d/%m/%Y')}</b> (siku {days_old} zilizopita).
        Kwa bei halisi tembelea
        <a href="https://www.bot.go.tz" target="_blank" style="color:#5C3A0A;font-weight:600">www.bot.go.tz</a>.
    </div>""", unsafe_allow_html=True)

# Shimmer divider
st.markdown('<div class="shimmer-line"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([translate('tab_advice'), translate('tab_trend'), translate('tab_analysis')])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — USHAURI
# ══════════════════════════════════════════════════════════════════
with tab1:
    # ── Row 1: 4 glass metric chips ─────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="glass-card g-indigo">
            <div class="gc-label">Bei ya Wastani</div>
            <div class="gc-value">TZS {current_rate:,.0f}</div>
            <div class="gc-note" style="{acolor_css};font-weight:600">{arrow} {abs(rate_delta_p):.2f}% leo</div>
            <div class="gc-icon">📊</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass-card g-emerald">
            <div class="gc-label">Bei ya Kununua</div>
            <div class="gc-value">TZS {df['buying'].iloc[-1]:,.0f}</div>
            <div class="gc-note">Benki inanunua {cur_display}</div>
            <div class="gc-icon">🏦</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass-card g-violet">
            <div class="gc-label">Bei ya Kuuza</div>
            <div class="gc-value">TZS {df['selling'].iloc[-1]:,.0f}</div>
            <div class="gc-note">Benki inakuuzia {cur_display}</div>
            <div class="gc-icon">💰</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        urg_color_map = {"urg-high":"g-rose","urg-mid":"g-amber","urg-low":"g-emerald"}
        st.markdown(f"""
        <div class="glass-card {urg_color_map.get(advice['urgency_class'],'g-sky')}">
            <div class="gc-label">Hatari ya Dharura</div>
            <div class="gc-value">{advice['urgency_level']}</div>
            <div class="gc-note">{advice['urgency_note']}</div>
            <div class="gc-icon">⚠️</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Recommendation + Payment ─────────────────────────
    col_rec, col_pay = st.columns([1.1, 1], gap="large")

    with col_rec:
        st.markdown('<div class="eyebrow">Ushauri wa Leo</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rec-3d {advice['rec_style']}">
            <div class="rec-3d-action">{advice['icon']} {advice['action']}</div>
            <div class="rec-3d-headline">{advice['headline']}</div>
            <div class="rec-3d-purpose">{advice['purpose_text']}</div>
        </div>
        <span class="urg-chip {advice['urgency_class']}">
            ● Hatari: {advice['urgency_level']} — {advice['urgency_note']}
        </span>
        <span class="confidence-badge {conf_class}">
            ✓ Uhakika: {conf_label} — {conf_note}
        </span>
        """, unsafe_allow_html=True)

    with col_pay:
        st.markdown('<div class="eyebrow">Hesabu ya Fedha Yako</div>', unsafe_allow_html=True)

        future_class = (
            "pay-future-okoa"   if advice['saving_direction'] == 'okoa'
            else "pay-future-hasara" if advice['saving_direction'] == 'hasara'
            else "pay-future-stable"
        )
        st.markdown(f"""
        <div class="pay-grid">
            <div class="pay-3d pay-today">
                <div class="pb-label">Ukilipa LEO</div>
                <div class="pb-val">TZS {advice['tzs_now']:,.0f}</div>
                <div class="pb-note">{cur_display} {payment_amount:,.2f} × {current_rate:,.2f}</div>
            </div>
            <div class="pay-3d {future_class}">
                <div class="pb-label">Baada ya Siku {steps}</div>
                <div class="pb-val">TZS {advice['tzs_later']:,.0f}</div>
                <div class="pb-note">Tabiri: TZS {advice['avg_future']:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if advice['saving_direction'] == 'okoa':
            st.markdown(f"""
            <div class="savings-banner sb-okoa">
                <div><div class="sb-label">💡 Ukifuata ushauri, utaokoa:</div>
                     <div class="sb-sub">Tofauti kati ya leo na baada ya siku {steps}</div></div>
                <div class="sb-amount">TZS {advice['saving_amount']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        elif advice['saving_direction'] == 'hasara':
            st.markdown(f"""
            <div class="savings-banner sb-hasara">
                <div><div class="sb-label">⚠️ Ukisubiri badala ya kulipa sasa, utapoteza:</div>
                     <div class="sb-sub">Bei itapanda — lipa sasa kulinda fedha yako</div></div>
                <div class="sb-amount">TZS {advice['saving_amount']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="savings-banner sb-stable">
                <div><div class="sb-label">📊 Tofauti ya kulipa leo vs siku {steps}:</div>
                     <div class="sb-sub">Bei iko stable — hakuna haraka</div></div>
                <div class="sb-amount">TZS {abs(advice['tzs_now']-advice['tzs_later']):,.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Best-day insight cards ───────────────────────────
    st.markdown('<div class="eyebrow">📅 Siku Bora ya Kulipa — Uchambuzi wa Kihistoria</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown(f"""
        <div class="insight-3d ins-purple">
            <div class="ins-label">Siku Bora ya Wiki</div>
            <div class="ins-value">🗓 {bda['best_weekday']}</div>
            <div class="ins-note">Bei ya wastani iko chini zaidi siku hii</div>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""
        <div class="insight-3d ins-indigo">
            <div class="ins-label">Wiki Bora ya Mwezi</div>
            <div class="ins-value">📆 {bda['best_wom']}</div>
            <div class="ins-note">Bei iko chini zaidi katika wiki hii ya mwezi</div>
        </div>""", unsafe_allow_html=True)
    with ic3:
        st.markdown(f"""
        <div class="insight-3d ins-teal">
            <div class="ins-label">Akiba ya Wastani kwa Wiki</div>
            <div class="ins-value">💰 TZS {bda['avg_saving_wd']:,.0f}</div>
            <div class="ins-note">Tofauti ya bei kati ya siku bora na mbaya zaidi</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — MWENENDO WA BEI
# ══════════════════════════════════════════════════════════════════
with tab2:
    hist = df.tail(120)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(upper_band) + list(lower_band[::-1]),
        fill='toself',
        fillcolor='rgba(14,124,123,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True, name='Upeo wa Utabiri (±)',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=hist['date'], y=hist['mean'],
        name='Bei ya Kihistoria',
        line=dict(color='#0B5457', width=2.5),
        mode='lines',
        hovertemplate='%{x|%d/%m/%Y}<br>TZS %{y:,.2f}<extra></extra>'
    ))

    if 'MA_30' in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist['date'], y=hist['MA_30'],
            name='Wastani wa Siku 30',
            line=dict(color='#E8A33D', width=1.8, dash='dot'),
            mode='lines',
            hovertemplate='MA30: TZS %{y:,.2f}<extra></extra>'
        ))

    fig.add_trace(go.Scatter(
        x=future_dates, y=ensemble,
        name=f'Utabiri (Siku {steps})',
        line=dict(color='#10B981', width=3),
        mode='lines+markers',
        marker=dict(size=6, color='#10B981',
                    line=dict(color='white', width=2)),
        hovertemplate='%{x|%d/%m/%Y}<br>Tabiri: TZS %{y:,.2f}<extra></extra>'
    ))

    fig.add_vrect(
        x0=df['date'].iloc[-1], x1=future_dates[-1],
        fillcolor="rgba(14,124,123,0.05)", layer="below", line_width=0,
        annotation_text="Eneo la utabiri", annotation_position="top left",
        annotation_font_size=11, annotation_font_color="#0E7C7B"
    )

    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='rgba(14,124,123,0.20)', borderwidth=1),
        xaxis=dict(title='Tarehe', showgrid=True,
                   gridcolor='rgba(14,124,123,0.08)', zeroline=False,
                   tickfont=dict(size=11)),
        yaxis=dict(title=f'TZS kwa 1 {cur_display}', showgrid=True,
                   gridcolor='rgba(14,124,123,0.08)', zeroline=False,
                   tickfont=dict(size=11)),
        hovermode='x unified',
        plot_bgcolor='rgba(255,255,255,0.35)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Manrope, sans-serif', color='#0B2545'),
    )

    st.markdown('<div class="eyebrow">📈 Mwenendo wa Bei na Utabiri</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="eyebrow">Bei Inayotarajiwa — Kila Siku</div>', unsafe_allow_html=True)

    rows = []
    for i, (d, v) in enumerate(zip(future_dates, ensemble)):
        change  = v - current_rate
        chg_p   = change / current_rate * 100
        lo, hi  = lower_band[i], upper_band[i]
        rows.append({
            "Tarehe":                          d.strftime('%d/%m/%Y (%a)'),
            f"Bei ({cur_display}/TZS)":        f"TZS {v:,.2f}",
            "Upeo wa Chini":                   f"TZS {lo:,.2f}",
            "Upeo wa Juu":                     f"TZS {hi:,.2f}",
            "Mabadiliko":                      f"{'▲' if change>=0 else '▼'} {abs(change):.2f} ({chg_p:+.2f}%)",
            f"Malipo ({payment_amount:.0f} {cur_display})": f"TZS {payment_amount*v:,.0f}",
        })

    forecast_df = pd.DataFrame(rows)
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(translate('download_csv'), csv,
                       file_name=f"{cur_key}_forecast_{steps}days.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — UCHAMBUZI WA KINA
# ══════════════════════════════════════════════════════════════════
with tab3:
    with st.expander("🔧 Maelezo ya Kiufundi (kwa wataalamu)", expanded=False):
        st.write(f"Models used in this forecast: {list(all_forecasts.keys())}")
        cur_errors = all_models.get('_load_errors', {}).get(cur_key, {})
        if cur_errors:
            st.warning(f"Models that FAILED to load for {cur_display}:")
            for model_name, err in cur_errors.items():
                st.code(f"{model_name}: {err}")
        else:
            st.success("All models loaded successfully for this currency.")

        if all_forecasts:
            st.write("Each model's predicted rate for the final day of this forecast:")
            compare_rows_dbg = []
            for name, vals in all_forecasts.items():
                pct_change = (vals[-1] - current_rate) / current_rate * 100
                compare_rows_dbg.append({
                    "Model": name,
                    f"Predicted {cur_display}/TZS (day {steps})": round(float(vals[-1]), 2),
                    "% change vs today": round(float(pct_change), 2)
                })
            st.dataframe(pd.DataFrame(compare_rows_dbg), use_container_width=True, hide_index=True)
            st.caption("If these numbers disagree wildly with each other, treat the "
                       "ensemble average with caution — it means the models "
                       "aren't actually agreeing, they're just being averaged.")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="eyebrow">✅ Kwa Nini Niamini Ushauri Huu?</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="model-3d">
            <div class="mc-left">
                <div class="mn">Uhakika wa Utabiri wa Leo</div>
                <div class="md">{conf_note}</div>
            </div>
            <div class="mr">
                <div class="mel">Kiwango</div>
                <div class="mev">{conf_label}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="model-3d">
            <div class="mc-left">
                <div class="mn">Data ya Kihistoria</div>
                <div class="md">Bei za miaka kadhaa zilizopita za {cur_display}/TZS</div>
            </div>
            <div class="mr">
                <div class="mel">Idadi ya siku</div>
                <div class="mev">{len(df):,}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="eyebrow" style="margin-top:1rem;">ℹ️ Jinsi Inavyofanya Kazi</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explainer-3d">
            Mfumo unachambua miaka kadhaa ya data ya bei halisi kutabiri mwenendo
            wa siku zijazo. Utabiri unajaribiwa dhidi ya bei halisi za nyuma kabla
            ya kutumika, ili kuhakikisha unaaminika. Eneo la rangi kwenye grafu
            (Mwenendo wa Bei) linaonyesha pengo la kutokuwa na uhakika — eneo
            dogo linamaanisha utabiri wenye uhakika zaidi. Ushauri ni kwa mtu
            anayetaka <strong>KULIPA</strong> kwa sarafu ya kigeni.
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(f'<div class="eyebrow">{translate("compare_title")}</div>', unsafe_allow_html=True)

        comparison_rows = []
        for label, (display, key) in CURRENCIES.items():
            od   = all_data[display]
            cv   = float(od['mean'].iloc[-1])
            pv   = float(od['mean'].iloc[-2])
            chg  = cv - pv
            chgp = chg / pv * 100
            comparison_rows.append({"display":display,"raw":cv,
                "fmt":f"{cv:,.2f}","chg":f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f} ({chgp:+.2f}%)"})

        best = min(comparison_rows, key=lambda x: x['raw'])
        cards_html = '<div class="cur-cards">'
        for row in comparison_rows:
            is_best = row['display'] == best['display']
            badge_html = ''
            if is_best:
                badge_html = '<div style="font-size:0.72rem;font-weight:700;color:#0E9F6E;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem">✅ Nafuu Zaidi</div>'

            card_html = '<div class="cur-card' + (' best-cur' if is_best else '') + '">'
            card_html += badge_html
            card_html += '<div class="cur-row"><span class="cur-lbl">Sarafu</span><span class="cur-val">' + row['display'] + '</span></div>'
            card_html += '<div class="cur-row"><span class="cur-lbl">Bei (TZS)</span><span class="cur-val">' + row['fmt'] + '</span></div>'
            card_html += '<div class="cur-row"><span class="cur-lbl">Mabadiliko</span><span class="cur-val">' + row['chg'] + '</span></div>'
            card_html += '</div>'

            cards_html += card_html
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<div class="eyebrow" style="margin-top:1rem;">📊 Mwenendo wa Siku 30 Zilizopita</div>', unsafe_allow_html=True)
        recent = df.tail(30)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=recent['date'], y=recent['mean'], name='Bei',
            line=dict(color='#0E7C7B', width=2.5),
            fill='tozeroy', fillcolor='rgba(14,124,123,0.08)',
            mode='lines',
            hovertemplate='%{x|%d/%m/%Y}<br>TZS %{y:,.2f}<extra></extra>'
        ))
        fig2.update_layout(
            height=240,
            margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor='rgba(14,124,123,0.08)', zeroline=False, tickfont=dict(size=10)),
            plot_bgcolor='rgba(255,255,255,0.35)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Manrope, sans-serif', color='#0B2545'),
        )
        st.plotly_chart(fig2, use_container_width=True)