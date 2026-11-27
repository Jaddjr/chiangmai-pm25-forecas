"""
เว็บแอปพยากรณ์ PM2.5 เชียงใหม่ล่วงหน้า 1-3 วัน (Streamlit)
- ดึงข้อมูลย้อนหลังและพยากรณ์อากาศจาก Open-Meteo (ฟรี ไม่ต้องมี key)
- เทรนโมเดล Gradient Boosting ตอนเปิดแอป (แคช 6 ชั่วโมง)
- ค่า PM2.5 เป็นข้อมูลจากแบบจำลอง CAMS ไม่ใช่ค่าวัดจากสถานีจริง
"""
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from sklearn.ensemble import HistGradientBoostingRegressor

# ----------------------------------------------------------------- ตั้งค่า
LAT, LON = 18.79, 98.98
START = "2023-01-01"
HORIZONS = [1, 2, 3]
# ความคลาดเคลื่อนเฉลี่ย (MAE) จากการทดสอบ walk-forward ปี 2024-2026 ของโมเดลนี้
MAE_BT = {1: 3.83, 2: 4.95, 3: 5.32}
MAE_BASE = {1: 4.05, 2: 5.64, 3: 6.32}   # baseline: เดาว่าเท่าวันนี้

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTFC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FC_URL = "https://api.open-meteo.com/v1/forecast"
WX_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,surface_pressure"
FC_VARS = WX_VARS + ",boundary_layer_height"

TH_MONTH = ["", "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
            "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
TH_DAY = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]

st.set_page_config(page_title="พยากรณ์ PM2.5 เชียงใหม่", page_icon="🌫️", layout="centered")


def now_bkk():
    return pd.Timestamp.now(tz="Asia/Bangkok").normalize().tz_localize(None)


def th_date(d):
    d = pd.Timestamp(d)
    return f"{TH_DAY[d.weekday()]} {d.day} {TH_MONTH[d.month]}"


def aqi_level(v):
    """เกณฑ์ AQI PM2.5 เฉลี่ย 24 ชม. (ประเทศไทย ตั้งแต่ปี 2566)"""
    if v <= 15:
        return "ดีมาก", "#3BB2E8", "ทำกิจกรรมกลางแจ้งได้ตามปกติ"
    if v <= 25:
        return "ดี", "#3FB950", "ทำกิจกรรมกลางแจ้งได้ตามปกติ"
    if v <= 37.5:
        return "ปานกลาง", "#F2C94C", "กลุ่มเสี่ยงควรลดกิจกรรมกลางแจ้งที่ใช้แรงนาน ๆ"
    if v <= 75:
        return "เริ่มมีผลต่อสุขภาพ", "#F2994A", "ควรสวมหน้ากากเมื่ออยู่กลางแจ้ง กลุ่มเสี่ยงควรเลี่ยงกิจกรรมกลางแจ้ง"
    return "มีผลต่อสุขภาพ", "#EB5757", "ควรอยู่ในอาคาร สวมหน้ากากเมื่อจำเป็นต้องออกนอกบ้าน"


# ----------------------------------------------------------------- ดึงข้อมูล
def fetch(url, params, start, end):
    frames = []
    s = pd.Timestamp(start)
    e_all = pd.Timestamp(end)
    while s <= e_all:
        e = min(s + pd.DateOffset(years=1) - pd.Timedelta(days=1), e_all)
        p = dict(params, start_date=s.date().isoformat(), end_date=e.date().isoformat())
        r = requests.get(url, params=p, timeout=90)
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["hourly"]))
        s = e + pd.Timedelta(days=1)
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out["time"] = pd.to_datetime(out["time"])
    return out.set_index("time")


def hourly_to_daily_wx(h):
    cols = [
        h["temperature_2m"].resample("D").mean().rename("temp"),
        h["relative_humidity_2m"].resample("D").mean().rename("rh"),
        h["precipitation"].resample("D").sum().rename("rain"),
        h["wind_speed_10m"].resample("D").mean().rename("wind"),
        h["surface_pressure"].resample("D").mean().rename("pressure"),
    ]
    if "boundary_layer_height" in h.columns:
        cols.append(h["boundary_layer_height"].resample("D").mean().rename("blh"))
    return pd.concat(cols, axis=1)


@st.cache_data(ttl=6 * 3600, show_spinner="กำลังดึงข้อมูลย้อนหลัง...")
def load_history(end_str):
    base = {"latitude": LAT, "longitude": LON, "timezone": "Asia/Bangkok"}
    aq = fetch(AQ_URL, dict(base, hourly="pm2_5"), START, end_str)
    wx = fetch(ARCHIVE_URL, dict(base, hourly=WX_VARS), START, end_str)
    try:
        hf = fetch(HISTFC_URL, dict(base, hourly=FC_VARS), START, end_str)
    except Exception:
        hf = fetch(HISTFC_URL, dict(base, hourly=WX_VARS), START, end_str)
    daily = pd.concat([aq["pm2_5"].resample("D").mean().rename("pm25"),
                       hourly_to_daily_wx(wx)], axis=1)
    daily = daily.interpolate(limit=3).dropna(subset=["pm25"])
    return daily, hourly_to_daily_wx(hf)


@st.cache_data(ttl=1800, show_spinner="กำลังดึงข้อมูลล่าสุด...")
def load_live():
    base = {"latitude": LAT, "longitude": LON, "past_days": 30,
            "forecast_days": 5, "timezone": "Asia/Bangkok"}
    aq = requests.get(AQ_URL, params=dict(base, hourly="pm2_5"), timeout=90).json()
    wx = requests.get(FC_URL, params=dict(base, hourly=FC_VARS), timeout=90).json()
    to_df = lambda j: (pd.DataFrame(j["hourly"])
                       .assign(time=lambda x: pd.to_datetime(x["time"])).set_index("time"))
    return to_df(aq), to_df(wx)


# ----------------------------------------------------------------- ฟีเจอร์และโมเดล
def make_features(d):
    f = pd.DataFrame(index=d.index)
    for k in [0, 1, 2, 3, 7]:
        f[f"pm25_lag{k}"] = d["pm25"].shift(k)
    f["pm25_ma3"] = d["pm25"].rolling(3).mean()
    f["pm25_ma7"] = d["pm25"].rolling(7).mean()
    f["pm25_std7"] = d["pm25"].rolling(7).std()
    f["pm25_diff1"] = d["pm25"].diff(1)
    for c in ["temp", "rh", "rain", "wind", "pressure"]:
        f[c] = d[c]
        f[f"{c}_ma3"] = d[c].rolling(3).mean()
    f["rain_7d"] = d["rain"].rolling(7).sum()
    doy = d.index.dayofyear
    f["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return f


def make_fc_features(fcd, h):
    f = pd.DataFrame(index=fcd.index)
    for c in fcd.columns:
        f[f"fc_{c}_target"] = fcd[c].shift(-h)
    for c in [c for c in ["wind", "rain", "blh"] if c in fcd.columns]:
        f[f"fc_{c}_mean"] = sum(fcd[c].shift(-k) for k in range(1, h + 1)) / h
    return f


@st.cache_resource(ttl=6 * 3600, show_spinner="กำลังเทรนโมเดล...")
def train_models(end_str):
    daily, fc_daily = load_history(end_str)
    X = make_features(daily)
    models = {}
    for h in HORIZONS:
        y = (daily["pm25"].shift(-h) - daily["pm25"]).rename("y")
        d = pd.concat([X, make_fc_features(fc_daily, h).reindex(daily.index), y], axis=1)
        d = d.dropna(subset=list(X.columns) + ["y"])
        cols = [c for c in d.columns if c != "y"]
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_depth=4,
            l2_regularization=5.0, random_state=42,
        ).fit(d[cols], d["y"])
        models[h] = (m, cols)
    return models, list(fc_daily.columns)


def forecast(models, fc_cols):
    today = now_bkk()
    aq_l, wx_l = load_live()
    wx_daily = hourly_to_daily_wx(wx_l)
    cams_daily = aq_l["pm2_5"].resample("D").mean().rename("pm25")
    live = pd.concat([cams_daily, wx_daily[["temp", "rh", "rain", "wind", "pressure"]]], axis=1)
    live_past = live.loc[:today].interpolate(limit=3)
    X_live = make_features(live_past).dropna()
    row_day = X_live.index[-1]
    now_val = float(live_past.loc[row_day, "pm25"])
    fc_use = [c for c in fc_cols if c in wx_daily.columns]
    out = {}
    for h in HORIZONS:
        model, cols = models[h]
        x = pd.concat([X_live.loc[[row_day]],
                       make_fc_features(wx_daily[fc_use], h).loc[[row_day]]],
                      axis=1).reindex(columns=cols)
        out[h] = max(0.0, float(model.predict(x)[0] + now_val))
    return row_day, now_val, out, live_past["pm25"], cams_daily


# ----------------------------------------------------------------- ส่วนแสดงผล
def card(title, val, mae=None):
    label, color, _ = aqi_level(val)
    err = f"<div style='font-size:.8rem;opacity:.7'>คลาดเคลื่อนปกติ ±{mae:.0f}</div>" if mae else ""
    st.markdown(
        f"""<div style="background:{color}22;border-left:6px solid {color};
        padding:12px 14px;border-radius:8px;height:100%">
        <div style="font-size:.9rem;opacity:.8">{title}</div>
        <div style="font-size:2rem;font-weight:700;line-height:1.2">{val:.0f}
        <span style="font-size:1rem;font-weight:400">µg/m³</span></div>
        <div style="font-weight:600">{label}</div>{err}</div>""",
        unsafe_allow_html=True,
    )


st.title("🌫️ พยากรณ์ PM2.5 เชียงใหม่")
st.caption("ล่วงหน้า 1-3 วัน ด้วยโมเดล Machine Learning จากข้อมูลย้อนหลังและพยากรณ์อากาศ")

try:
    end_str = (now_bkk() - pd.Timedelta(days=5)).date().isoformat()
    models, fc_cols = train_models(end_str)
    row_day, now_val, preds, past, cams = forecast(models, fc_cols)
except Exception as ex:
    st.error("ดึงข้อมูลหรือสร้างพยากรณ์ไม่สำเร็จ ลองรีเฟรชอีกครั้งในอีกสักครู่")
    st.exception(ex)
    st.stop()

# การ์ดวันนี้ + พยากรณ์
card(f"ล่าสุด {th_date(row_day)}", now_val)
st.write("")
cols = st.columns(3)
for c, h in zip(cols, HORIZONS):
    with c:
        card(f"{th_date(row_day + pd.Timedelta(days=h))} (+{h} วัน)", preds[h], MAE_BT[h])

worst = max(preds.values())
st.info(f"**คำแนะนำ:** {aqi_level(worst)[2]} (อ้างอิงค่าสูงสุดที่พยากรณ์ใน 3 วันข้างหน้า)")

# กราฟ
fc_df = pd.DataFrame({
    "date": [row_day] + [row_day + pd.Timedelta(days=h) for h in HORIZONS],
    "value": [now_val] + [preds[h] for h in HORIZONS],
    "lo": [now_val] + [max(0, preds[h] - MAE_BT[h]) for h in HORIZONS],
    "hi": [now_val] + [preds[h] + MAE_BT[h] for h in HORIZONS],
})
past_df = past.loc[row_day - pd.Timedelta(days=30):row_day].rename("pm25").reset_index()
past_df.columns = ["date", "pm25"]
cams_f = cams[cams.index > row_day].rename("pm25").reset_index()
cams_f.columns = ["date", "pm25"]
cams_f = pd.concat([past_df.tail(1), cams_f.head(4)], ignore_index=True)

x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b"))
y = alt.Y("pm25:Q", title="PM2.5 (µg/m³)", scale=alt.Scale(zero=True))
rules = alt.Chart(pd.DataFrame({"v": [25, 37.5]})).mark_rule(
    strokeDash=[2, 4], color="#999").encode(y="v:Q")
chart = (
    alt.Chart(past_df).mark_line(color="#555", strokeWidth=2).encode(x=x, y=y)
    + alt.Chart(cams_f).mark_line(color="#888", strokeDash=[5, 4]).encode(x=x, y=y)
    + alt.Chart(fc_df).mark_area(opacity=0.2, color="#d62728").encode(
        x="date:T", y=alt.Y("lo:Q"), y2="hi:Q")
    + alt.Chart(fc_df).mark_line(color="#d62728", point=True, strokeWidth=2).encode(
        x="date:T", y=alt.Y("value:Q"))
    + rules
).properties(height=300)
st.subheader("แนวโน้ม 30 วันล่าสุดและพยากรณ์")
st.altair_chart(chart, use_container_width=True)
st.caption("เส้นเทาทึบ = ค่าย้อนหลัง · เส้นแดง = พยากรณ์ของโมเดลนี้ (แถบ = ช่วงคลาดเคลื่อนปกติ) · "
           "เส้นเทาประ = พยากรณ์ของ CAMS (Open-Meteo) เทียบ · เส้นประแนวนอน = 25 และ 37.5 µg/m³")

with st.expander("โมเดลแม่นแค่ไหน"):
    st.markdown(
        "ทดสอบแบบ walk-forward ปี 2024-2026 (เทรนด้วยข้อมูลก่อนปีที่ทดสอบเท่านั้น) "
        "เทียบกับ baseline คือการเดาว่าเท่ากับวันนี้ (ค่า MAE ยิ่งต่ำยิ่งดี)"
    )
    st.dataframe(pd.DataFrame({
        "ล่วงหน้า": [f"+{h} วัน" for h in HORIZONS],
        "MAE โมเดล (µg/m³)": [MAE_BT[h] for h in HORIZONS],
        "MAE baseline (µg/m³)": [MAE_BASE[h] for h in HORIZONS],
        "ดีขึ้น": [f"{(1 - MAE_BT[h] / MAE_BASE[h]) * 100:.0f}%" for h in HORIZONS],
    }), hide_index=True, use_container_width=True)
    st.markdown(
        "ในช่วงหน้าแล้ง (ม.ค.-เม.ย.) โมเดลช่วยได้ชัดเจนที่สุดที่ +3 วัน "
        "ส่วน +1 วันยังไม่ชนะ baseline เพราะค่าเปลี่ยนช้าในช่วงสั้น ๆ"
    )

with st.expander("ข้อจำกัดและแหล่งข้อมูล"):
    st.markdown(
        "- ค่า PM2.5 มาจากแบบจำลอง **CAMS** ผ่าน Open-Meteo **ไม่ใช่เครื่องวัดจากสถานีจริง** "
        "จึงอาจต่างจากที่เห็นในแอปอย่าง Air4Thai หรือ AirVisual\n"
        "- โมเดลยังไม่มีข้อมูลจุดความร้อนไฟป่า ช่วงหน้าแล้ง (ม.ค.-เม.ย.) ความคลาดเคลื่อนจึงสูงกว่าช่วงอื่น\n"
        "- ใช้เพื่อการศึกษาและประกอบการวางแผนเท่านั้น ไม่ใช่ข้อมูลทางการ "
        "ค่าตรวจวัดอย่างเป็นทางการดูที่ [Air4Thai](http://air4thai.pcd.go.th)\n"
        "- ข้อมูลสภาพอากาศและ PM2.5 จาก [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0)"
    )

if st.button("รีเฟรชข้อมูล"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
