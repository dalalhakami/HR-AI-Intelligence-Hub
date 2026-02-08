import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. إعدادات الصفحة والهوية البصرية
st.set_page_config(page_title="Strategic Workforce Intelligence", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #1E293B, #0F172A, #020617); }
    div[data-testid="stMetric"] { 
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 245, 255, 0.2);
        padding: 30px; border-radius: 20px;
    }
    .rec-box { 
        background: rgba(0, 245, 255, 0.05); 
        padding: 15px; border-radius: 12px; 
        border-right: 5px solid #00F5FF; margin-bottom: 10px; color: #F8FAFC; 
    }
    h1 { background: linear-gradient(to right, #F8FAFC, #00F5FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900 !important; }
    </style>
    """, unsafe_allow_html=True)

# 2. وظيفة تحميل البيانات وتدريب خوارزمية Poisson Regression
@st.cache_resource
def initialize_engine():
    try:
        file_path = "Resigned Report Date Range.xlsx"
        df = pd.read_excel(file_path)
        df["تاريخ انتهاء الخدمة"] = pd.to_datetime(df["تاريخ انتهاء الخدمة"], errors="coerce")
        df = df.dropna(subset=["تاريخ انتهاء الخدمة"]).copy()
        
        # هندسة الميزات للتنبؤ
        df["year"] = df["تاريخ انتهاء الخدمة"].dt.year
        df["month"] = df["تاريخ انتهاء الخدمة"].dt.month
        
        # بناء الأنبوب البرمجي (Pipeline)
        cat_features = ["الجهة", "الجنسية"]
        transformer = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough")
        model = Pipeline([("prep", transformer), ("reg", PoissonRegressor(alpha=0.1))])
        
        # تدريب الموديل
        X = df[["year", "month", "الجهة", "الجنسية"]].fillna("Unknown")
        y = df.groupby(["year", "month", "الجهة", "الجنسية"]).size().reindex(X.index, fill_value=1)
        model.fit(X, y)
        
        return df, model
    except Exception as e:
        return None, str(e)

df, model_or_error = initialize_engine()

# التحقق من وجود الملف
if df is None:
    st.error(f"⚠️ تأكدي من وجود ملف البيانات بجانب الكود باسم: Resigned Report Date Range.xlsx")
    st.stop()

# 3. بناء واجهة المستخدم (UI)
st.title("Strategic Workforce Intelligence Hub")
st.caption(f"نظام التحليل التنبئي المدعوم بالذكاء الاصطناعي • {datetime.now().strftime('%H:%M')}")

with st.sidebar:
    st.markdown("### ✨ Strategy Shortcuts")
    btn_analysis = st.button("📊 التحليل الاستراتيجي والحلول")
    btn_forecast = st.button("🔮 النمذجة التنبؤية القادمة")
    
    st.markdown("---")
    st.markdown("### 🤖 مساعد القرار الذكي")
    query = st.chat_input("اسألي عن أي تفاصيل إضافية...")
    
    # توقيع دلال حكمي
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align: right; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 20px;">
            <p style="color: #00F5FF; font-weight: bold; margin-bottom: 0;">إعداد</p>
            <p style="color: white; font-size: 20px; font-weight: 900;">دلال حكمي</p>
            <p style="color: rgba(255,255,255,0.5); font-size: 12px;">dalal3021@gmail.com</p>
        </div>
    """, unsafe_allow_html=True)

# 4. منطق عرض النتائج
if btn_analysis or (query and "حلل" in query):
    st.markdown("---")
    l, r = st.columns([2, 1])
    with l:
        st.markdown("#### 🎯 مؤشرات الأداء الحيوية")
        c1, c2 = st.columns(2)
        top_dept = df["الجهة"].mode()[0]
        c1.metric("القطاع الأكثر تسرباً", top_dept)
        saudi_count = df[df["الجنسية"].str.contains("سعودي", na=False)].shape[0]
        c2.metric("كفاءة التوطين", f"{(saudi_count/len(df))*100:.1f}%")
        
        st.markdown("#### 💡 مبادرات تعزيز الاستبقاء")
        recs = [f"🚀 خطة تحسين بيئة العمل في {top_dept}", "🎯 مراجعة برامج الولاء الوظيفي", "📅 تفعيل المقابلات الاسترجاعية"]
        for rec in recs: st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)
    with r:
        st.markdown("#### 📄 آخر سجلات البيانات")
        st.dataframe(df[["الجهة", "الجنسية", "تاريخ انتهاء الخدمة"]].tail(10), use_container_width=True)

elif btn_forecast or (query and "توقع" in query):
    st.markdown("---")
    st.markdown("#### 📈 التوقعات المستقبلية (6 أشهر قادمة)")
    last_date = df["تاريخ انتهاء الخدمة"].max()
    future_dates = pd.date_range(last_date, periods=7, freq="MS")[1:]
    
    preds = []
    for d in future_dates:
        p = model_or_error.predict(pd.DataFrame([[d.year, d.month, "Unknown", "Unknown"]], 
                                    columns=["year", "month", "الجهة", "الجنسية"]))[0]
        preds.append(int(p))
    
    chart_df = pd.DataFrame({"الشهر": [d.strftime('%Y-%m') for d in future_dates], "التوقع": preds})
    st.line_chart(chart_df.set_index("الشهر"), color="#00F5FF")
    st.info("💡 تم استخدام خوارزمية Poisson Regression لنمذجة هذه التوقعات بناءً على الأنماط التاريخية.")

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info(f"👋 مرحباً بك .. المنصة جاهزة لتحليل البيانات التنبؤية، اختاري نوع التحليل من القائمة الجانبية.")