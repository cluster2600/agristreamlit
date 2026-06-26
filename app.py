import streamlit as st
import sqlite3
import requests
import numpy as np
import pickle
from irrigation import get_irrigation_plan
from PIL import Image
import streamlit.components.v1 as components
import joblib

# Load trained crop model
with open("xgb_crop_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as le_file:
    le = pickle.load(le_file)

# Descriptive mappings
SOIL_MOISTURE_MAP = {
    "Very Dry": (0, 20),
    "Dry": (21, 40),
    "Moist": (41, 70),
    "Wet": (71, 100)
}

TEMPERATURE_MAP = {
    "Cold": (0, 15),
    "Warm": (16, 30),
    "Hot": (31, 50)
}

HUMIDITY_MAP = {
    "Low": (0, 40),
    "Medium": (41, 70),
    "High": (71, 100)
}

# Convert descriptive input to numerical
def to_mid(value):
    if isinstance(value, int):
        return value
    if value in SOIL_MOISTURE_MAP:
        return sum(SOIL_MOISTURE_MAP[value]) // 2
    if value in TEMPERATURE_MAP:
        return sum(TEMPERATURE_MAP[value]) // 2
    if value in HUMIDITY_MAP:
        return sum(HUMIDITY_MAP[value]) // 2
    try:
        return int(value)
    except:
        return None

# FAQ chatbot
def get_farming_info(query):
    conn = sqlite3.connect("farming_data.db")
    cursor = conn.cursor()
    
    query = query.lower()
    cursor.execute("SELECT question, response FROM farming_info")
    data = cursor.fetchall()
    
    best_match = None
    highest_score = 0

    for question, response in data:
        score = sum(1 for word in query.split() if word in question.lower() or word in response.lower())
        if score > highest_score:
            highest_score = score
            best_match = response

    conn.close()
    return best_match if best_match else "⚠️ No relevant farming info found."

# Crop prediction logic
def predict_crop(input_features):
    input_array = np.array([input_features]).reshape(1, -1)
    predicted_label = model.predict(input_array)[0]
    predicted_crop = le.inverse_transform([predicted_label])[0]
    return predicted_crop

# UI
st.set_page_config(page_title="AgriAssistant", page_icon="🌾", layout="centered")

# Modern-SaaS styling. Colours come from .streamlit/config.toml; this restyles
# typography, the Streamlit chrome, tabs, inputs, buttons and metric cards.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, button, input, select, textarea {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Hide Streamlit's default chrome for a cleaner product feel */
    [data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"] {
        display: none !important;
    }

    .block-container { max-width: 920px; padding-top: 2rem; padding-bottom: 4rem; }

    /* Typography */
    h1 { font-weight: 800; letter-spacing: -0.03em; }
    h2 { font-weight: 700; letter-spacing: -0.02em; }
    h3 { font-weight: 600; }
    p, li { color: #3a463a; line-height: 1.6; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 60%, #66bb6a 100%);
        border-radius: 24px; padding: 48px 40px; color: #fff; margin-bottom: 28px;
        box-shadow: 0 12px 30px rgba(46,125,50,0.25);
    }
    .hero h1 { color: #fff; font-size: 2.6rem; margin: 0 0 8px 0; }
    .hero p { color: rgba(255,255,255,0.92); font-size: 1.15rem; margin: 0; }
    .hero .eyebrow {
        text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.78rem;
        font-weight: 700; color: rgba(255,255,255,0.8); margin-bottom: 14px;
    }

    /* Tab bar: clean underline pills */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #e6e9e6; }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px 8px 0 0; padding: 10px 18px;
        font-weight: 600; color: #6b776b;
    }
    .stTabs [aria-selected="true"] { color: #2e7d32; }

    /* Inputs: rounded, white, subtle border */
    [data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
        border-radius: 10px !important; border-color: #dfe4df !important;
        background: #fff !important;
    }

    /* Buttons: solid green, pill */
    .stButton > button {
        background: #2e7d32; color: #fff; border: none; border-radius: 12px;
        font-weight: 600; padding: 0.6rem 1.5rem;
        box-shadow: 0 4px 12px rgba(46,125,50,0.22); transition: all .15s ease;
    }
    .stButton > button:hover {
        background: #256528; color: #fff; transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(46,125,50,0.3);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e6e9e6; border-radius: 16px;
        padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] p { color: #6b776b; font-weight: 600; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: #1b5e20; font-weight: 800; }

    /* Alert / result boxes as soft cards */
    .stAlert { border-radius: 14px; border: 1px solid rgba(46,125,50,0.15); }

    /* Section card for the detailed plan text */
    .plan-card {
        background: #f6faf4; border: 1px solid #e1ece0; border-radius: 16px;
        padding: 18px 22px; margin-top: 8px; color: #2f3b2f; line-height: 1.7;
    }
    .plan-card .pill {
        display: inline-block; background: #e8f5e9; color: #2e7d32;
        border-radius: 999px; padding: 3px 12px; font-weight: 600; font-size: 0.85rem;
        margin: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏡 Home", "🌱 Crop Recommendation", "💧 Irrigation", "Chat with AgriBot 🤖", "🤖 FAQ Chatbot", "🌾 Yield Prediction"])

# Home Tab
with tabs[0]:
    st.markdown("""
        <div class="hero">
            <div class="eyebrow">🌾 AgriAssistant</div>
            <h1>Smart farming, powered by AI</h1>
            <p>Crop recommendations, water-aware irrigation plans and climate-driven
            yield forecasts — in one clean dashboard built for farmers.</p>
        </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="plan-card"><h3>🌱 Crop fit</h3>
        <p>AI-picked crops from your soil nutrients, pH and local climate.</p></div>""",
        unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="plan-card"><h3>💧 Irrigation</h3>
        <p>Turn soil & weather into litres, passes and runtime for your field.</p></div>""",
        unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="plan-card"><h3>🌾 Yield</h3>
        <p>See how climate stress could shift your tons-per-hectare.</p></div>""",
        unsafe_allow_html=True)

    st.markdown("### Watch our Introduction Video 🎥")
    components.html(
        """
        <iframe width="560" height="315" src="https://www.youtube.com/embed/NMhoUELo3Cc" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        """,
        height=350
    )

# Crop Recommendation Tab
with tabs[1]:
    st.subheader("🌱 Enter Soil & Climate Data")

    st.markdown("## How the Crop Recommendation System Works 🧠")
    st.markdown("""
    Our system is powered by a machine learning model trained on a large dataset that includes various environmental factors such as soil composition, temperature, humidity, and rainfall.
    """)

    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200)
        P = st.number_input("Phosphorus (P)", 0, 200)
        K = st.number_input("Potassium (K)", 0, 200)
        pH = st.number_input("Soil pH", 0.0, 14.0)

    with col2:
        temp = st.number_input("Temperature (°C)", 0.0, 50.0)
        hum = st.number_input("Humidity (%)", 0.0, 100.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)

    if st.button("🚀 Predict Crop"):
        features = [N, P, K, temp, hum, pH, rainfall]
        try:
            crop = predict_crop(features)
            st.markdown(f"""
                <div class="plan-card" style="text-align:center;">
                    <div style="color:#6b776b;font-weight:600;font-size:0.85rem;
                        text-transform:uppercase;letter-spacing:0.08em;">Best-fit crop</div>
                    <div style="font-size:2.2rem;font-weight:800;color:#1b5e20;
                        text-transform:capitalize;margin-top:4px;">🌾 {crop}</div>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# Irrigation Tab
with tabs[2]:
    st.subheader("💧 Get Irrigation Advice")

    crop_categories = {
        "Cereal Crops": ["Maize", "Wheat", "Barley"],
        "Vegetables": ["Tomato", "Potato", "Cabbage"],
        "Fruits": ["Strawberry", "Mango", "Banana"],
        "Legumes": ["Beans", "Peas"],
        "Root Crops": ["Carrot", "Cassava", "Beetroot"]
    }

    category = st.selectbox("🌾 Select Crop Category", list(crop_categories.keys()))
    crop = st.selectbox("🌱 Select Specific Crop", crop_categories[category])

    soil = st.selectbox("Soil Moisture", list(SOIL_MOISTURE_MAP.keys()) + [10, 30, 50, 70])
    temp = st.selectbox("Temperature", list(TEMPERATURE_MAP.keys()) + [10, 25, 40])
    hum = st.selectbox("Humidity", list(HUMIDITY_MAP.keys()) + [20, 60, 80])

    st.markdown("#### 🌍 Field & Water (optional — for a concrete plan)")
    colA, colB = st.columns(2)
    with colA:
        terrain = st.selectbox("Terrain (slope)", ["flat", "gentle", "steep"],
                               help="flat = 0–2%, gentle = 2–8%, steep = >8%")
        soil_texture = st.selectbox("Soil texture", ["loam", "sandy", "clay"])
    with colB:
        area_ha = st.number_input("Field area (hectares)", 0.0, 1000.0, 0.0, step=0.1,
                                  help="0 = skip the water plan")
        debit_lpm = st.number_input("Water flow rate / débit (L/min)", 0.0, 100000.0, 0.0, step=10.0,
                                    help="0 = unknown")
    water_available_m3 = st.number_input("Water available (m³)", 0.0, 1000000.0, 0.0, step=1.0,
                                         help="0 = unknown / unlimited")

    if st.button("💧 Recommend Irrigation"):
        soil_val = to_mid(soil)
        temp_val = to_mid(temp)
        hum_val = to_mid(hum)

        if None in (soil_val, temp_val, hum_val):
            st.warning("⚠️ Please enter valid values.")
        else:
            plan = get_irrigation_plan(
                soil_val, temp_val, hum_val, crop,
                terrain=terrain,
                soil_texture=soil_texture,
                area_ha=area_ha or None,
                water_available_m3=water_available_m3 or None,
                debit_lpm=debit_lpm or None,
            )
            if plan.get("error"):
                st.error(plan["error"])
            else:
                st.markdown(f"#### Plan for {crop}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Sprinkling", f"{plan['sprinkling']:.0f}%", plan["level"].title())
                if plan["volume_m3"] is not None:
                    m2.metric("Water needed", f"{plan['volume_m3']:.1f} m³",
                              f"{plan['depth_mm']} mm depth")
                else:
                    m2.metric("Water depth", f"{plan['depth_mm']} mm", "add area for m³")
                if plan["minutes"] is not None:
                    m3.metric("Run time", f"{plan['minutes']:.0f} min",
                              f"{plan['passes']} pass(es)")
                else:
                    m3.metric("Passes", f"{plan['passes']}", "add débit for time")

                pills = "".join(f"<span class='pill'>{x}</span>" for x in plan["methods"])
                detail = (
                    f"<div class='plan-card'>{plan['message']}<br><br>"
                    f"🌍 <b>Terrain:</b> {plan['terrain']} — suitable methods: {pills}<br>"
                    f"🪨 <b>Soil:</b> {plan['soil_texture']} "
                    f"(≈{plan['cap_mm']:.1f} mm max per pass before runoff)"
                )
                if plan["passes"] > 1:
                    detail += (f"<br>⚠️ Split into <b>{plan['passes']} passes</b> "
                               f"to avoid runoff/erosion on {plan['terrain']} land.")
                if plan["minutes_per_pass"] is not None:
                    detail += (f"<br>⏱️ ≈{plan['minutes_per_pass']:.0f} min per pass "
                               f"at {plan['debit_lpm']:.0f} L/min.")
                detail += "</div>"
                st.markdown(detail, unsafe_allow_html=True)

                if plan["covered"] is False:
                    st.warning(f"🚱 Shortfall: you have {plan['water_available_m3']:.0f} m³ but "
                               f"need {plan['volume_m3']:.1f} m³ "
                               f"(missing {plan['shortfall_m3']:.1f} m³). "
                               f"Reduce area or irrigate partially.")
                elif plan["covered"] is True:
                    st.success(f"✅ Your {plan['water_available_m3']:.0f} m³ covers the need.")

# Chat with AgriBot
with tabs[3]:
    st.markdown("## 💬 Chat with AgriBot")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### 🤖 Your Farming Assistant")
        components.html(
            """
            <div id="chatbot-container" style="height: 600px; width: 100%;"></div>

            <script>
(function(){if(!window.chatbase||window.chatbase("getState")!=="initialized"){window.chatbase=(...arguments)=>{if(!window.chatbase.q){window.chatbase.q=[]}window.chatbase.q.push(arguments)};window.chatbase=new Proxy(window.chatbase,{get(target,prop){if(prop==="q"){return target.q}return(...args)=>target(prop,...args)}})}const onLoad=function(){const script=document.createElement("script");script.src="https://www.chatbase.co/embed.min.js";script.id="2cWA9yXY2No0Nkcu5iB6P";script.domain="www.chatbase.co";document.body.appendChild(script)};if(document.readyState==="complete"){onLoad()}else{window.addEventListener("load",onLoad)}})();
</script>

            <script src="https://www.chatbase.co/embed.min.js" id="chatbase-script" defer></script>
            """,
            height=540
        )
    with col2:
        st.markdown("### 📋 How to Use AgriBot")
        st.write("""
        You can ask questions like:
        - “What’s the best crop to grow this month in Tigray?”
        - “How do I deal with tomato pests naturally?”
        - “What irrigation system is ideal for small farms?”
        """)
        st.success("💡 Click the chat window below and 👈 start chatting with AgriBot on the left!")


# FAQ Tab
with tabs[4]:
    st.subheader("🤖 Ask the FAQ Bot")
    
    st.markdown("### 💡 Example Questions You Can Ask:")
    st.markdown("""
    - What is the best fertilizer for maize?
    - How often should I irrigate tomato plants?
    - What is the ideal pH for growing wheat?
    - How do I prevent pests on beans?
    - What temperature is best for banana farming?
    """)

    user_question = st.text_input("Ask a farming question")
    if st.button("🔍 Search Answer"):
        if user_question:
            answer = get_farming_info(user_question)
            st.info(f"🧠 Answer: {answer}")
        else:
            st.warning("❗ Please type a question.")

    # Show common Q&A
    st.markdown("### 📌 Frequently Asked Questions (FAQs)")

    conn = sqlite3.connect("farming_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT question, response FROM farming_info LIMIT 5")
    faqs = cursor.fetchall()
    conn.close()

    for i, (q, a) in enumerate(faqs, start=1):
        st.markdown(f"**Q{i}: {q}**")
        st.markdown(f"🟢 *A{i}: {a}*")


# start tab 5 which is about climate impact on agriculture, a research based ML project
# Load model and encoder once.
# NB: kept under distinct names so they don't overwrite the crop-recommendation
# `model`/`le` (xgb) loaded at the top — those are used by predict_crop().
@st.cache_resource
def load_yield_model():
    yield_model = joblib.load("random_forest_model.joblib")
    yield_le = joblib.load("label_encoder.joblib")
    return yield_model, yield_le

yield_model, yield_le = load_yield_model()

# =======================
# Tab 5: Prediction Page
# =======================
# Last tab - Yield Prediction
with tabs[5]:
						st.title("🌾 Climate Impact Prediction on Crop Yield")
						st.markdown("""
---
### 🌍 Why This Matters

FarmBuddy is designed to help agriculture adapt, grow, and succeed — even in the face of climate uncertainty.

By integrating climate-aware crop yield prediction, the platform evolves into a **scientific, decision-support tool**, supporting farmers, researchers, and agri-tech innovators who need to understand and respond to environmental change.

- It helps farmers **understand how climate stress affects their yield**.
- It supports **early planning** and **policy development**.
- It shows that FarmBuddy is **built on real research**, not just AI tricks.

> 🎯 **This bridges the gap between advanced modeling and local impact.**
""")

						st.markdown("Use the form below to predict crop yield based on climate and agricultural inputs.")

						with st.form("yield_prediction_form"):
							st.markdown("### 🌍 Climate & Region")
							year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
							region = st.selectbox("Region", yield_le.classes_)

							st.markdown("### 🌡️ Climate Features")
							temp = st.slider("Average Temperature (°C)", 0.0, 50.0, 25.0)
							rain = st.slider("Total Precipitation (mm)", 0.0, 2000.0, 500.0)
							events = st.number_input("Extreme Weather Events (annual)", min_value=0, value=2)
							co2 = st.slider("CO₂ Emissions (metric tons)", 0.0, 100.0, 30.0)

							st.markdown("### 🌾 Farming Inputs")
							irrigation = st.slider("Irrigation Access (%)", 0, 100, 60)
							fertilizer = st.slider("Fertilizer Use (kg/ha)", 0.0, 300.0, 100.0)
							pesticide = st.slider("Pesticide Use (kg/ha)", 0.0, 50.0, 10.0)  # Not yet used in model
							soil_health = st.slider("Soil Health Index (0–100)", 0.0, 100.0, 70.0)

							submitted = st.form_submit_button("📊 Predict Crop Yield")

							if submitted:
								region_encoded = yield_le.transform([region])[0]
								temp_x_rain = temp * rain
								weather_impact = events * temp
								temp_sq = temp ** 2
								rain_sq = rain ** 2

								# Build feature array in the order model expects
								X_input = np.array([[
									temp,
									rain,
									events,
									co2,
									irrigation,
									fertilizer,
									soil_health,
									region_encoded,
									temp_x_rain,
									weather_impact,
									temp_sq,
									rain_sq
								]])

								# Predict
								prediction = yield_model.predict(X_input)[0]

								st.success(f"✅ **Predicted Crop Yield: {prediction:.2f} tons/hectare**")

								# 👇 Research-based and human-readable interpretation
								st.markdown(f"""
### 📈 What does this mean?

Based on the climate and farming inputs you provided, the expected crop yield is approximately  
**{prediction:.2f} metric tons per hectare**.

**This value reflects how climate factors like temperature, rainfall, and extreme events — along with farming practices such as irrigation, fertilizer use, and soil health — impact the productivity of farmland.**

🧠 **Key Insight**:  
- Crop yields around **2.0–4.0 tons/ha** are average for staple crops like wheat and maize.
- Values below **2.0 tons/ha** may indicate climate stress, poor soil health, or limited irrigation.
- Higher values may suggest optimal growing conditions or improved agricultural inputs.

This prediction is generated using a machine learning model trained on simulated data that reflects real-world agricultural patterns, helping us explore **how climate change could affect food production**.
""")

								# 👇 Optional: detailed input summary
								with st.expander("📋 View Input Summary"):
									st.markdown("""
Here are the climate and farming inputs used to generate the prediction:
""")
									st.json({
										"Region": region,
										"Year": year,
										"Temperature (°C)": temp,
										"Precipitation (mm)": rain,
										"CO₂ Emissions (MT)": co2,
										"Extreme Events": events,
										"Irrigation (%)": irrigation,
										"Fertilizer (kg/ha)": fertilizer,
										"Pesticide (kg/ha)": pesticide,
										"Soil Health Index": soil_health
									})

