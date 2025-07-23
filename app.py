import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta


#this is the app we are presenting
# ---- SESSION STATE LOGIN INIT ----
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "login_attempted" not in st.session_state:
    st.session_state["login_attempted"] = False

# ---- LOGIN SCREEN ----
def login():
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🔐 AgriLens Login</h1>", unsafe_allow_html=True)
    st.markdown("<div style='padding: 20px; background-color: #F1F8E9; border-radius: 10px;'>", unsafe_allow_html=True)
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Login", key="login_button", help="Click to log in"):
            st.session_state["login_attempted"] = True
            if username == "Bruce" and password == "1234":
                st.session_state["authenticated"] = True
            else:
                st.session_state["authenticated"] = False
    
    if st.session_state["login_attempted"] and not st.session_state["authenticated"]:
        st.error("Incorrect username or password.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.stop()

# ---- SHOW LOGIN IF NOT AUTHENTICATED ----
if not st.session_state["authenticated"]:
    login()

# ---- CROP REQUIREMENTS ----
CROP_REQUIREMENTS = {
    "Wheat": {"min_temp": 10, "max_temp": 25, "min_rainfall": 50, "max_rainfall": 100},
    "Maize": {"min_temp": 15, "max_temp": 30, "min_rainfall": 60, "max_rainfall": 150},
    "Rice": {"min_temp": 20, "max_temp": 35, "min_rainfall": 100, "max_rainfall": 200}
}

# ---- SIMULATED WEATHER DATA (REPLACING API) ----
def get_simulated_weather(city):
    today = datetime.now()
    weather_data = []
    for i in range(7):
        date = (today + timedelta(days=i)).date()
        temp = np.random.uniform(10, 35)  # Random temp between 10-35°C
        rainfall = np.random.uniform(0, 200)  # Random rainfall between 0-200mm
        weather_data.append({"date": date, "temp": temp, "rainfall": rainfall})
    return pd.DataFrame(weather_data)

def is_suitable_for_planting(temp, rainfall, crop):
    req = CROP_REQUIREMENTS[crop]
    return (req["min_temp"] <= temp <= req["max_temp"] and
            req["min_rainfall"] <= rainfall <= req["max_rainfall"])

# ---- LOAD DATA ----
try:
    df = pd.read_excel("crop_data1.xlsx")
except FileNotFoundError:
    st.error("crop_data1.xlsx not found. Please ensure the file is in the same directory as the app.")
    st.stop()

# ---- MAIN DASHBOARD ----
st.markdown("<h1 style='color: #2E7D32;'>🌾 Crop Yield Optimization Dashboard</h1>", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("<h2 style='color: #388E3C;'>🔧 Input Parameters</h2>", unsafe_allow_html=True)
    rain = st.slider("Rainfall (mm)", 300, 1300, 800, key="rain_slider")
    fert = st.slider("Fertilizer (kg/ha)", 40, 100, 70, key="fert_slider")
    temp = st.slider("Temperature (°C)", 20, 40, 30, key="temp_slider")
    n = st.slider("Nitrogen (N, kg/ha)", 60, 90, 75, key="n_slider")
    p = st.slider("Phosphorus (P, kg/ha)", 15, 30, 20, key="p_slider")
    k = st.slider("Potassium (K, kg/ha)", 15, 30, 20, key="k_slider")
    
    st.markdown("<h2 style='color: #388E3C;'>🌱 Planting Scheduler</h2>", unsafe_allow_html=True)
    city = st.text_input("City (e.g., Nairobi, Iowa)", "Nairobi", key="city_input")
    crop = st.selectbox("Select Crop", list(CROP_REQUIREMENTS.keys()), key="crop_select")

# ---- TABS FOR ORGANIZED LAYOUT ----
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Yield Predictor", "📌 Recommendations", "🗓️ Planting Scheduler"])

# ---- TAB 1: DATA OVERVIEW ----
with tab1:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("### Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Yield Distribution")
        fig = px.histogram(df, x="Yield (Q/acre)", nbins=20, title="Yield Distribution", color_discrete_sequence=["#4CAF50"])
        st.plotly_chart(fig, use_container_width=True)
    
    # with col2:
    #     st.markdown("#### Correlation Heatmap")
    #     corr = df[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Yield (Q/acre)"]].corr()
    #     fig, ax = plt.subplots()
    #     sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    #     st.pyplot(fig)

# ---- TAB 2: YIELD PREDICTOR ----
with tab2:
    st.markdown("### Yield Prediction")
    X = df[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]]
    y = df["Yield (Q/acre)"]
    model = LinearRegression()
    model.fit(X, y)
    
    input_data = np.array([[rain, fert, temp, n, p, k]])
    predicted_yield = model.predict(input_data)[0]
    
    st.markdown(f"<h3 style='color: #388E3C;'>Estimated Yield: {predicted_yield:.2f} Q/acre</h3>", unsafe_allow_html=True)
    
    st.markdown("#### Input Parameters Impact")
    coef_df = pd.DataFrame({
        "Parameter": ["Rainfall", "Fertilizer", "Temperature", "Nitrogen", "Phosphorus", "Potassium"],
        "Coefficient": model.coef_
    })
    fig = px.bar(coef_df, x="Parameter", y="Coefficient", title="Parameter Impact on Yield", color_discrete_sequence=["#81C784"])
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: RECOMMENDATIONS ----
with tab3:
    st.markdown("### Yield Recommendations")
    if predicted_yield < 9:
        st.markdown(
            "<div style='background-color: #FFECB3; padding: 15px; border-radius: 10px;'>"
            "<strong>⚠️ Warning:</strong> Yield is below average. Consider:<br>"
            "- Increasing Nitrogen by 5-10 kg/ha<br>"
            "- Checking rainfall patterns for consistency<br>"
            "- Adjusting irrigation if rainfall is insufficient"
            "</div>", unsafe_allow_html=True)
    elif predicted_yield > 11:
        st.markdown(
            "<div style='background-color: #E8F5E9; padding: 15px; border-radius: 10px;'>"
            "<strong>✅ Great:</strong> Conditions are favorable for high yield. Maintain current inputs."
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background-color: #E3F2FD; padding: 15px; border-radius: 10px;'>"
            "<strong>🟡 Moderate:</strong> Yield is average. Fine-tune:<br>"
            "- Fertilizer levels (N, P, K)<br>"
            "- Irrigation scheduling"
            "</div>", unsafe_allow_html=True)
    
    st.markdown("#### Download Recommendations")
    rec_text = f"Predicted Yield: {predicted_yield:.2f} Q/acre\n"
    if predicted_yield < 9:
        rec_text += "Recommendations:\n- Increase Nitrogen by 5-10 kg/ha\n- Check rainfall patterns\n- Adjust irrigation"
    elif predicted_yield > 11:
        rec_text += "Recommendations:\n- Maintain current input levels\n- Monitor for pest/disease risks"
    else:
        rec_text += "Recommendations:\n- Fine-tune fertilizer levels\n- Optimize irrigation scheduling"
    
    st.download_button(
        label="Download Recommendations",
        data=rec_text,
        file_name="crop_yield_recommendations.txt",
        mime="text/plain"
    )

# ---- TAB 4: PLANTING SCHEDULER ----
with tab4:
    st.markdown("### Planting Scheduler")
    if st.button("Generate Planting Schedule", key="schedule_button"):
        weather_df = get_simulated_weather(city)
        suitable_dates = []
        
        for _, row in weather_df.iterrows():
            if is_suitable_for_planting(row["temp"], row["rainfall"], crop):
                suitable_dates.append(row["date"])
        
        if suitable_dates:
            st.markdown(f"<h3 style='color: #388E3C;'>Recommended Planting Dates for {crop} in {city}</h3>", unsafe_allow_html=True)
            date_range = f"{min(suitable_dates).strftime('%B %d')} - {max(suitable_dates).strftime('%B %d')}"
            st.markdown(f"**Best planting period:** {date_range}")
            
            df_dates = pd.DataFrame({
                "Date": suitable_dates,
                "Suitability": ["Suitable" for _ in suitable_dates]
            })
            fig = px.scatter(df_dates, x="Date", y="Suitability", title=f"Suitable Planting Dates for {crop}",
                            labels={"Suitability": ""}, height=200)
            fig.update_yaxes(showticklabels=False)
            fig.update_traces(marker=dict(size=12, color="#4CAF50"))
            st.plotly_chart(fig, use_container_width=True)
            
            # Download planting schedule
            schedule_text = f"Planting Schedule for {crop} in {city}\nBest Period: {date_range}\nDates: {', '.join([d.strftime('%Y-%m-%d') for d in suitable_dates])}"
            st.download_button(
                label="Download Planting Schedule",
                data=schedule_text,
                file_name="planting_schedule.txt",
                mime="text/plain"
            )
        else:
            st.warning(f"No suitable planting dates found for {crop} in the next 7 days.")

# ---- LOGOUT BUTTON ----
with st.sidebar:
    if st.button("Logout", key="logout_button"):
        st.session_state["authenticated"] = False
        st.session_state["login_attempted"] = False
        st.rerun()
