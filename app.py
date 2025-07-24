import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import numpy as np
from datetime import datetime, timedelta
import requests

# ---- JAVASCRIPT ERROR HANDLER FOR DEPLOYMENT ISSUES ----
components.html(
    """
    <script>
    window.addEventListener('error', (event) => {
        if (event.message.includes('Failed to fetch dynamically imported module')) {
            window.location.reload();
        }
    });
    </script>
    """,
    height=0
)

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
            if username == "Bruce" and password == "1234":  # TODO: Replace with secure authentication
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

# ---- CROP REQUIREMENTS (Extended with Millet) ----
CROP_REQUIREMENTS = {
    "Wheat": {"min_temp": 10, "max_temp": 25, "min_rainfall": 50, "max_rainfall": 100},
    "Maize": {"min_temp": 15, "max_temp": 30, "min_rainfall": 60, "max_rainfall": 150},
    "Rice": {"min_temp": 20, "max_temp": 35, "min_rainfall": 100, "max_rainfall": 200},
    "Soybean": {"min_temp": 18, "max_temp": 30, "min_rainfall": 60, "max_rainfall": 120},
    "Sorghum": {"min_temp": 20, "max_temp": 35, "min_rainfall": 40, "max_rainfall": 100},
    "Sunflower": {"min_temp": 18, "max_temp": 30, "min_rainfall": 50, "max_rainfall": 100},
    "Barley": {"min_temp": 12, "max_temp": 25, "min_rainfall": 60, "max_rainfall": 100},
    "Groundnut": {"min_temp": 20, "max_temp": 30, "min_rainfall": 50, "max_rainfall": 100},
    "Millet": {"min_temp": 15, "max_temp": 35, "min_rainfall": 30, "max_rainfall": 150}
}

# ---- CITY COORDINATES (Top 10 African Cities with Best Weather) ----
CITY_COORDINATES = {
    "Nairobi, Kenya": (-1.2921, 36.8219),
    "Kigali, Rwanda": (-1.9441, 30.0619),
    "Addis Ababa, Ethiopia": (9.0249, 38.7468),
    "Kampala, Uganda": (0.3476, 32.5825),
    "Cape Town, South Africa": (-33.9249, 18.4241),
    "Pretoria, South Africa": (-25.7479, 28.2293),
    "Gaborone, Botswana": (-24.6282, 25.9231),
    "Windhoek, Namibia": (-22.5597, 17.0832),
    "Antananarivo, Madagascar": (-18.8792, 47.5079),
    "Port Louis, Mauritius": (-20.1644, 57.5012)
}

# ---- WEATHER DATA FROM OPEN-METEO ----
@st.cache_data
def get_weather_data(city):
    # Temporarily force simulated data for testing (7 days)
    return get_simulated_weather(city)

def get_simulated_weather(city):
    today = datetime.now()
    weather_data = []
    for i in range(7):  # Set to 7 days
        date = (today + timedelta(days=i)).date()
        temp = np.random.uniform(15, 35)  # Favor Millet's requirements
        rainfall = np.random.uniform(30, 150)
        weather_data.append({"date": date, "temp": temp, "rainfall": rainfall})
    return pd.DataFrame(weather_data)

def is_suitable_for_planting(temp, rainfall, crop):
    req = CROP_REQUIREMENTS[crop]
    # Allow zero rainfall as a valid case for dry-season crops
    return (req["min_temp"] <= temp <= req["max_temp"] and
            (rainfall >= req["min_rainfall"] or rainfall == 0) and rainfall <= req["max_rainfall"])

# ---- INPUT VALIDATION ----
def validate_inputs(rain, fert, temp, n, p, k, crop):
    req = CROP_REQUIREMENTS[crop]
    warnings = []
    if not (300 <= rain <= 1500):
        warnings.append("Rainfall should be between 300-1500 mm.")
    if not (req["min_temp"] <= temp <= req["max_temp"]):
        warnings.append(f"Temperature should be between {req['min_temp']} and {req['max_temp']}°C for {crop}.")
    if not (30 <= fert <= 120):
        warnings.append("Fertilizer should be between 30-120 kg/ha.")
    if not (50 <= n <= 100):
        warnings.append("Nitrogen should be between 50-100 kg/ha.")
    if not (10 <= p <= 40):
        warnings.append("Phosphorus should be between 10-40 kg/ha.")
    if not (10 <= k <= 40):
        warnings.append("Potassium should be between 10-40 kg/ha.")
    return warnings

# ---- LOAD DATA ----
try:
    df = pd.read_excel("crop_data1.xlsx")
    required_columns = ["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Yield (Q/acre)"]
    if not all(col in df.columns for col in required_columns):
        st.error("Excel file missing required columns.")
        st.stop()
    df = df.fillna(df.median(numeric_only=True))
except FileNotFoundError:
    st.error("crop_data1.xlsx not found. Please ensure the file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# ---- HOMEPAGE ----
with st.container():
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌾 Welcome to AgriLens</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; background-color: #F1F8E9; padding: 20px; border-radius: 10px;'>
        <p style='font-size: 18px;'>AgriLens empowers farmers with data-driven insights to optimize crop yields and plan planting schedules.</p>
        <p style='font-size: 16px; color: #388E3C;'>Explore the tabs below to analyze data, predict yields, get recommendations, and schedule planting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    #st.image("https://images.unsplash.com/photo-1500595046743-ff22c10ab070", caption="Visualize Your Farming Future", use_container_width=True)


# ##################
# with st.sidebar:
#     st.markdown("## 📞 Judas Critical 0711112222")
# ##############

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("<h2 style='color: #388E3C;'>🔧 Input Parameters</h2>", unsafe_allow_html=True)
    rain = st.slider("Rainfall (mm)", 300, 1300, 800, key="rain_slider", help="Annual rainfall in millimeters.")
    fert = st.slider("Fertilizer (kg/ha)", 40, 100, 70, key="fert_slider", help="Fertilizer applied per hectare.")
    temp = st.slider("Temperature (°C)", 20, 40, 30, key="temp_slider", help="Average temperature during growing season.")
    n = st.slider("Nitrogen (N, kg/ha)", 60, 90, 75, key="n_slider", help="Nitrogen applied per hectare.")
    p = st.slider("Phosphorus (P, kg/ha)", 15, 30, 20, key="p_slider", help="Phosphorus applied per hectare.")
    k = st.slider("Potassium (K, kg/ha)", 15, 30, 20, key="k_slider", help="Potassium applied per hectare.")
    
    st.markdown("<h2 style='color: #388E3C;'>🌱 Planting Scheduler</h2>", unsafe_allow_html=True)
    city = st.selectbox("Select City", list(CITY_COORDINATES.keys()), index=0, key="city_select", help="Choose a city for weather forecast.")
    crop = st.selectbox("Select Crop", list(CROP_REQUIREMENTS.keys()), index=list(CROP_REQUIREMENTS.keys()).index("Millet"), key="crop_select", help="Choose crop for planting schedule.")

# ---- TABS FOR ORGANIZED LAYOUT ----
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Yield Predictor", "📌 Recommendations", "🗓️ Planting Scheduler"])

# ---- TAB 1: DATA OVERVIEW ----
with tab1:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("### Data Visualizations")
    st.markdown("#### Yield Distribution")
    @st.cache_data
    def get_histogram_data():
        return df["Yield (Q/acre)"]
    fig = px.histogram(get_histogram_data(), x="Yield (Q/acre)", nbins=20, title="Yield Distribution", color_discrete_sequence=["#4CAF50"])
    fig.update_layout(showlegend=False, dragmode=False)
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: YIELD PREDICTOR ----
with tab2:
    st.markdown("### Yield Prediction")
    warnings = validate_inputs(rain, fert, temp, n, p, k, crop)
    for warning in warnings:
        st.warning(warning)
    
    X = df[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]]
    y = df["Yield (Q/acre)"]
    
    @st.cache_resource
    def train_model(X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    model = train_model(X, y)
    input_data = np.array([[rain, fert, temp, n, p, k]])
    predicted_yield = model.predict(input_data)[0]
    
    st.markdown(f"<h3 style='color: #388E3C;'>Estimated Yield: {predicted_yield:.2f} Q/acre</h3>", unsafe_allow_html=True)

    st.markdown("#### Feature Importance")
    @st.cache_data
    def compute_importance(_model, X, y):
        return permutation_importance(_model, X, y, n_repeats=10, random_state=42)
    
    perm_importance = compute_importance(model, X, y)
    importance_df = pd.DataFrame({
        "Parameter": ["Rainfall", "Fertilizer", "Temperature", "Nitrogen", "Phosphorus", "Potassium"],
        "Importance": perm_importance.importances_mean
    })
    fig = px.bar(importance_df, x="Parameter", y="Importance", title="Feature Importance for Yield Prediction",
                 color_discrete_sequence=["#81C784"], text="Importance",
                 hover_data={"Parameter": True, "Importance": ":.3f"})
    fig.update_traces(texttemplate="%{text:.3f}", textposition="auto")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: RECOMMENDATIONS ----
with tab3:
    st.markdown("### Yield Recommendations")
    top_feature = importance_df.loc[importance_df["Importance"].idxmax(), "Parameter"]
    if predicted_yield < 9:
        st.markdown(
            "<div style='background-color: #FFECB3; padding: 15px; border-radius: 10px;'>"
            "<strong>⚠️ Warning:</strong> Yield is below average. Consider:<br>"
            f"- Increasing {top_feature} (highest impact)<br>"
            "- Increasing Nitrogen by 5-10 kg/ha<br>"
            "- Checking rainfall patterns for consistency<br>"
            "- Adjusting irrigation if rainfall is insufficient"
            "</div>", unsafe_allow_html=True)
    elif predicted_yield > 11:
        st.markdown(
            "<div style='background-color: #E8F5E9; padding: 15px; border-radius: 10px;'>"
            "<strong>✅ Great:</strong> Conditions are favorable for high yield. Maintain current inputs.<br>"
            f"- Monitor {top_feature} closely for consistency."
            "</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background-color: #E3F2FD; padding: 15px; border-radius: 10px;'>"
            "<strong>🟡 Moderate:</strong> Yield is average. Fine-tune:<br>"
            f"- Optimize {top_feature} (highest impact)<br>"
            "- Fertilizer levels (N, P, K)<br>"
            "- Irrigation scheduling"
            "</div>", unsafe_allow_html=True)
    
    st.markdown("#### Download Recommendations")
    rec_text = f"Predicted Yield: {predicted_yield:.2f} Q/acre\n"
    if predicted_yield < 9:
        rec_text += f"Recommendations:\n- Increase {top_feature} (highest impact)\n- Increase Nitrogen by 5-10 kg/ha\n- Check rainfall patterns\n- Adjust irrigation"
    elif predicted_yield > 11:
        rec_text += f"Recommendations:\n- Maintain current input levels\n- Monitor {top_feature} closely\n- Check for pest/disease risks"
    else:
        rec_text += f"Recommendations:\n- Optimize {top_feature} (highest impact)\n- Fine-tune fertilizer levels\n- Optimize irrigation scheduling"
    
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
        weather_df = get_weather_data(city)
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

