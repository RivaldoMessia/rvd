import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from io import StringIO

# ---- SESSION STATE INIT ----
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "login_attempted" not in st.session_state:
    st.session_state["login_attempted"] = False
if "date_range" not in st.session_state:
    st.session_state["date_range"] = "N/A"
if "suitable_dates" not in st.session_state:
    st.session_state["suitable_dates"] = []
if "weather_df" not in st.session_state:
    st.session_state["weather_df"] = pd.DataFrame()

# ---- CUSTOM CSS FOR STYLING ----
st.markdown("""
<style>
    .main { background-color: #F5F5F5; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #E8F5E9;
    }
    h1, h2, h3 { font-family: 'Arial', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ---- LOGIN SCREEN ----
def login():
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🔐 Farmer's Crop Dashboard</h1>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div style='padding: 20px; background-color: #F1F8E9; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Login", key="login_button"):
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
    "Corn": {"min_temp": 15, "max_temp": 30, "min_rainfall": 60, "max_rainfall": 150},
    "Rice": {"min_temp": 20, "max_temp": 35, "min_rainfall": 100, "max_rainfall": 200},
    "Barley": {"min_temp": 10, "max_temp": 25, "min_rainfall": 50, "max_rainfall": 100},
    "Sorghum": {"min_temp": 20, "max_temp": 35, "min_rainfall": 60, "max_rainfall": 150},
    "Soybean": {"min_temp": 15, "max_temp": 30, "min_rainfall": 60, "max_rainfall": 150}
}

# ---- SIMULATED WEATHER DATA ----
@st.cache_data
def get_simulated_weather(city):
    today = datetime.now()
    weather_data = []
    for i in range(7):
        date = (today + timedelta(days=i)).date()
        temp = np.random.uniform(10, 40)
        rainfall = np.random.uniform(0, 200)
        weather_data.append({"date": date, "temp": temp, "rainfall": rainfall})
    return pd.DataFrame(weather_data)

def is_suitable_for_planting(temp, rainfall, crop):
    req = CROP_REQUIREMENTS[crop]
    return (req["min_temp"] <= temp <= req["max_temp"] and
            req["min_rainfall"] <= rainfall <= req["max_rainfall"])

# ---- EMBEDDED CROP DATA ----
@st.cache_data
def load_data():
    csv_data = """Crop Type,Rain Fall (mm),Fertilizer,Temperatue,Nitrogen (N),Phosphorus (P),Potassium (K),Yield (Q/acre)
Wheat,500,60,20,70,20,15,5.5
Corn,800,80,25,80,25,20,7.2
Rice,1200,90,30,85,30,25,6.8
Barley,550,65,18,72,18,14,5.0
Sorghum,700,75,28,78,22,18,6.5
Soybean,750,70,26,76,24,20,6.0
Wheat,600,62,22,71,21,16,5.7
Corn,850,82,27,81,26,21,7.5
Rice,1100,88,29,84,29,24,6.7
Barley,520,60,19,70,17,13,4.8"""
    df = pd.read_csv(StringIO(csv_data))
    df = df.dropna()
    for col in ["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Yield (Q/acre)"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# ---- MAIN DASHBOARD ----
st.markdown("<h1 style='color: #2E7D32;'>🌾 Crop Yield Optimization Dashboard</h1>", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("<h2 style='color: #388E3C;'>🔧 Input Parameters</h2>", unsafe_allow_html=True)
    rain = st.slider("Rainfall (mm)", 300, 1400, 800, key="rain_slider")
    fert = st.slider("Fertilizer (kg/ha)", 40, 100, 70, key="fert_slider")
    temp = st.slider("Temperature (°C)", 15, 40, 30, key="temp_slider")
    n = st.slider("Nitrogen (N, kg/ha)", 60, 90, 75, key="n_slider")
    p = st.slider("Phosphorus (P, kg/ha)", 15, 30, 20, key="p_slider")
    k = st.slider("Potassium (K, kg/ha)", 10, 30, 20, key="k_slider")
    
    st.markdown("<h2 style='color: #388E3C;'>🌱 Planting Scheduler</h2>", unsafe_allow_html=True)
    city = st.text_input("City (e.g., Nairobi, Iowa)", "Nairobi", key="city_input")
    crop = st.selectbox("Select Crop", list(CROP_REQUIREMENTS.keys()), key="crop_select")

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🌱 Crop Analysis", "🤖 Yield Predictor", "📌 Recommendations", "🗓️ Planting Scheduler"])

# ---- TAB 1: DATA OVERVIEW ----
with tab1:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("### Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Yield Distribution by Crop")
        fig = px.histogram(df, x="Yield (Q/acre)", color="Crop Type", nbins=20, title="Yield Distribution by Crop",
                          color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Correlation Heatmap")
        corr = df[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Yield (Q/acre)"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="YlGnBu", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: CROP ANALYSIS ----
with tab2:
    st.markdown("### Crop-Specific Analysis")
    selected_crop = st.selectbox("Select Crop for Analysis", df["Crop Type"].unique(), key="crop_analysis_select")
    crop_df = df[df["Crop Type"] == selected_crop]
    
    st.markdown(f"#### {selected_crop} Yield Statistics")
    stats = crop_df["Yield (Q/acre)"].describe()
    stats_df = pd.DataFrame({
        "Statistic": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
        "Value": [stats["count"], stats["mean"], stats["std"], stats["min"], stats["25%"], stats["50%"], stats["75%"], stats["max"]]
    })
    st.dataframe(stats_df, use_container_width=True)
    
    st.markdown(f"#### {selected_crop} Yield vs. Factors")
    factor = st.selectbox("Select Factor", ["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
                         key="factor_select")
    fig = px.scatter(crop_df, x=factor, y="Yield (Q/acre)", title=f"{selected_crop} Yield vs {factor}",
                    color_discrete_sequence=["#4CAF50"])
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: YIELD PREDICTOR ----
with tab3:
    st.markdown("### Yield Prediction")
    X = df[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]]
    y = df["Yield (Q/acre)"]
    model = LinearRegression()
    model.fit(X, y)
    
    input_data = np.array([[rain, fert, temp, n, p, k]])
    try:
        predicted_yield = model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        predicted_yield = 0.0
    
    st.markdown(f"<h3 style='color: #388E3C;'>Estimated Yield: {predicted_yield:.2f} Q/acre</h3>", unsafe_allow_html=True)
    
    st.markdown("#### Compare Yields Across Crops")
    crop_yields = {}
    for crop_type in CROP_REQUIREMENTS.keys():
        crop_df = df[df["Crop Type"] == crop_type]
        if not crop_df.empty:
            X_crop = crop_df[["Rain Fall (mm)", "Fertilizer", "Temperatue", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]]
            y_crop = crop_df["Yield (Q/acre)"]
            model_crop = LinearRegression()
            model_crop.fit(X_crop, y_crop)
            try:
                crop_yields[crop_type] = model_crop.predict(input_data)[0]
            except Exception as e:
                crop_yields[crop_type] = 0.0
    
    yields_df = pd.DataFrame({
        "Crop": list(crop_yields.keys()),
        "Predicted Yield": list(crop_yields.values())
    })
    fig = px.bar(yields_df, x="Crop", y="Predicted Yield", title="Predicted Yield by Crop",
                 color_discrete_sequence=["#81C784"])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Parameter Impact on Yield")
    coef_df = pd.DataFrame({
        "Parameter": ["Rainfall", "Fertilizer", "Temperature", "Nitrogen", "Phosphorus", "Potassium"],
        "Coefficient": model.coef_
    })
    fig = px.bar(coef_df, x="Parameter", y="Coefficient", title="Parameter Impact on Yield",
                 color_discrete_sequence=["#81C784"])
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 4: RECOMMENDATIONS ----
with tab4:
    st.markdown("### Yield Recommendations")
    avg_yield = df["Yield (Q/acre)"].mean()
    crop_avg_yield = df.groupby("Crop Type")["Yield (Q/acre)"].mean().to_dict()
    crop_specific_avg = crop_avg_yield.get(crop, avg_yield)
    
    if predicted_yield < crop_specific_avg - 0.5:
        recommendations = [
            f"Increase Nitrogen by 5-10 kg/ha (current: {n} kg/ha)",
            "Ensure consistent rainfall or supplement with irrigation",
            f"Optimize {crop} planting conditions (ideal temp: {CROP_REQUIREMENTS[crop]['min_temp']}-{CROP_REQUIREMENTS[crop]['max_temp']}°C)"
        ]
        st.markdown(
            "<div style='background-color: #FFECB3; padding: 15px; border-radius: 10px;'>"
            f"<strong>⚠️ Warning:</strong> Predicted yield ({predicted_yield:.2f} Q/acre) is below {crop} average ({crop_specific_avg:.2f} Q/acre). Consider:<br>"
            f"- {recommendations[0]}<br>"
            f"- {recommendations[1]}<br>"
            f"- {recommendations[2]}"
            "</div>", unsafe_allow_html=True)
    elif predicted_yield > crop_specific_avg + 0.5:
        recommendations = [
            "Maintain current input levels",
            "Monitor for pest and disease risks",
            f"Continue optimal {crop} management practices"
        ]
        st.markdown(
            "<div style='background-color: #E8F5E9; padding: 15px; border-radius: 10px;'>"
            f"<strong>✅ Great:</strong> Predicted yield ({predicted_yield:.2f} Q/acre) is above {crop} average ({crop_specific_avg:.2f} Q/acre). Actions:<br>"
            f"- {recommendations[0]}<br>"
            f"- {recommendations[1]}<br>"
            f"- {recommendations[2]}"
            "</div>", unsafe_allow_html=True)
    else:
        recommendations = [
            f"Fine-tune fertilizer levels (N: {n}, P: {p}, K: {k} kg/ha)",
            "Optimize irrigation scheduling",
            f"Check {crop} soil conditions"
        ]
        st.markdown(
            "<div style='background-color: #E3F2FD; padding: 15px; border-radius: 10px;'>"
            f"<strong>🟡 Moderate:</strong> Predicted yield ({predicted_yield:.2f} Q/acre) is near {crop} average ({crop_specific_avg:.2f} Q/acre). Fine-tune:<br>"
            f"- {recommendations[0]}<br>"
            f"- {recommendations[1]}<br>"
            f"- {recommendations[2]}"
            "</div>", unsafe_allow_html=True)
    
    st.markdown("#### Download Recommendations")
    rec_text = f"Predicted Yield for {crop}: {predicted_yield:.2f} Q/acre\nAverage {crop} Yield: {crop_specific_avg:.2f} Q/acre\nRecommendations:\n" + "\n".join([f"- {rec}" for rec in recommendations])
    st.download_button(
        label="Download Recommendations",
        data=rec_text,
        file_name="crop_yield_recommendations.txt",
        mime="text/plain"
    )

# ---- TAB 5: PLANTING SCHEDULER ----
with tab5:
    st.markdown("### Planting Scheduler")
    if st.button("Generate Planting Schedule", key="schedule_button"):
        try:
            st.session_state["weather_df"] = get_simulated_weather(city)
            if st.session_state["weather_df"].empty:
                raise ValueError("Weather data is empty")
            suitable_dates = []
            for _, row in st.session_state["weather_df"].iterrows():
                if is_suitable_for_planting(row["temp"], row["rainfall"], crop):
                    suitable_dates.append(row["date"])
            st.session_state["suitable_dates"] = suitable_dates
        except Exception as e:
            st.error(f"Error generating weather data: {str(e)}")
            st.session_state["weather_df"] = pd.DataFrame()
            st.session_state["suitable_dates"] = []
        
        st.markdown(f"#### Weather Forecast for {city}")
        if not st.session_state["weather_df"].empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state["weather_df"]["date"],
                y=st.session_state["weather_df"]["temp"],
                mode="lines+markers",
                name="Temperature (°C)",
                line=dict(color="#FF5722")
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state["weather_df"]["date"],
                y=st.session_state["weather_df"]["rainfall"],
                mode="lines+markers",
                name="Rainfall (mm)",
                line=dict(color="#2196F3"),
                yaxis="y2"
            ))
            fig.update_layout(
                title=f"Weather Forecast for {city}",
                xaxis=dict(title="Date"),
                yaxis=dict(title="Temperature (°C)", titlefont=dict(color="#FF5722"), tickfont=dict(color="#FF5722")),
                yaxis2=dict(
                    title="Rainfall (mm)",
                    titlefont=dict(color="#2196F3"),
                    tickfont=dict(color="#2196F3"),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                template="plotly_white",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No weather data available to display.")
        
        if st.session_state["suitable_dates"]:
            st.session_state["date_range"] = f"{min(st.session_state['suitable_dates']).strftime('%B %d')} - {max(st.session_state['suitable_dates']).strftime('%B %d')}"
            st.markdown(f"<h3 style='color: #388E3C;'>Recommended Planting Dates for {crop} in {city}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Best planting period:** {st.session_state['date_range']}")
            
            df_dates = pd.DataFrame({
                "Date": st.session_state["suitable_dates"],
                "Suitability": ["Suitable" for _ in st.session_state["suitable_dates"]]
            })
            fig = px.scatter(df_dates, x="Date", y="Suitability", title=f"Suitable Planting Dates for {crop}",
                            labels={"Suitability": ""}, height=200)
            fig.update_yaxes(showticklabels=False)
            fig.update_traces(marker=dict(size=12, color="#4CAF50"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.session_state["date_range"] = "N/A"
            st.warning(f"No suitable planting dates found for {crop} in the next 7 days.")
        
        # Download planting schedule
        schedule_text = f"Planting Schedule for {crop} in {city}\nBest Period: {st.session_state['date_range']}\nDates: {', '.join([d.strftime('%Y-%m-%d') for d in st.session_state['suitable_dates']]) if st.session_state['suitable_dates'] else 'None'}\n\nWeather Forecast:\n{st.session_state['weather_df'].to_string(index=False) if not st.session_state['weather_df'].empty else 'No data available'}"
        st.download_button(
            label="Download Planting Schedule",
            data=schedule_text,
            file_name="planting_schedule.txt",
            mime="text/plain"
        )

# ---- COMPREHENSIVE REPORT ----
st.markdown("### Generate Comprehensive Report")
if st.button("Download Full Report", key="full_report_button"):
    report_text = f"Crop Yield Optimization Report\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_text += f"Dataset Summary:\n{df.describe().to_string()}\n\n"
    report_text += f"Yield Prediction for {crop}:\n- Predicted Yield: {predicted_yield:.2f} Q/acre\n- Average {crop} Yield: {crop_avg_yield.get(crop, avg_yield):.2f} Q/acre\n- Recommendations:\n" + "\n".join([f"  - {rec}" for rec in recommendations]) + "\n\n"
    report_text += f"Planting Schedule for {crop} in {city}:\n- Best Period: {st.session_state['date_range']}\n- Suitable Dates: {', '.join([d.strftime('%Y-%m-%d') for d in st.session_state['suitable_dates']]) if st.session_state['suitable_dates'] else 'None'}\n\n"
    report_text += f"Weather Forecast for {city}:\n{st.session_state['weather_df'].to_string(index=False) if not st.session_state['weather_df'].empty else 'No data available'}"
    
    st.download_button(
        label="Download Full Report",
        data=report_text,
        file_name="crop_optimization_report.txt",
        mime="text/plain"
    )

# ---- LOGOUT BUTTON ----
with st.sidebar:
    if st.button("Logout", key="logout_button"):
        st.session_state["authenticated"] = False
        st.session_state["login_attempted"] = False
        st.session_state["date_range"] = "N/A"
        st.session_state["suitable_dates"] = []
        st.session_state["weather_df"] = pd.DataFrame()
        st.rerun()