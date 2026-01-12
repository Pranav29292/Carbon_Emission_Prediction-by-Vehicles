import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# APP CONFIG
st.set_page_config(
    page_title="Vehicle CO₂ Emission Prediction System",
    layout="wide"
)
# LOAD TRAINED ML MODEL
model = joblib.load("vehicle_co2_model.pkl")
# FUEL ADJUSTMENT FACTORS
FUEL_ADJUSTMENT = {
    "Petrol": 1.00,
    "Diesel": 1.06,
    "CNG": 0.90
}
# SESSION STATE
if "page" not in st.session_state:
    st.session_state.page = "input"
# HELPER FUNCTIONS
def co2_status(co2):
    if co2 <= 120:
        return "Safe (Low Emission)"
    elif co2 <= 180:
        return "Moderate Emission"
    else:
        return "Unsafe (High Emission)"

def reduction_tips():
    tips = [
        "Maintain proper tyre pressure",
        "Drive at steady speeds",
        "Avoid sudden acceleration and braking",
        "Regular vehicle servicing",
        "Reduce unnecessary vehicle load",
        "Use carpooling or public transport",
        "Consider switching to Hybrid or Electric vehicles"
    ]
    for tip in tips:
        st.write("•", tip)
# PAGE 1 : INPUT
if st.session_state.page == "input":

    st.title("🚗 Vehicle CO₂ Emission Prediction System")
    st.subheader("Step 1: Enter Vehicle Details")

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["Electric (EV)", "Petrol", "Diesel", "CNG"]
    )

    distance = st.number_input(
        "Distance Travelled (km)",
        min_value=1.0,
        value=20.0
    )

    st.session_state.vehicle_type = vehicle_type
    st.session_state.distance = distance

    if vehicle_type == "Electric (EV)":
        energy_consumption = st.number_input(
            "Energy Consumption (kWh / 100 km)",
            min_value=0.1,
            value=6.0
        )
        grid_emission = st.number_input(
            "Grid Emission Factor (g CO₂ / kWh)",
            min_value=0.1,
            value=700.0
        )

        st.session_state.energy_consumption = energy_consumption
        st.session_state.grid_emission = grid_emission

    else:
        engine_size = st.number_input(
            "Engine Size (Litre)",
            min_value=0.1,
            value=2.0
        )
        fuel_consumption = st.number_input(
            "Fuel Consumption (L / 100 km)",
            min_value=0.1,
            value=8.0
        )

        st.session_state.engine_size = engine_size
        st.session_state.fuel_consumption = fuel_consumption

    if st.button("🔮 Predict Emissions"):
        st.session_state.page = "output"
        st.rerun()

# PAGE 2 : OUTPUT
elif st.session_state.page == "output":

    st.title("📊 Emission Analysis Results")

    vehicle_type = st.session_state.vehicle_type
    distance = st.session_state.distance
    distances = np.arange(1, int(distance) + 1)

    # =================================================
    # ELECTRIC VEHICLE
    # =================================================
    if vehicle_type == "Electric (EV)":

        final_co2 = (
            st.session_state.energy_consumption *
            st.session_state.grid_emission
        ) / 100

        total_co2 = (final_co2 * distance) / 1000

        st.success(f"Electric Vehicle CO₂: {final_co2:.2f} g/km")
        st.info(f"Total CO₂ for {distance:.2f} km: {total_co2:.2f} kg")
        st.info(f"Emission Status: {co2_status(final_co2)}")

        ice_val = 160
        hybrid_val = final_co2 * 0.6
        ev_val = final_co2

    # ICE VEHICLES (ML PREDICTED)
    else:
        ml_co2 = model.predict(
            np.array([[st.session_state.engine_size,
                       st.session_state.fuel_consumption]])
        )[0]

        final_co2 = ml_co2 * FUEL_ADJUSTMENT[vehicle_type]
        total_co2 = (final_co2 * distance) / 1000

        st.success(
            f"{vehicle_type} Vehicle CO₂ (ML Predicted): {final_co2:.2f} g/km"
        )
        st.info(f"Total CO₂ for {distance:.2f} km: {total_co2:.2f} kg")
        st.info(f"Emission Status: {co2_status(final_co2)}")

        ice_val = final_co2
        hybrid_val = final_co2 * 0.6
        ev_val = 0
        
    # COMPARISON TABLE
    comparison_df = pd.DataFrame({
        "Vehicle Type": ["ICE Vehicle", "Hybrid Vehicle", "Electric Vehicle"],
        "CO₂ Emissions (g/km)": [ice_val, hybrid_val, ev_val]
    })

    st.subheader("Vehicle Emission Comparison")
    st.dataframe(comparison_df)
    
    # BAR GRAPH
    plt.figure(figsize=(8, 4))
    plt.bar(
        comparison_df["Vehicle Type"],
        comparison_df["CO₂ Emissions (g/km)"]
    )
    plt.xlabel("Vehicle Type")
    plt.ylabel("CO₂ Emissions (g/km)")
    plt.title("CO₂ Emission Comparison")
    st.pyplot(plt)
    plt.close()

    # LINE GRAPH
    plt.figure(figsize=(9, 4))
    plt.plot(distances, ice_val * distances, label="ICE Vehicle")
    plt.plot(distances, hybrid_val * distances, label="Hybrid Vehicle")
    plt.plot(distances, ev_val * distances, label="Electric Vehicle")
    plt.xlabel("Distance (km)")
    plt.ylabel("Total CO₂ Emissions (grams)")
    plt.title("CO₂ Emission Trend vs Distance")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

    # RECOMMENDATION
    st.subheader("Recommended Vehicle Choice")

    sorted_df = comparison_df.sort_values("CO₂ Emissions (g/km)")
    best = sorted_df.iloc[0]
    second = sorted_df.iloc[1]

    st.success(
        f"✅ Best Choice: {best['Vehicle Type']} "
        f"({best['CO₂ Emissions (g/km)']:.2f} g/km)"
    )
    st.info(
        f"ℹ️ Practical Alternative: {second['Vehicle Type']} "
        f"({second['CO₂ Emissions (g/km)']:.2f} g/km)"
    )

    # TIPS
    st.subheader("How to Reduce CO₂ Emissions")
    reduction_tips()

    if st.button("⬅ Back to Input Page"):
        st.session_state.page = "input"
        st.rerun()
