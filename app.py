import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(
    page_title="Vehicle CO₂ Emission Prediction System",
    layout="wide"
)

# ===============================
# LOAD TRAINED ML MODEL
# ===============================
model = joblib.load("vehicle_co2_model.pkl")

# ===============================
# FUEL ADJUSTMENT FACTORS
# ===============================
FUEL_ADJUSTMENT = {
    "Petrol": 1.00,
    "Diesel": 1.06,
    "CNG": 0.90
}

# ===============================
# SESSION STATE
# ===============================
if "page" not in st.session_state:
    st.session_state.page = "input"

# ===============================
# HELPER FUNCTIONS
# ===============================
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

# =================================================
# PAGE 1 : INPUT
# =================================================
if st.session_state.page == "input":

    st.title("🚗 Vehicle CO₂ Emission Prediction System")
    st.subheader("Step 1: Enter Vehicle Details")

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["Petrol", "Diesel", "CNG","Electric (EV)",]
    )

    distance = st.number_input(
        "Distance Travelled (km)",
        min_value=1.0,
        value=20.0
    )

    st.session_state.vehicle_type = vehicle_type
    st.session_state.distance = distance

    # ===============================
    # ELECTRIC VEHICLE INPUT
    # ===============================
    if vehicle_type == "Electric (EV)":

        st.subheader("Electric Vehicle Parameters")

        st.info(
            "Typical EV ranges → "
            "Energy Consumption: 10–25 kWh/100 km | "
            "Grid Emission Factor: 300–900 g CO₂/kWh"
        )

        energy_consumption = st.slider(
            "Energy Consumption (kWh / 100 km)",
            min_value=10.0,
            max_value=25.0,
            value=15.0,
            step=0.5
        )

        grid_emission = st.slider(
            "Grid Emission Factor (g CO₂ / kWh)",
            min_value=300.0,
            max_value=900.0,
            value=700.0,
            step=10.0
        )

        st.session_state.energy_consumption = energy_consumption
        st.session_state.grid_emission = grid_emission

    # ===============================
    # ICE / CNG VEHICLE INPUT
    # ===============================
    else:
        st.subheader("Engine & Fuel Details")

        model_type = st.selectbox(
            "Vehicle Category",
            ["Two-Wheeler", "Hatchback", "Sedan", "SUV"]
        )

        # -------------------------------
        # BIKE MODELS
        # -------------------------------
        if model_type == "Two-Wheeler":

            bike_type = st.selectbox(
                "Two-Wheeler Model Type",
                ["Commuter", "Cruiser", "Sports", "Scooter"]
            )

            bike_engine = {
                "Commuter": (0.10, 0.15),
                "Cruiser": (0.25, 0.50),
                "Sports": (0.20, 0.40),
                "Scooter": (0.10, 0.13)
            }

            bike_fuel = {
                "Commuter": (2.0, 3.0),
                "Cruiser": (3.5, 5.0),
                "Sports": (3.0, 4.5),
                "Scooter": (1.8, 2.5)
            }

            min_eng, max_eng = bike_engine[bike_type]
            min_fc, max_fc = bike_fuel[bike_type]

            st.info(
                f"{bike_type} Bike → "
                f"Engine: {min_eng}-{max_eng} L | "
                f"Fuel: {min_fc}-{max_fc} L/100 km"
            )

            engine_size = st.slider(
                "Engine Size (litres)",
                min_value=min_eng,
                max_value=max_eng,
                value=round((min_eng + max_eng) / 2, 2),
                step=0.01
            )

            fuel_consumption = st.slider(
                "Fuel Consumption (L / 100 km)",
                min_value=min_fc,
                max_value=max_fc,
                value=round((min_fc + max_fc) / 2, 1),
                step=0.1
            )

        # -------------------------------
        # CAR MODELS (FIXED RANGE DISPLAY)
        # -------------------------------
        else:
            engine_ranges = {
                "Hatchback": (0.8, 1.2),
                "Sedan": (1.2, 1.8),
                "SUV": (1.8, 3.0)
            }

            fuel_ranges = {
                "Hatchback": (4.0, 6.0),
                "Sedan": (5.0, 8.0),
                "SUV": (7.0, 12.0)
            }

            min_eng, max_eng = engine_ranges[model_type]
            min_fc, max_fc = fuel_ranges[model_type]

            # ✅ RANGE INFO (THIS WAS MISSING BEFORE)
            st.info(
                f"{model_type} typical ranges → "
                f"Engine: {min_eng}-{max_eng} L | "
                f"Fuel: {min_fc}-{max_fc} L/100 km"
            )

            engine_size = st.slider(
                "Engine Size (litres)",
                min_value=min_eng,
                max_value=max_eng,
                value=round((min_eng + max_eng) / 2, 1),
                step=0.1
            )

            fuel_consumption = st.slider(
                "Fuel Consumption (L / 100 km)",
                min_value=min_fc,
                max_value=max_fc,
                value=round((min_fc + max_fc) / 2, 1),
                step=0.1
            )

        st.session_state.engine_size = engine_size
        st.session_state.fuel_consumption = fuel_consumption

    if st.button("🔮 Predict Emissions"):
        st.session_state.page = "output"
        st.rerun()

# =================================================
# PAGE 2 : OUTPUT
# =================================================
elif st.session_state.page == "output":

    st.title("📊 Emission Analysis Results")

    vehicle_type = st.session_state.vehicle_type
    distance = st.session_state.distance
    distances = np.arange(1, int(distance) + 1)

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

    else:
        ml_co2 = model.predict(
            np.array([[st.session_state.engine_size,
                       st.session_state.fuel_consumption]])
        )[0]

        final_co2 = ml_co2 * FUEL_ADJUSTMENT[vehicle_type]
        total_co2 = (final_co2 * distance) / 1000

        st.success(f"{vehicle_type} Vehicle CO₂: {final_co2:.2f} g/km")
        st.info(f"Total CO₂ for {distance:.2f} km: {total_co2:.2f} kg")
        st.info(f"Emission Status: {co2_status(final_co2)}")

        ice_val = final_co2
        hybrid_val = final_co2 * 0.6
        ev_val = 0

    comparison_df = pd.DataFrame({
        "Vehicle Type": ["ICE Vehicle", "Hybrid Vehicle", "Electric Vehicle"],
        "CO₂ Emissions (g/km)": [ice_val, hybrid_val, ev_val]
    })

    st.subheader("Vehicle Emission Comparison")
    st.dataframe(comparison_df)

    # BAR GRAPH
    plt.figure(figsize=(8, 4))
    plt.bar(comparison_df["Vehicle Type"],
            comparison_df["CO₂ Emissions (g/km)"])
    st.pyplot(plt)
    plt.close()

    # LINE GRAPH
    plt.figure(figsize=(9, 4))
    plt.plot(distances, ice_val * distances, label="ICE")
    plt.plot(distances, hybrid_val * distances, label="Hybrid")
    plt.plot(distances, ev_val * distances, label="EV")
    plt.legend()
    st.pyplot(plt)
    plt.close()

    st.subheader("How to Reduce CO₂ Emissions")
    reduction_tips()

    if st.button("⬅ Back to Input Page"):
        st.session_state.page = "input"
        st.rerun()

    

