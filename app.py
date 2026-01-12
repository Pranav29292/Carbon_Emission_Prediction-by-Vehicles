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
        "Fuel Type",
        ["Petrol", "Diesel", "CNG", "Electric (EV)"]
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

        energy_consumption = st.slider(
            "Energy Consumption (kWh / 100 km)",
            10.0, 25.0, 15.0, 0.5
        )

        grid_emission = st.slider(
            "Grid Emission Factor (g CO₂ / kWh)",
            300.0, 900.0, 700.0, 10.0
        )

        st.session_state.energy_consumption = energy_consumption
        st.session_state.grid_emission = grid_emission

    # ===============================
    # ICE / CNG VEHICLE INPUT
    # ===============================
    else:
        st.subheader("Vehicle Category")

        vehicle_category = st.selectbox(
            "Select Vehicle Category",
            ["Two-Wheeler", "Four-Wheeler"]
        )

        # ===============================
        # TWO-WHEELER
        # ===============================
        if vehicle_category == "Two-Wheeler":

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

        # ===============================
        # FOUR-WHEELER
        # ===============================
        else:
            car_type = st.selectbox(
                "Four-Wheeler Type",
                ["Hatchback", "Sedan", "SUV"]
            )

            if car_type == "Hatchback":
                model_type = st.selectbox(
                    "Hatchback Model Type",
                    ["Entry Hatchback", "Premium Hatchback"]
                )

                engine_range = {
                    "Entry Hatchback": (0.8, 1.0),
                    "Premium Hatchback": (1.0, 1.2)
                }

                fuel_range = {
                    "Entry Hatchback": (4.0, 5.0),
                    "Premium Hatchback": (5.0, 6.0)
                }

            elif car_type == "Sedan":
                model_type = st.selectbox(
                    "Sedan Model Type",
                    ["Compact Sedan", "Mid-Size Sedan", "Luxury Sedan"]
                )

                engine_range = {
                    "Compact Sedan": (1.2, 1.4),
                    "Mid-Size Sedan": (1.5, 2.0),
                    "Luxury Sedan": (2.0, 3.0)
                }

                fuel_range = {
                    "Compact Sedan": (5.0, 6.5),
                    "Mid-Size Sedan": (6.5, 8.0),
                    "Luxury Sedan": (8.0, 11.0)
                }

            else:
                model_type = st.selectbox(
                    "SUV Model Type",
                    ["Compact SUV", "Mid-Size SUV", "Full-Size SUV"]
                )

                engine_range = {
                    "Compact SUV": (1.3, 1.6),
                    "Mid-Size SUV": (1.8, 2.4),
                    "Full-Size SUV": (2.5, 3.5)
                }

                fuel_range = {
                    "Compact SUV": (6.0, 8.0),
                    "Mid-Size SUV": (8.0, 11.0),
                    "Full-Size SUV": (10.0, 14.0)
                }

            min_eng, max_eng = engine_range[model_type]
            min_fc, max_fc = fuel_range[model_type]

        st.info(
            f"Typical Range → Engine: {min_eng}-{max_eng} L | "
            f"Fuel: {min_fc}-{max_fc} L/100 km"
        )

        engine_size = st.slider(
            "Engine Size (litres)",
            min_eng, max_eng,
            round((min_eng + max_eng) / 2, 2),
            0.01
        )

        fuel_consumption = st.slider(
            "Fuel Consumption (L / 100 km)",
            min_fc, max_fc,
            round((min_fc + max_fc) / 2, 1),
            0.1
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

        co2_per_km = (
            st.session_state.energy_consumption *
            st.session_state.grid_emission
        ) / 100

        total_co2 = (co2_per_km * distance) / 1000

        ice_val = 160
        hybrid_val = co2_per_km * 0.6
        ev_val = co2_per_km

    else:
        ml_pred = model.predict(
            np.array([[st.session_state.engine_size,
                       st.session_state.fuel_consumption]])
        )[0]

        co2_per_km = ml_pred * FUEL_ADJUSTMENT[vehicle_type]
        total_co2 = (co2_per_km * distance) / 1000

        ice_val = co2_per_km
        hybrid_val = co2_per_km * 0.6
        ev_val = 0

    st.success(f"CO₂ Emission: {co2_per_km:.2f} g/km")
    st.info(f"Total CO₂ for {distance} km: {total_co2:.2f} kg")
    st.info(f"Emission Status: {co2_status(co2_per_km)}")

    df = pd.DataFrame({
        "Vehicle Type": ["ICE", "Hybrid", "Electric"],
        "CO₂ Emissions (g/km)": [ice_val, hybrid_val, ev_val]
    })

    st.subheader("Vehicle Emission Comparison")
    st.dataframe(df)

    plt.figure(figsize=(7, 4))
    plt.bar(df["Vehicle Type"], df["CO₂ Emissions (g/km)"])
    st.pyplot(plt)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(distances, ice_val * distances, label="ICE")
    plt.plot(distances, hybrid_val * distances, label="Hybrid")
    plt.plot(distances, ev_val * distances, label="EV")
    plt.legend()
    st.pyplot(plt)
    plt.close()

    st.subheader("How to Reduce CO₂ Emissions")
    reduction_tips()

    if st.button("⬅ Back"):
        st.session_state.page = "input"
        st.rerun()


    

