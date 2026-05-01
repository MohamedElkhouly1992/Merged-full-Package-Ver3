from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd
import streamlit as st

from hvac_v3_engine import (
    BuildingSpec,
    HVACConfig,
    HVAC_PRESETS,
    SCENARIOS,
    SEVERITY_LEVELS,
    CLIMATE_LEVELS,
    run_scenario_model,
    train_surrogate_models,
    run_early_sensitivity_analysis,
    run_robustness_analysis,
)
from report_addons import (
    read_weather_upload,
    build_detailed_tables,
    save_detailed_outputs,
    load_validation_file,
    build_validation_comparison,
    create_zip_from_folder,
    find_result_paths,
    setup_to_json_bytes,
    setup_from_upload,
)

st.set_page_config(page_title="HVAC ROM-Degradation Suite", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {background: linear-gradient(180deg, #07101f 0%, #101729 55%, #151827 100%);} 
.block-container {padding-top: 1.15rem; padding-bottom: 2.2rem; max-width: 1360px;}
h1, h2, h3, h4, h5, h6, p, label, span, div {color: #eaf0fb;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);} 
div[data-baseweb="tab-list"] {gap: 0.55rem; border-bottom: 1px solid rgba(255,255,255,0.10); padding-bottom: 0.25rem;}
button[data-baseweb="tab"] {background: rgba(255,255,255,0.035) !important; border-radius: 14px 14px 0 0 !important; padding: 0.8rem 1.0rem !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.07) !important;}
button[data-baseweb="tab"][aria-selected="true"] {color: #ff686b !important; border-bottom: 2px solid #ff686b !important; background: rgba(255,255,255,0.075) !important;}
div[data-testid="stExpander"] {border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; background: rgba(255,255,255,0.035); margin-bottom: 0.9rem;}
div[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 0.5rem 0.7rem;}
div[data-testid="stDataFrame"] {border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; overflow: hidden;}
div.stButton > button {border-radius: 14px !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.18) !important;}
.small-muted {color:#aeb8ce; font-size:0.94rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div style="padding: 0.55rem 0 1.0rem 0;">
      <div style="font-size: 2.7rem; font-weight: 850; letter-spacing: -0.035em; color: #f6f8fc;">
        HVAC ROM-Degradation Suite
      </div>
      <div class="small-muted" style="max-width: 1040px; margin-top:0.35rem;">
        Reduced-order HVAC energy, degradation, maintenance, climate-scenario, early sensitivity, robustness, validation, reporting, and CatBoost surrogate modelling platform.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def default_zone_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"zone_name": "Lecture_01", "zone_type": "Lecture", "area_m2": 200.0, "occ_density": 0.12, "term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
        {"zone_name": "Office_01", "zone_type": "Office", "area_m2": 120.0, "occ_density": 0.06, "term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
        {"zone_name": "Lab_01", "zone_type": "Lab", "area_m2": 180.0, "occ_density": 0.08, "term_factor": 0.90, "break_factor": 0.45, "summer_factor": 0.30},
        {"zone_name": "Corridor", "zone_type": "Corridor", "area_m2": 100.0, "occ_density": 0.01, "term_factor": 0.60, "break_factor": 0.45, "summer_factor": 0.35},
        {"zone_name": "Service_01", "zone_type": "Service", "area_m2": 80.0, "occ_density": 0.02, "term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
    ])


def download_file_button(path: str | Path, label: str, key: str | None = None):
    path = Path(path)
    if path.exists() and path.is_file():
        with path.open("rb") as f:
            st.download_button(label, f.read(), file_name=path.name, key=key or f"dl_{path.name}")


def apply_setup_dict(data: dict):
    for k, v in data.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                st.session_state[kk] = vv
        else:
            st.session_state[k] = v


BUILT_IN_SETUPS = {
    "Educational medium building": {
        "building_type": "Educational / University building", "location": "User-defined", "area_m2": 5000.0, "floors": 4, "n_spaces": 40,
        "occupancy_density": 0.08, "lighting_w_m2": 10.0, "equipment_w_m2": 8.0, "sensible_w_per_person": 75.0,
        "airflow_m3h_m2": 4.0, "cooling_w_m2": 100.0, "heating_w_m2": 55.0,
        "wall_u": 0.60, "roof_u": 0.35, "window_u": 2.70, "shgc": 0.35, "glazing_ratio": 0.30, "infiltration_ach": 0.50,
        "hvac_system_type": "Chiller_AHU",
    },
    "Small office / training center": {
        "building_type": "Office / Training center", "location": "User-defined", "area_m2": 1200.0, "floors": 3, "n_spaces": 18,
        "occupancy_density": 0.06, "lighting_w_m2": 9.0, "equipment_w_m2": 11.0, "sensible_w_per_person": 75.0,
        "airflow_m3h_m2": 3.5, "cooling_w_m2": 95.0, "heating_w_m2": 45.0,
        "wall_u": 0.65, "roof_u": 0.40, "window_u": 2.80, "shgc": 0.38, "glazing_ratio": 0.25, "infiltration_ach": 0.45,
        "hvac_system_type": "VRF",
    },
}


def current_setup_dict() -> dict:
    keys = [
        "building_type", "location", "area_m2", "floors", "n_spaces", "occupancy_density", "lighting_w_m2", "equipment_w_m2", "sensible_w_per_person",
        "airflow_m3h_m2", "cooling_w_m2", "heating_w_m2", "wall_u", "roof_u", "window_u", "shgc", "glazing_ratio", "infiltration_ach",
        "hvac_system_type", "use_hvac_preset", "cop_cool_nom", "cop_heat_nom", "fan_eff", "dp_clean", "dp_warn", "dp_thresh", "dp_max",
        "years", "time_step_label", "t_set", "t_sp_min", "t_sp_max", "af_min", "af_max", "cop_aging_rate", "rf_star", "b_foul", "dust_rate", "k_clog", "deg_trigger",
        "filter_interval", "hx_interval", "e_price", "co2_factor", "cost_filter", "cost_hx", "degradation_model", "linear_deg_per_day", "exp_deg_rate_per_day",
    ]
    return {k: st.session_state.get(k) for k in keys if k in st.session_state}


def cfg_with_switches(cfg: HVACConfig, switches: dict[str, bool]) -> HVACConfig:
    for attr, val in switches.items():
        if hasattr(cfg, attr):
            setattr(cfg, attr, bool(val))
    return cfg


def build_weather_controls(prefix: str = "main"):
    c1, c2, c3 = st.columns([1.1, 1.0, 1.0])
    weather_mode_ui = c1.selectbox(
        "Weather source",
        ["synthetic", "upload_csv_epw", "epw_path", "csv_path"],
        format_func=lambda x: {"synthetic": "Synthetic daily weather", "upload_csv_epw": "Upload CSV/EPW directly", "epw_path": "EPW path", "csv_path": "CSV path"}[x],
        key=f"{prefix}_weather_mode",
    )
    random_state = int(c2.number_input("Random state", min_value=1, value=42, step=1, key=f"{prefix}_random_state"))
    out_dir = c3.text_input("Output folder", f"{prefix}_run", key=f"{prefix}_out_dir")
    weather_df = None
    epw_path = None
    csv_path = None
    if weather_mode_ui == "upload_csv_epw":
        uploaded_weather = st.file_uploader("Upload weather file (.csv or .epw)", type=["csv", "epw", "txt"], key=f"{prefix}_weather_upload")
        if uploaded_weather is not None:
            try:
                weather_df = read_weather_upload(uploaded_weather)
                st.session_state[f"{prefix}_uploaded_weather_df"] = weather_df
                st.success(f"Weather upload parsed successfully: {len(weather_df)} normalized daily records")
                st.dataframe(weather_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Weather upload error: {e}")
        else:
            weather_df = st.session_state.get(f"{prefix}_uploaded_weather_df")
    elif weather_mode_ui == "epw_path":
        epw_path = st.text_input("EPW file path", "", key=f"{prefix}_epw_path")
    elif weather_mode_ui == "csv_path":
        csv_path = st.text_input("CSV weather file path", "", key=f"{prefix}_csv_path")
    engine_weather_mode = {"synthetic": "synthetic", "upload_csv_epw": "uploaded", "epw_path": "epw", "csv_path": "csv"}[weather_mode_ui]
    return engine_weather_mode, epw_path, csv_path, weather_df, random_state, out_dir


# Defaults
for k, v in BUILT_IN_SETUPS["Educational medium building"].items():
    st.session_state.setdefault(k, v)

# Main tabs: setup comes first by design.
tabs = st.tabs([
    "Building Identity & Setup",
    "Parameter Switches",
    "Scenario Modeling",
    "Sensitivity & Robustness",
    "Extra UI Tools",
    "KPI Charts",
    "Surrogate Train / Predict",
    "Exports",
    "Guide",
])

with tabs[0]:
    st.subheader("Building identity and configuration setup")
    c1, c2, c3 = st.columns([1.1, 1.2, 1.0])
    preset_name = c1.selectbox("Saved / built-in setup", list(BUILT_IN_SETUPS.keys()), key="preset_selector")
    if c2.button("Apply selected setup"):
        apply_setup_dict(BUILT_IN_SETUPS[preset_name])
        st.success(f"Applied setup: {preset_name}")
    upload_setup = c3.file_uploader("Upload setup JSON", type=["json"], key="setup_upload")
    if upload_setup is not None:
        try:
            apply_setup_dict(setup_from_upload(upload_setup))
            st.success("Setup JSON loaded into the current session.")
        except Exception as e:
            st.error(str(e))
    st.download_button("Download current setup JSON", setup_to_json_bytes(current_setup_dict()), file_name="building_setup.json")

    st.markdown("### 1. Building identity")
    c1, c2 = st.columns(2)
    building_type = c1.text_input("Building type", key="building_type")
    location = c2.text_input("Location / weather label", key="location")

    st.markdown("### 2. Geometry")
    c1, c2, c3 = st.columns(3)
    area_m2 = c1.number_input("Conditioned area (m²)", min_value=100.0, step=100.0, key="area_m2")
    floors = c2.number_input("Floors", min_value=1, step=1, key="floors")
    n_spaces = c3.number_input("Number of spaces", min_value=1, step=1, key="n_spaces")

    st.markdown("### 3. Envelope")
    c1, c2, c3 = st.columns(3)
    wall_u = c1.number_input("Wall U-value (W/m²K)", min_value=0.01, step=0.05, key="wall_u")
    roof_u = c2.number_input("Roof U-value (W/m²K)", min_value=0.01, step=0.05, key="roof_u")
    window_u = c3.number_input("Window U-value (W/m²K)", min_value=0.01, step=0.1, key="window_u")
    c1, c2, c3 = st.columns(3)
    shgc = c1.number_input("SHGC", min_value=0.01, max_value=0.95, step=0.01, key="shgc")
    glazing_ratio = c2.number_input("Glazing ratio", min_value=0.01, max_value=0.95, step=0.01, key="glazing_ratio")
    infiltration_ach = c3.number_input("Infiltration (ACH)", min_value=0.0, step=0.1, key="infiltration_ach")

    st.markdown("### 4. Internal loads")
    c1, c2, c3, c4 = st.columns(4)
    occupancy_density = c1.number_input("Occupancy density (person/m²)", min_value=0.0001, step=0.01, format="%.4f", key="occupancy_density")
    lighting_w_m2 = c2.number_input("Lighting power density (W/m²)", min_value=0.0, step=1.0, key="lighting_w_m2")
    equipment_w_m2 = c3.number_input("Equipment power density (W/m²)", min_value=0.0, step=1.0, key="equipment_w_m2")
    sensible_w_per_person = c4.number_input("Sensible heat/person (W)", min_value=1.0, step=5.0, key="sensible_w_per_person")

    st.markdown("### 5. HVAC sizing and component")
    c1, c2, c3 = st.columns(3)
    hvac_system_type = c1.selectbox("HVAC system type", list(HVAC_PRESETS.keys()), key="hvac_system_type")
    use_hvac_preset = c2.checkbox("Apply selected HVAC preset", value=st.session_state.get("use_hvac_preset", True), key="use_hvac_preset")
    years = c3.number_input("Simulation years", min_value=1, max_value=50, step=1, key="years")

    c1, c2, c3 = st.columns(3)
    airflow_m3h_m2 = c1.number_input("Airflow intensity (m³/h·m²)", min_value=0.01, step=0.1, key="airflow_m3h_m2")
    cooling_w_m2 = c2.number_input("Cooling design intensity (W/m²)", min_value=1.0, step=5.0, key="cooling_w_m2")
    heating_w_m2 = c3.number_input("Heating design intensity (W/m²)", min_value=1.0, step=5.0, key="heating_w_m2")

    c1, c2, c3 = st.columns(3)
    cop_cool_nom = c1.number_input("Nominal cooling COP", min_value=0.8, value=float(st.session_state.get("cop_cool_nom", 4.5)), step=0.1, key="cop_cool_nom")
    cop_heat_nom = c2.number_input("Nominal heating COP", min_value=0.8, value=float(st.session_state.get("cop_heat_nom", 3.2)), step=0.1, key="cop_heat_nom")
    fan_eff = c3.number_input("Fan total efficiency", min_value=0.1, max_value=0.95, value=float(st.session_state.get("fan_eff", 0.70)), step=0.01, key="fan_eff")

    c1, c2, c3, c4 = st.columns(4)
    dp_clean = c1.number_input("Clean static pressure (Pa)", min_value=1.0, value=float(st.session_state.get("dp_clean", 150.0)), step=10.0, key="dp_clean")
    dp_warn = c2.number_input("Warning pressure (Pa)", min_value=1.0, value=float(st.session_state.get("dp_warn", 320.0)), step=10.0, key="dp_warn")
    dp_thresh = c3.number_input("Replacement threshold pressure (Pa)", min_value=1.0, value=float(st.session_state.get("dp_thresh", 420.0)), step=10.0, key="dp_thresh")
    dp_max = c4.number_input("Maximum pressure (Pa)", min_value=1.0, value=float(st.session_state.get("dp_max", 450.0)), step=10.0, key="dp_max")

    st.markdown("### 6. Time-series, controls, cost, carbon")
    c1, c2, c3 = st.columns(3)
    time_step_label = c1.selectbox("Calculation time step", ["Daily", "12-hour", "6-hour", "3-hour", "Hourly"], key="time_step_label")
    time_step_hours = {"Daily": 24.0, "12-hour": 12.0, "6-hour": 6.0, "3-hour": 3.0, "Hourly": 1.0}[time_step_label]
    t_set = c2.number_input("Main setpoint T_SET (°C)", min_value=16.0, max_value=30.0, value=float(st.session_state.get("t_set", 23.0)), step=0.5, key="t_set")
    e_price = c3.number_input("Electricity price ($/kWh)", min_value=0.0, value=float(st.session_state.get("e_price", 0.12)), step=0.01, key="e_price")
    c1, c2, c3, c4 = st.columns(4)
    t_sp_min = c1.number_input("S3 min setpoint (°C)", min_value=16.0, max_value=30.0, value=float(st.session_state.get("t_sp_min", 21.0)), step=0.5, key="t_sp_min")
    t_sp_max = c2.number_input("S3 max setpoint (°C)", min_value=16.0, max_value=30.0, value=float(st.session_state.get("t_sp_max", 26.0)), step=0.5, key="t_sp_max")
    af_min = c3.number_input("S3 min airflow factor", min_value=0.1, max_value=1.0, value=float(st.session_state.get("af_min", 0.55)), step=0.05, key="af_min")
    af_max = c4.number_input("S3 max airflow factor", min_value=0.1, max_value=1.5, value=float(st.session_state.get("af_max", 1.0)), step=0.05, key="af_max")
    c1, c2, c3, c4 = st.columns(4)
    co2_factor = c1.number_input("CO₂ factor (kg/kWh)", min_value=0.0, value=float(st.session_state.get("co2_factor", 0.536)), step=0.01, key="co2_factor")
    cost_filter = c2.number_input("Filter cost", min_value=0.0, value=float(st.session_state.get("cost_filter", 50.0)), step=5.0, key="cost_filter")
    cost_hx = c3.number_input("HX cleaning cost", min_value=0.0, value=float(st.session_state.get("cost_hx", 300.0)), step=10.0, key="cost_hx")
    filter_interval = c4.number_input("Filter interval (days)", min_value=1, value=int(st.session_state.get("filter_interval", 90)), step=1, key="filter_interval")
    hx_interval = st.number_input("HX cleaning interval (days)", min_value=1, value=int(st.session_state.get("hx_interval", 180)), step=1, key="hx_interval")

    st.markdown("### 7. Degradation parameters")
    c1, c2, c3 = st.columns(3)
    degradation_model = c1.selectbox("Degradation model", ["physics", "linear_ts", "exponential_ts"], key="degradation_model", format_func=lambda x: {"physics":"Physics-based fouling/clogging", "linear_ts":"Linear time-series", "exponential_ts":"Exponential time-series"}[x])
    cop_aging_rate = c2.number_input("COP aging rate", min_value=0.0, value=float(st.session_state.get("cop_aging_rate", 0.005)), step=0.001, format="%.4f", key="cop_aging_rate")
    deg_trigger = c3.number_input("Degradation trigger", min_value=0.0, max_value=1.5, value=float(st.session_state.get("deg_trigger", 0.55)), step=0.01, key="deg_trigger")
    c1, c2, c3, c4 = st.columns(4)
    rf_star = c1.number_input("RF* fouling asymptote", min_value=1e-8, value=float(st.session_state.get("rf_star", 2e-4)), format="%.7f", key="rf_star")
    b_foul = c2.number_input("Fouling growth constant B", min_value=0.0, value=float(st.session_state.get("b_foul", 0.015)), step=0.001, format="%.4f", key="b_foul")
    dust_rate = c3.number_input("Dust accumulation rate", min_value=0.0, value=float(st.session_state.get("dust_rate", 1.2)), step=0.1, key="dust_rate")
    k_clog = c4.number_input("Clogging coefficient", min_value=0.0, value=float(st.session_state.get("k_clog", 6.0)), step=0.1, key="k_clog")
    c1, c2 = st.columns(2)
    linear_deg_per_day = c1.number_input("Linear degradation slope per day", min_value=0.0, value=float(st.session_state.get("linear_deg_per_day", 0.00012)), step=0.00001, format="%.6f", key="linear_deg_per_day")
    exp_deg_rate_per_day = c2.number_input("Exponential degradation rate per day", min_value=0.0, value=float(st.session_state.get("exp_deg_rate_per_day", 0.00018)), step=0.00001, format="%.6f", key="exp_deg_rate_per_day")

with tabs[1]:
    st.subheader("Parameter switches / quick control")
    st.markdown("These switches are passed into `HVACConfig` and affect the engine calculation directly. They do not create a duplicate model in the UI.")
    quick = st.columns(4)
    if quick[0].button("Enable all main terms"):
        for k in ["USE_ENVELOPE", "USE_WALLS", "USE_ROOF", "USE_WINDOWS", "USE_SOLAR", "USE_INFILTRATION", "USE_INTERNAL_GAINS", "USE_PEOPLE_GAINS", "USE_LIGHTING_GAINS", "USE_EQUIPMENT_GAINS", "USE_HVAC_FANS", "USE_COOLING", "USE_HEATING", "USE_DEGRADATION", "USE_CARBON", "USE_MAINTENANCE_COST"]:
            st.session_state[k] = True
    if quick[1].button("Thermal only: no degradation"):
        st.session_state["USE_DEGRADATION"] = False
        st.session_state["USE_MAINTENANCE_COST"] = False
    if quick[2].button("Envelope + weather only"):
        for k in ["USE_INTERNAL_GAINS", "USE_PEOPLE_GAINS", "USE_LIGHTING_GAINS", "USE_EQUIPMENT_GAINS", "USE_DEGRADATION", "USE_MAINTENANCE_COST"]:
            st.session_state[k] = False
    if quick[3].button("Disable optional post-tools"):
        for k in ["post_zone_analysis", "post_validation", "post_benchmark", "post_surrogate"]:
            st.session_state[k] = False

    switch_names = [
        ("USE_ENVELOPE", "Envelope terms"), ("USE_WALLS", "Walls"), ("USE_ROOF", "Roof"), ("USE_WINDOWS", "Windows"),
        ("USE_SOLAR", "Solar gains"), ("USE_INFILTRATION", "Infiltration"), ("USE_INTERNAL_GAINS", "Internal gains"),
        ("USE_PEOPLE_GAINS", "People gains"), ("USE_LIGHTING_GAINS", "Lighting gains"), ("USE_EQUIPMENT_GAINS", "Equipment gains"),
        ("USE_HVAC_FANS", "HVAC fan energy"), ("USE_COOLING", "Cooling"), ("USE_HEATING", "Heating"),
        ("USE_DEGRADATION", "Degradation"), ("USE_CARBON", "Carbon"), ("USE_MAINTENANCE_COST", "Maintenance cost"),
    ]
    cols = st.columns(4)
    switches = {}
    for i, (key, label) in enumerate(switch_names):
        st.session_state.setdefault(key, True)
        with cols[i % 4]:
            switches[key] = st.checkbox(label, key=key)
    st.markdown("### Post-processing switches")
    c1, c2, c3, c4 = st.columns(4)
    post_zone = c1.checkbox("Zone analysis", value=st.session_state.get("post_zone_analysis", True), key="post_zone_analysis")
    post_validation = c2.checkbox("Validation", value=st.session_state.get("post_validation", True), key="post_validation")
    post_benchmark = c3.checkbox("Benchmark/sensitivity sheets", value=st.session_state.get("post_benchmark", True), key="post_benchmark")
    post_surrogate = c4.checkbox("Surrogate modelling", value=st.session_state.get("post_surrogate", True), key="post_surrogate")

# Build engine objects after setup/switches are defined.
bldg = BuildingSpec(
    building_type=st.session_state.get("building_type", "Educational / University building"),
    location=st.session_state.get("location", "User-defined"),
    conditioned_area_m2=float(st.session_state.get("area_m2", 5000.0)),
    floors=int(st.session_state.get("floors", 4)),
    n_spaces=int(st.session_state.get("n_spaces", 40)),
    occupancy_density_p_m2=float(st.session_state.get("occupancy_density", 0.08)),
    lighting_w_m2=float(st.session_state.get("lighting_w_m2", 10.0)),
    equipment_w_m2=float(st.session_state.get("equipment_w_m2", 8.0)),
    airflow_m3h_m2=float(st.session_state.get("airflow_m3h_m2", 4.0)),
    infiltration_ach=float(st.session_state.get("infiltration_ach", 0.5)),
    sensible_w_per_person=float(st.session_state.get("sensible_w_per_person", 75.0)),
    cooling_intensity_w_m2=float(st.session_state.get("cooling_w_m2", 100.0)),
    heating_intensity_w_m2=float(st.session_state.get("heating_w_m2", 55.0)),
    wall_u=float(st.session_state.get("wall_u", 0.6)),
    roof_u=float(st.session_state.get("roof_u", 0.35)),
    window_u=float(st.session_state.get("window_u", 2.7)),
    shgc=float(st.session_state.get("shgc", 0.35)),
    glazing_ratio=float(st.session_state.get("glazing_ratio", 0.30)),
)
time_step_hours = {"Daily": 24.0, "12-hour": 12.0, "6-hour": 6.0, "3-hour": 3.0, "Hourly": 1.0}[st.session_state.get("time_step_label", "Daily")]
cfg = HVACConfig(
    years=int(st.session_state.get("years", 20)),
    hvac_system_type=st.session_state.get("hvac_system_type", "Chiller_AHU"),
    COP_COOL_NOM=float(st.session_state.get("cop_cool_nom", 4.5)),
    COP_HEAT_NOM=float(st.session_state.get("cop_heat_nom", 3.2)),
    FAN_EFF=float(st.session_state.get("fan_eff", 0.70)),
    T_SET=float(st.session_state.get("t_set", 23.0)),
    T_SP_MIN=float(st.session_state.get("t_sp_min", 21.0)),
    T_SP_MAX=float(st.session_state.get("t_sp_max", 26.0)),
    AF_MIN=float(st.session_state.get("af_min", 0.55)),
    AF_MAX=float(st.session_state.get("af_max", 1.0)),
    DP_CLEAN=float(st.session_state.get("dp_clean", 150.0)),
    DP_WARN=float(st.session_state.get("dp_warn", 320.0)),
    DP_THRESH=float(st.session_state.get("dp_thresh", 420.0)),
    DP_MAX=float(st.session_state.get("dp_max", 450.0)),
    COP_AGING_RATE=float(st.session_state.get("cop_aging_rate", 0.005)),
    RF_STAR=float(st.session_state.get("rf_star", 2e-4)),
    B_FOUL=float(st.session_state.get("b_foul", 0.015)),
    DUST_RATE=float(st.session_state.get("dust_rate", 1.2)),
    K_CLOG=float(st.session_state.get("k_clog", 6.0)),
    DEG_TRIGGER=float(st.session_state.get("deg_trigger", 0.55)),
    E_PRICE=float(st.session_state.get("e_price", 0.12)),
    CO2_FACTOR=float(st.session_state.get("co2_factor", 0.536)),
    COST_FILTER=float(st.session_state.get("cost_filter", 50.0)),
    COST_HX=float(st.session_state.get("cost_hx", 300.0)),
    FILTER_INTERVAL=int(st.session_state.get("filter_interval", 90)),
    HX_INTERVAL=int(st.session_state.get("hx_interval", 180)),
    degradation_model=st.session_state.get("degradation_model", "physics"),
    LINEAR_DEG_PER_DAY=float(st.session_state.get("linear_deg_per_day", 0.00012)),
    EXP_DEG_RATE_PER_DAY=float(st.session_state.get("exp_deg_rate_per_day", 0.00018)),
    TIME_STEP_HOURS=time_step_hours,
    USE_HVAC_PRESET=bool(st.session_state.get("use_hvac_preset", True)),
)
cfg = cfg_with_switches(cfg, {k: st.session_state.get(k, True) for k, _ in switch_names})

with tabs[2]:
    st.subheader("Scenario modeling")
    c1, c2, c3 = st.columns(3)
    axis_mode = c1.selectbox(
        "Analysis mode",
        ["baseline_scenario", "one_severity", "one_strategy", "two_axis", "three_axis"],
        format_func=lambda x: {"baseline_scenario": "Baseline Scenario only", "one_severity": "One-axis severity", "one_strategy": "One-axis strategy S0–S3", "two_axis": "Strategy × severity", "three_axis": "Strategy × severity × climate"}[x],
    )
    fixed_strategy = c2.selectbox("Fixed / baseline strategy", list(SCENARIOS.keys()), format_func=lambda x: f"{x} — {SCENARIOS[x]}")
    fixed_severity = c3.selectbox("Fixed severity", list(SEVERITY_LEVELS.keys()), index=1)
    fixed_climate = st.selectbox("Fixed climate", list(CLIMATE_LEVELS.keys()))

    st.info(f"Selected calculation time step: {time_step_hours:g} h. Original daily model is preserved when 24 h is selected. Sub-daily modes scale energy, degradation growth, dust accumulation, and cost by period length.")
    engine_weather_mode, epw_path, csv_path, weather_df, random_state, out_dir = build_weather_controls("scenario")

    include_baseline_layer = st.checkbox("Export baseline no-degradation layer", value=True)
    include_baseline_as_scenario = st.checkbox("Add Baseline Scenario to main output calculation", value=True)
    use_zone_occ = st.checkbox("Use zone-specific occupancy input", value=False)
    zone_df = None
    if use_zone_occ:
        zone_df = st.data_editor(default_zone_table(), num_rows="dynamic", use_container_width=True)

    if st.button("Run selected model", type="primary"):
        try:
            if engine_weather_mode == "uploaded" and weather_df is None:
                st.warning("Upload a CSV/EPW weather file first, or select another weather source.")
                st.stop()
            result = run_scenario_model(
                output_dir=out_dir,
                axis_mode=axis_mode,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode,
                epw_path=epw_path if epw_path else None,
                csv_path=csv_path if csv_path else None,
                weather_df=weather_df,
                fixed_strategy=fixed_strategy,
                fixed_severity=fixed_severity,
                fixed_climate=fixed_climate,
                zone_df=zone_df,
                random_state=random_state,
                include_baseline_layer=include_baseline_layer,
                include_baseline_as_scenario=include_baseline_as_scenario,
                degradation_model=st.session_state.get("degradation_model", "physics"),
                time_step_hours=time_step_hours,
            )
            st.session_state["last_result"] = result
            st.session_state["last_result_dir"] = out_dir
            st.session_state["last_zone_df"] = zone_df
            tables = build_detailed_tables(out_dir, bldg=bldg, cfg=cfg, zone_df=zone_df)
            detailed_paths = save_detailed_outputs(out_dir, tables)
            st.session_state["last_detailed_paths"] = detailed_paths
            st.success("Model run and detailed outputs finished.")
            st.json({**result, "extra_detailed_outputs": detailed_paths})
            summary_path = Path(result["summary_csv"])
            if summary_path.exists():
                st.dataframe(pd.read_csv(summary_path), use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[3]:
    st.subheader("Early benchmark sensitivity and robustness analysis")
    st.markdown(
        """
        **Early benchmark sensitivity** runs a fast one-at-a-time perturbation around the current setup and ranks parameters by dimensionless elasticity against key KPIs.  
        **Robustness analysis** runs bounded Monte-Carlo input perturbations and reports KPI spread, coefficient of variation, and 5–95% bands.
        """
    )
    c1, c2, c3 = st.columns(3)
    analysis_years = int(c1.number_input("Analysis years", min_value=1, max_value=10, value=1, step=1))
    sens_pct = float(c2.number_input("Sensitivity perturbation ±%", min_value=1.0, max_value=50.0, value=10.0, step=1.0)) / 100.0
    robust_pct = float(c3.number_input("Robustness uncertainty ±%", min_value=1.0, max_value=50.0, value=10.0, step=1.0)) / 100.0
    c1, c2, c3, c4 = st.columns(4)
    sens_strategy = c1.selectbox("Strategy for analysis", list(SCENARIOS.keys()), index=2, key="sens_strategy")
    sens_severity = c2.selectbox("Severity for analysis", list(SEVERITY_LEVELS.keys()), index=1, key="sens_severity")
    sens_climate = c3.selectbox("Climate for analysis", list(CLIMATE_LEVELS.keys()), key="sens_climate")
    n_samples = int(c4.number_input("Robustness samples", min_value=3, max_value=200, value=12, step=1))
    engine_weather_mode_s, epw_path_s, csv_path_s, weather_df_s, random_state_s, out_dir_s = build_weather_controls("sensitivity")
    out_dir_s = str(Path(out_dir_s) / "sensitivity_robustness")
    st.caption(f"Outputs will be written to: {out_dir_s}")

    selected_params = st.multiselect(
        "Optional: limit parameters; leave empty to screen all supported parameters",
        ["conditioned_area_m2", "occupancy_density_p_m2", "lighting_w_m2", "equipment_w_m2", "airflow_m3h_m2", "cooling_intensity_w_m2", "heating_intensity_w_m2", "wall_u", "roof_u", "window_u", "shgc", "glazing_ratio", "infiltration_ach", "COP_COOL_NOM", "COP_HEAT_NOM", "FAN_EFF", "COP_AGING_RATE", "RF_STAR", "B_FOUL", "DUST_RATE", "K_CLOG"],
        default=[],
    )
    col_a, col_b = st.columns(2)
    if col_a.button("Run early benchmark sensitivity", type="primary"):
        try:
            if engine_weather_mode_s == "uploaded" and weather_df_s is None:
                st.warning("Upload weather first, or select synthetic/path weather.")
                st.stop()
            result = run_early_sensitivity_analysis(
                output_dir=out_dir_s,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode_s,
                epw_path=epw_path_s,
                csv_path=csv_path_s,
                weather_df=weather_df_s,
                fixed_strategy=sens_strategy,
                fixed_severity=sens_severity,
                fixed_climate=sens_climate,
                degradation_model=st.session_state.get("degradation_model", "physics"),
                perturbation_pct=sens_pct,
                analysis_years=analysis_years,
                random_state=random_state_s,
                time_step_hours=time_step_hours,
                parameter_names=selected_params or None,
            )
            st.success("Early sensitivity analysis finished.")
            st.json(result)
            df = pd.read_csv(result["ranking_csv"])
            st.dataframe(df, use_container_width=True)
            if not df.empty:
                st.bar_chart(df.set_index("label")["composite_importance"])
        except Exception as e:
            st.exception(e)
    if col_b.button("Run robustness analysis"):
        try:
            if engine_weather_mode_s == "uploaded" and weather_df_s is None:
                st.warning("Upload weather first, or select synthetic/path weather.")
                st.stop()
            result = run_robustness_analysis(
                output_dir=out_dir_s,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode_s,
                epw_path=epw_path_s,
                csv_path=csv_path_s,
                weather_df=weather_df_s,
                fixed_strategy=sens_strategy,
                fixed_severity=sens_severity,
                fixed_climate=sens_climate,
                degradation_model=st.session_state.get("degradation_model", "physics"),
                n_samples=n_samples,
                uncertainty_pct=robust_pct,
                analysis_years=analysis_years,
                random_state=random_state_s,
                time_step_hours=time_step_hours,
                parameter_names=selected_params or None,
            )
            st.success("Robustness analysis finished.")
            st.json(result)
            df = pd.read_csv(result["summary_csv"])
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[4]:
    st.subheader("Extra UI tools: validation, benchmark summary, zone tables, upload handling")
    target_folder = st.text_input("Result folder for extra tools", st.session_state.get("last_result_dir", "scenario_run"), key="extra_folder")
    paths = find_result_paths(target_folder)
    if paths["summary"].exists():
        summary_df = pd.read_csv(paths["summary"])
        st.markdown("### Validation upload")
        vfile = st.file_uploader("Upload validation CSV from DesignBuilder, EnergyPlus, measured data, or published reference", type=["csv"], key="validation_file")
        if vfile is not None:
            validation_df = load_validation_file(vfile)
            comparison = build_validation_comparison(summary_df, validation_df, source_name=Path(vfile.name).stem)
            comparison_path = Path(target_folder) / "validation_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            st.dataframe(comparison, use_container_width=True)
            download_file_button(comparison_path, "Download validation_comparison.csv")

        st.markdown("### Benchmark / sensitivity summary sheet")
        if (Path(target_folder) / "benchmark_summary.csv").exists():
            bench = pd.read_csv(Path(target_folder) / "benchmark_summary.csv")
        else:
            tables = build_detailed_tables(target_folder, bldg=bldg, cfg=cfg, zone_df=st.session_state.get("last_zone_df"))
            save_detailed_outputs(target_folder, tables)
            bench = tables.get("benchmark_summary", pd.DataFrame())
        st.dataframe(bench, use_container_width=True)
        if len(bench) and "energy_delta_pct" in bench.columns and "scenario_combo_3axis" in bench.columns:
            st.bar_chart(bench.set_index("scenario_combo_3axis")["energy_delta_pct"])

        st.markdown("### Zone analysis")
        zone_path = Path(target_folder) / "zone_analysis.csv"
        if zone_path.exists():
            zdf = pd.read_csv(zone_path)
            st.dataframe(zdf.head(300), use_container_width=True)
            download_file_button(zone_path, "Download zone_analysis.csv")
        else:
            st.info("Run the model with zone-specific occupancy enabled to generate zone analysis.")
    else:
        st.info("Run a model first, or type an existing result folder.")

with tabs[5]:
    st.subheader("KPI charts")
    folder = Path(st.text_input("Result folder", st.session_state.get("last_result_dir", "scenario_run"), key="kpi_folder"))
    if folder.exists():
        paths = find_result_paths(folder)
        if paths["summary"].exists():
            kpi = pd.read_csv(paths["summary"])
            st.dataframe(kpi, use_container_width=True)
            for metric in ["Total Energy MWh", "Mean Degradation Index", "Mean Comfort Deviation C", "Total CO2 tonne"]:
                if metric in kpi.columns and "scenario_combo_3axis" in kpi.columns:
                    st.line_chart(kpi.set_index("scenario_combo_3axis")[metric])
        figs = folder / "figures"
        if figs.exists():
            img_files = sorted(figs.glob("*.png"))[:24]
            cols = st.columns(2)
            for i, img in enumerate(img_files):
                with cols[i % 2]:
                    st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No result folder found yet.")

with tabs[6]:
    st.subheader("Train CatBoost surrogate")
    dataset_path = st.text_input("Input dataset CSV", str(Path(st.session_state.get("last_result_dir", "scenario_run")) / "matrix_ml_dataset.csv"))
    surrogate_out = st.text_input("Surrogate output folder", "surrogate_run")
    n_iter_search = int(st.number_input("CatBoost search iterations", min_value=2, value=6, step=1))
    shap_sample = int(st.number_input("SHAP sample size", min_value=100, value=1000, step=100))
    if st.button("Train CatBoost surrogate"):
        try:
            result = train_surrogate_models(dataset_path, surrogate_out, n_iter_search, shap_sample, int(42))
            st.success("Surrogate training finished.")
            st.json(result)
            p = Path(result["metrics_csv"])
            if p.exists():
                st.dataframe(pd.read_csv(p), use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[7]:
    st.subheader("Exports and results")
    folder = Path(st.text_input("Folder to inspect/export", st.session_state.get("last_result_dir", "scenario_run"), key="export_folder"))
    if folder.exists():
        csvs = sorted(folder.glob("*.csv"))
        st.write(f"CSV files found: {len(csvs)}")
        for csvf in csvs[:18]:
            with st.expander(csvf.name):
                try:
                    st.dataframe(pd.read_csv(csvf).head(80), use_container_width=True)
                except Exception as e:
                    st.warning(str(e))
                download_file_button(csvf, f"Download {csvf.name}", key=f"download_{csvf.name}")
        for special in ["results_export.xlsx", "detailed_outputs.xlsx", "results_report.pdf", "surrogate_export.xlsx", "surrogate_report.pdf"]:
            download_file_button(folder / special, f"Download {special}", key=f"download_{special}")
        if st.button("Create ZIP bundle for this run"):
            zip_path = create_zip_from_folder(folder)
            st.success(f"ZIP created: {zip_path}")
            download_file_button(zip_path, "Download ZIP bundle")
    else:
        st.info("No folder found yet.")

with tabs[8]:
    st.subheader("Model and deployment guide")
    st.markdown(
        """
        ### Calculation basis
        The engine remains the single calculation authority. The Streamlit app only collects inputs, calls `run_scenario_model()`, and displays/exports results.

        ### Time-series selector
        The calculation time-step selector supports Daily, 12-hour, 6-hour, 3-hour, and Hourly. The original model is recovered at 24 h. Sub-daily modes preserve the same load, COP, maintenance, and degradation equations, while scaling duration-dependent terms by the selected period.

        ### Early benchmark sensitivity
        The early sensitivity analysis perturbs each selected parameter down and up around the baseline and ranks parameters by central elasticity:

        `elasticity = (% KPI change) / (% input change)`

        A high absolute elasticity means that the KPI is more sensitive to that input.

        ### Robustness analysis
        Robustness analysis samples uncertain inputs inside a bounded uniform range, runs the selected scenario repeatedly, and reports mean, standard deviation, coefficient of variation, and 5–95% KPI bands.

        ### Run locally
        ```bash
        pip install -r requirements.txt
        streamlit run streamlit_app.py
        ```
        """
    )
