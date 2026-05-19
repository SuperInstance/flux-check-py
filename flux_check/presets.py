"""
FLUX industry presets — constraint bounds for common domains.

Each preset defines up to 8 constraints with lo/hi bounds and names.
"""

from typing import Dict, List

PRESETS: Dict[str, Dict] = {
    "automotive": {
        "description": "Automotive CAN bus sensor ranges",
        "constraints": [
            {"lo": 0, "hi": 8000, "name": "engine_rpm"},
            {"lo": 0, "hi": 300, "name": "vehicle_speed_kmh"},
            {"lo": -40, "hi": 150, "name": "coolant_temp_c"},
            {"lo": 0, "hi": 100, "name": "throttle_pct"},
            {"lo": 0, "hi": 200, "name": "brake_pressure_bar"},
            {"lo": -720, "hi": 720, "name": "steering_angle_deg"},
            {"lo": 9, "hi": 16, "name": "battery_voltage_v"},
            {"lo": 0, "hi": 100, "name": "fuel_level_pct"},
        ],
    },
    "aviation": {
        "description": "Aviation ADS-B / flight data ranges",
        "constraints": [
            {"lo": -1000, "hi": 45000, "name": "altitude_ft"},
            {"lo": 0, "hi": 600, "name": "ground_speed_kt"},
            {"lo": -180, "hi": 180, "name": "heading_deg"},
            {"lo": -55, "hi": 70, "name": "cabin_temp_c"},
            {"lo": 75, "hi": 101, "name": "cabin_pressure_kpa"},
            {"lo": 0, "hi": 100, "name": "fuel_flow_pct"},
            {"lo": 60, "hi": 100, "name": "hydraulic_pct"},
            {"lo": -90, "hi": 90, "name": "pitch_deg"},
        ],
    },
    "medical": {
        "description": "Medical vital signs (FHIR-compatible)",
        "constraints": [
            {"lo": 36.1, "hi": 37.8, "name": "body_temp_c"},
            {"lo": 60, "hi": 100, "name": "heart_rate_bpm"},
            {"lo": 95, "hi": 100, "name": "spo2_pct"},
            {"lo": 80, "hi": 120, "name": "bp_systolic_mmhg"},
            {"lo": 60, "hi": 100, "name": "bp_diastolic_mmhg"},
            {"lo": 12, "hi": 20, "name": "respiratory_rate"},
            {"lo": 7.35, "hi": 7.45, "name": "ph"},
            {"lo": 0, "hi": 300, "name": "glucose_mg_dl"},
        ],
    },
    "energy": {
        "description": "Energy grid SCADA ranges",
        "constraints": [
            {"lo": 49.0, "hi": 51.0, "name": "grid_freq_hz"},
            {"lo": 0.9, "hi": 1.1, "name": "voltage_pu"},
            {"lo": 0, "hi": 80, "name": "transformer_temp_c"},
            {"lo": 0, "hi": 100, "name": "line_load_pct"},
            {"lo": 0, "hi": 500, "name": "current_a"},
            {"lo": -100, "hi": 100, "name": "power_factor_pct_offset"},
            {"lo": 0, "hi": 360, "name": "phase_angle_deg"},
            {"lo": 0, "hi": 50, "name": "thd_pct"},
        ],
    },
    "iot": {
        "description": "IoT MQTT environmental sensors",
        "constraints": [
            {"lo": -40, "hi": 85, "name": "ambient_temp_c"},
            {"lo": 0, "hi": 100, "name": "humidity_pct"},
            {"lo": 300, "hi": 1100, "name": "pressure_hpa"},
            {"lo": 0, "hi": 1000, "name": "co2_ppm"},
            {"lo": 0, "hi": 500, "name": "pm25_ug_m3"},
            {"lo": 0, "hi": 5000, "name": "light_lux"},
            {"lo": 0, "hi": 100, "name": "battery_pct"},
            {"lo": -120, "hi": -20, "name": "wifi_rssi_dbm"},
        ],
    },
    "financial": {
        "description": "Financial FIX protocol ranges",
        "constraints": [
            {"lo": 0.0001, "hi": 100000, "name": "price"},
            {"lo": 1, "hi": 10000000, "name": "volume"},
            {"lo": -100, "hi": 100, "name": "pct_change"},
            {"lo": 0.001, "hi": 1000, "name": "volatility"},
            {"lo": 0, "hi": 1, "name": "correlation"},
            {"lo": -100000, "hi": 100000, "name": "spread_bps"},
            {"lo": 0, "hi": 86400, "name": "time_offset_s"},
            {"lo": 0.01, "hi": 100, "name": "duration_years"},
        ],
    },
}


def list_presets() -> List[str]:
    """Return available preset names."""
    return list(PRESETS.keys())


def get_preset(name: str) -> List[Dict]:
    """Get constraint list for a preset."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {', '.join(PRESETS.keys())}")
    return PRESETS[name]["constraints"]


def get_preset_description(name: str) -> str:
    """Get description for a preset."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {', '.join(PRESETS.keys())}")
    return PRESETS[name]["description"]
