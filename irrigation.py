from functools import lru_cache
from math import ceil

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Category map for flexible classification
CROP_CATEGORIES = {
    "maize": "grains",
    "wheat": "grains",
    "tomato": "vegetables",
    "spinach": "vegetables",
    "potato": "roots",
    "cassava": "roots",
    "banana": "fruits",
    "mango": "fruits"
}

# Terrain (slope class) -> runoff factor + irrigation methods that stay viable.
# Steeper land sheds water faster, so usable depth per pass drops and flood/furrow
# is no longer safe (runoff + erosion).
TERRAIN = {
    "flat":   {"runoff_factor": 1.0,  "methods": ["flood/furrow", "sprinkler", "drip"]},
    "gentle": {"runoff_factor": 0.85, "methods": ["sprinkler", "drip"]},
    "steep":  {"runoff_factor": 0.70, "methods": ["drip"]},
}

# Soil texture -> comfortable application depth per pass (mm) before water is lost
# to runoff (clay) or deep percolation (sand). Calibrate to local field tests.
TEXTURE_MAX_DEPTH_MM = {"sandy": 12, "loam": 8, "clay": 6}

# Intensity (% sprinkling from the fuzzy model) maps linearly onto an application
# depth, 0-100% -> 0-MAX_DEPTH_MM of water to put on the field this session.
MAX_DEPTH_MM = 10.0


@lru_cache(maxsize=None)
def _build_control_system(crop_category):
    """Build (once per crop category) the fuzzy control system.

    Cached because defining the antecedents, membership functions and rules is the
    expensive part; a fresh ControlSystemSimulation per call is cheap.
    """
    soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil_moisture')
    temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
    sprinkling = ctrl.Consequent(np.arange(0, 101, 1), 'sprinkling')

    # Temperature and sprinkling (same for all)
    temperature['cold'] = fuzz.trimf(temperature.universe, [0, 10, 20])
    temperature['warm'] = fuzz.trimf(temperature.universe, [15, 25, 35])
    temperature['hot'] = fuzz.trimf(temperature.universe, [30, 40, 50])

    sprinkling['low'] = fuzz.trimf(sprinkling.universe, [0, 25, 50])
    sprinkling['medium'] = fuzz.trimf(sprinkling.universe, [30, 50, 70])
    sprinkling['high'] = fuzz.trimf(sprinkling.universe, [60, 80, 100])

    # Category-specific membership functions
    if crop_category == "grains":
        soil_moisture.automf(names=['low', 'medium', 'high'])
        humidity.automf(names=['low', 'medium', 'high'])

    elif crop_category == "vegetables":
        soil_moisture['low'] = fuzz.trimf(soil_moisture.universe, [0, 15, 35])
        soil_moisture['medium'] = fuzz.trimf(soil_moisture.universe, [30, 50, 70])
        soil_moisture['high'] = fuzz.trimf(soil_moisture.universe, [60, 85, 100])

        humidity['low'] = fuzz.trimf(humidity.universe, [0, 20, 40])
        humidity['medium'] = fuzz.trimf(humidity.universe, [35, 55, 75])
        humidity['high'] = fuzz.trimf(humidity.universe, [70, 90, 100])

    elif crop_category == "roots":
        soil_moisture['low'] = fuzz.trimf(soil_moisture.universe, [0, 20, 40])
        soil_moisture['medium'] = fuzz.trimf(soil_moisture.universe, [35, 55, 75])
        soil_moisture['high'] = fuzz.trimf(soil_moisture.universe, [70, 90, 100])

        humidity['low'] = fuzz.trimf(humidity.universe, [0, 25, 50])
        humidity['medium'] = fuzz.trimf(humidity.universe, [45, 60, 75])
        humidity['high'] = fuzz.trimf(humidity.universe, [70, 85, 100])

    elif crop_category == "fruits":
        soil_moisture['low'] = fuzz.trimf(soil_moisture.universe, [0, 20, 45])
        soil_moisture['medium'] = fuzz.trimf(soil_moisture.universe, [40, 60, 80])
        soil_moisture['high'] = fuzz.trimf(soil_moisture.universe, [75, 90, 100])

        humidity['low'] = fuzz.trimf(humidity.universe, [0, 30, 50])
        humidity['medium'] = fuzz.trimf(humidity.universe, [45, 65, 85])
        humidity['high'] = fuzz.trimf(humidity.universe, [80, 90, 100])

    else:  # general/default
        soil_moisture.automf(names=['low', 'medium', 'high'])
        humidity.automf(names=['low', 'medium', 'high'])

    # Define fuzzy rules
    rules = [
        ctrl.Rule(soil_moisture['low'] & temperature['hot'] & humidity['low'], sprinkling['high']),
        ctrl.Rule(soil_moisture['low'] & temperature['warm'] & humidity['medium'], sprinkling['medium']),
        ctrl.Rule(soil_moisture['medium'] & temperature['warm'] & humidity['medium'], sprinkling['medium']),
        ctrl.Rule(soil_moisture['medium'] & temperature['hot'], sprinkling['high']),
        ctrl.Rule(soil_moisture['high'] | humidity['high'], sprinkling['low']),
        ctrl.Rule(temperature['cold'] & soil_moisture['medium'], sprinkling['low']),
        ctrl.Rule(soil_moisture['medium'] & humidity['low'], sprinkling['medium']),
        ctrl.Rule(temperature['hot'] & humidity['high'], sprinkling['medium']),
        ctrl.Rule(temperature['cold'] & humidity['low'], sprinkling['low']),
    ]

    return ctrl.ControlSystem(rules)


def _sprinkling_intensity(soil_input, temp_input, hum_input, crop_type):
    """Run the fuzzy model and return the sprinkling intensity (0-100%)."""
    crop_category = CROP_CATEGORIES.get(crop_type.lower(), "general")
    sim = ctrl.ControlSystemSimulation(_build_control_system(crop_category))
    sim.input['soil_moisture'] = soil_input
    sim.input['temperature'] = temp_input
    sim.input['humidity'] = hum_input
    sim.compute()
    return round(sim.output['sprinkling'], 2)


def get_irrigation_recommendation(soil_input, temp_input, hum_input, crop_type="general",
                                  terrain="flat", soil_texture="loam",
                                  area_ha=None, water_available_m3=None, debit_lpm=None):
    """Recommend irrigation.

    Always returns the fuzzy sprinkling intensity. When terrain/soil/area/water are
    supplied, it also turns that intensity into a concrete plan: method choice,
    number of passes, litres of water needed and irrigation time at the given flow
    rate (debit), plus a feasibility check against the water available.
    """
    try:
        result = _sprinkling_intensity(soil_input, temp_input, hum_input, crop_type)
    except Exception as e:
        return f"Error in fuzzy computation: {e}"

    if result < 30:
        msg = "Low sprinkling level. Water lightly."
    elif result < 70:
        msg = "Medium sprinkling level. Monitor and water moderately."
    else:
        msg = "High sprinkling level. Strong irrigation recommended."

    lines = [f"Recommended sprinkling: {result}%.", "", msg]

    terrain = (terrain or "flat").lower()
    soil_texture = (soil_texture or "loam").lower()
    t = TERRAIN.get(terrain, TERRAIN["flat"])
    cap = TEXTURE_MAX_DEPTH_MM.get(soil_texture, 8) * t["runoff_factor"]

    demand_depth_mm = round(result / 100.0 * MAX_DEPTH_MM, 1)
    passes = max(1, ceil(demand_depth_mm / cap)) if demand_depth_mm > 0 else 0

    lines.append("")
    lines.append(f"🌍 Terrain: {terrain} → suitable methods: {', '.join(t['methods'])}.")
    lines.append(f"🪨 Soil: {soil_texture} (≈{cap:.1f} mm max per pass before runoff).")
    if passes > 1:
        lines.append(f"⚠️ Split into {passes} passes to avoid runoff/erosion on {terrain} land.")

    if area_ha:
        # 1 mm of water over 1 ha (10,000 m²) = 10,000 litres.
        volume_l = demand_depth_mm * area_ha * 10000
        volume_m3 = volume_l / 1000
        lines.append("")
        lines.append(f"💧 Water needed: ≈{volume_m3:.1f} m³ ({volume_l:,.0f} L) "
                     f"for {area_ha} ha at {demand_depth_mm} mm.")
        if debit_lpm:
            minutes = volume_l / debit_lpm
            per_pass = minutes / max(passes, 1)
            lines.append(f"⏱️ Irrigation time: ≈{minutes:.0f} min total at {debit_lpm} L/min "
                         f"(≈{per_pass:.0f} min/pass).")
        if water_available_m3 is not None:
            if volume_m3 > water_available_m3:
                short = volume_m3 - water_available_m3
                lines.append(f"🚱 Shortfall: have {water_available_m3} m³, need {volume_m3:.1f} m³ "
                             f"(missing {short:.1f} m³). Reduce area or irrigate partially.")
            else:
                lines.append(f"✅ Water available ({water_available_m3} m³) covers the need.")

    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test: dry + hot + low humidity should call for strong irrigation, and the
    # practical plan must produce sane litres / time / feasibility numbers.
    out = get_irrigation_recommendation(
        soil_input=10, temp_input=40, hum_input=20, crop_type="maize",
        terrain="steep", soil_texture="clay",
        area_ha=2.0, water_available_m3=5.0, debit_lpm=200,
    )
    print(out)
    intensity = _sprinkling_intensity(10, 40, 20, "maize")
    assert intensity > 50, f"expected high intensity, got {intensity}"
    # 2 ha, steep clay -> demand depth ~ intensity/100*10 mm; volume = depth*2*10000 L.
    depth = round(intensity / 100.0 * MAX_DEPTH_MM, 1)
    expected_m3 = depth * 2.0 * 10000 / 1000
    assert f"{expected_m3:.1f} m³" in out, f"volume math off: expected {expected_m3:.1f} m³"
    # 5 m³ available, need > 5 m³ -> shortfall must be reported.
    assert "Shortfall" in out, "expected a shortfall warning"
    # steep land -> drip only, no flood/furrow.
    assert "flood/furrow" not in out, "steep land should not offer flood/furrow"
    print("\nOK: irrigation self-check passed")
