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


def get_irrigation_plan(soil_input, temp_input, hum_input, crop_type="general",
                        terrain="flat", soil_texture="loam",
                        area_ha=None, water_available_m3=None, debit_lpm=None):
    """Return a structured irrigation plan (dict) — the single source of truth.

    Always includes the fuzzy sprinkling intensity and the terrain/soil method
    advice. Volume / time / feasibility keys are populated only when the relevant
    inputs (area, debit, available water) are supplied, otherwise they stay None.
    On a fuzzy-computation failure the dict has just {"error": "..."}.
    """
    try:
        result = _sprinkling_intensity(soil_input, temp_input, hum_input, crop_type)
    except Exception as e:
        return {"error": f"Error in fuzzy computation: {e}"}

    if result < 30:
        level, message = "low", "Low sprinkling level. Water lightly."
    elif result < 70:
        level, message = "medium", "Medium sprinkling level. Monitor and water moderately."
    else:
        level, message = "high", "High sprinkling level. Strong irrigation recommended."

    terrain = (terrain or "flat").lower()
    soil_texture = (soil_texture or "loam").lower()
    t = TERRAIN.get(terrain, TERRAIN["flat"])
    cap = TEXTURE_MAX_DEPTH_MM.get(soil_texture, 8) * t["runoff_factor"]
    depth_mm = round(result / 100.0 * MAX_DEPTH_MM, 1)
    passes = max(1, ceil(depth_mm / cap)) if depth_mm > 0 else 0

    plan = {
        "error": None,
        "sprinkling": result,
        "level": level,
        "message": message,
        "terrain": terrain,
        "methods": list(t["methods"]),
        "soil_texture": soil_texture,
        "cap_mm": round(cap, 1),
        "depth_mm": depth_mm,
        "passes": passes,
        "area_ha": area_ha,
        "debit_lpm": debit_lpm,
        "water_available_m3": water_available_m3,
        "volume_l": None,
        "volume_m3": None,
        "minutes": None,
        "minutes_per_pass": None,
        "shortfall_m3": None,
        "covered": None,
    }

    if area_ha:
        # 1 mm of water over 1 ha (10,000 m²) = 10,000 litres.
        volume_l = depth_mm * area_ha * 10000
        plan["volume_l"] = volume_l
        plan["volume_m3"] = volume_l / 1000
        if debit_lpm:
            plan["minutes"] = volume_l / debit_lpm
            plan["minutes_per_pass"] = plan["minutes"] / max(passes, 1)
        if water_available_m3 is not None:
            if plan["volume_m3"] > water_available_m3:
                plan["shortfall_m3"] = plan["volume_m3"] - water_available_m3
                plan["covered"] = False
            else:
                plan["covered"] = True

    return plan


def get_irrigation_recommendation(*args, **kwargs):
    """Backwards-compatible text rendering of get_irrigation_plan()."""
    plan = get_irrigation_plan(*args, **kwargs)
    if plan.get("error"):
        return plan["error"]

    lines = [f"Recommended sprinkling: {plan['sprinkling']}%.", "", plan["message"], ""]
    lines.append(f"🌍 Terrain: {plan['terrain']} → suitable methods: {', '.join(plan['methods'])}.")
    lines.append(f"🪨 Soil: {plan['soil_texture']} (≈{plan['cap_mm']:.1f} mm max per pass before runoff).")
    if plan["passes"] > 1:
        lines.append(f"⚠️ Split into {plan['passes']} passes to avoid runoff/erosion on {plan['terrain']} land.")

    if plan["volume_m3"] is not None:
        lines.append("")
        lines.append(f"💧 Water needed: ≈{plan['volume_m3']:.1f} m³ ({plan['volume_l']:,.0f} L) "
                     f"for {plan['area_ha']} ha at {plan['depth_mm']} mm.")
        if plan["minutes"] is not None:
            lines.append(f"⏱️ Irrigation time: ≈{plan['minutes']:.0f} min total at {plan['debit_lpm']} L/min "
                         f"(≈{plan['minutes_per_pass']:.0f} min/pass).")
        if plan["covered"] is False:
            lines.append(f"🚱 Shortfall: have {plan['water_available_m3']} m³, need {plan['volume_m3']:.1f} m³ "
                         f"(missing {plan['shortfall_m3']:.1f} m³). Reduce area or irrigate partially.")
        elif plan["covered"] is True:
            lines.append(f"✅ Water available ({plan['water_available_m3']} m³) covers the need.")

    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test: dry + hot + low humidity should call for strong irrigation, and the
    # practical plan must produce sane litres / time / feasibility numbers.
    plan = get_irrigation_plan(
        soil_input=10, temp_input=40, hum_input=20, crop_type="maize",
        terrain="steep", soil_texture="clay",
        area_ha=2.0, water_available_m3=5.0, debit_lpm=200,
    )
    out = get_irrigation_recommendation(
        soil_input=10, temp_input=40, hum_input=20, crop_type="maize",
        terrain="steep", soil_texture="clay",
        area_ha=2.0, water_available_m3=5.0, debit_lpm=200,
    )
    print(out)
    assert plan["sprinkling"] > 50, f"expected high intensity, got {plan['sprinkling']}"
    # 2 ha, steep clay -> volume = depth*2*10000 L; time = volume/debit.
    assert abs(plan["volume_m3"] - plan["depth_mm"] * 2.0 * 10) < 1e-6, "volume math off"
    assert abs(plan["minutes"] - plan["volume_l"] / 200) < 1e-6, "time math off"
    # 5 m³ available, need > 5 m³ -> shortfall reported, both in dict and text.
    assert plan["covered"] is False and plan["shortfall_m3"] > 0, "expected a shortfall"
    assert "Shortfall" in out, "expected a shortfall warning in text"
    # steep land -> drip only, no flood/furrow.
    assert "flood/furrow" not in plan["methods"], "steep land should not offer flood/furrow"
    assert "flood/furrow" not in out
    print("\nOK: irrigation self-check passed")
