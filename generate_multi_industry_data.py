import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / "dataset"
OUT.mkdir(exist_ok=True)

np.random.seed(42)

DAYS = 100
INTERVALS = DAYS * 24 * 60
TIME = np.arange(INTERVALS + 1) / 1440.0
N = len(TIME)

INDUSTRY_PROFILES = {
    "brewery": {
        "display_name": "Brewery & Beverage Processing",
        "asm1_suitability": "High (biodegradable carbohydrates & ethanol; standard ASM1 kinetics apply)",
        "flow_mean": 18000.0, "flow_std": 1200.0, "flow_clip": (14000.0, 22000.0),
        "cod_mean": 500.0, "cod_std": 45.0, "cod_clip": (350.0, 650.0),
        "bod_cod_ratio": (0.48, 0.62),
        "tss_mean": 250.0, "tss_std": 25.0, "tss_clip": (150.0, 350.0),
        "ph_mean": 6.8, "ph_std": 0.2,
        "temp_mean": 25.0, "temp_std": 1.5,
        "fog_mean": 18.0, "tkn_mean": 30.0,
        "shock_every_days": 5, "shock_cod_add": (1200.0, 2500.0), "shock_tss_add": (100.0, 300.0),
        "shock_ph_choices": [3.5, 10.5, 11.5],
        "asm1_frac": {"SI": 0.05, "SS": 0.35, "XI": 0.10, "XS": 0.40, "XBH": 0.04, "XBA": 0.005, "XP": 0.055},
        "snh_base": 25.0,
        "eff_bod_base": 26.4,
    },
    "dairy": {
        "display_name": "Dairy & Milk Processing",
        "asm1_suitability": "High (lactose & milk proteins; requires FOG pre-skimming assumption)",
        "flow_mean": 12000.0, "flow_std": 1000.0, "flow_clip": (8500.0, 16000.0),
        "cod_mean": 1450.0, "cod_std": 160.0, "cod_clip": (900.0, 2100.0),
        "bod_cod_ratio": (0.55, 0.70),
        "tss_mean": 420.0, "tss_std": 55.0, "tss_clip": (250.0, 650.0),
        "ph_mean": 7.2, "ph_std": 0.35,
        "temp_mean": 28.0, "temp_std": 2.0,
        "fog_mean": 140.0, "tkn_mean": 55.0,
        "shock_every_days": 4, "shock_cod_add": (1500.0, 3200.0), "shock_tss_add": (180.0, 420.0),
        "shock_ph_choices": [3.2, 11.2, 12.0],
        "asm1_frac": {"SI": 0.04, "SS": 0.42, "XI": 0.08, "XS": 0.37, "XBH": 0.035, "XBA": 0.005, "XP": 0.05},
        "snh_base": 42.0,
        "eff_bod_base": 34.8,
    },
    "food_processing": {
        "display_name": "Food Processing (Cannery & Ready Meals)",
        "asm1_suitability": "High (starches, sugars, and biodegradable organics)",
        "flow_mean": 15000.0, "flow_std": 1800.0, "flow_clip": (10000.0, 21000.0),
        "cod_mean": 1100.0, "cod_std": 140.0, "cod_clip": (650.0, 1650.0),
        "bod_cod_ratio": (0.50, 0.65),
        "tss_mean": 480.0, "tss_std": 70.0, "tss_clip": (260.0, 750.0),
        "ph_mean": 6.5, "ph_std": 0.3,
        "temp_mean": 27.0, "temp_std": 2.2,
        "fog_mean": 95.0, "tkn_mean": 38.0,
        "shock_every_days": 6, "shock_cod_add": (1000.0, 2200.0), "shock_tss_add": (200.0, 450.0),
        "shock_ph_choices": [4.2, 9.8, 10.8],
        "asm1_frac": {"SI": 0.06, "SS": 0.32, "XI": 0.12, "XS": 0.41, "XBH": 0.035, "XBA": 0.005, "XP": 0.05},
        "snh_base": 30.0,
        "eff_bod_base": 31.2,
    },
    "agro_processing": {
        "display_name": "Agro-Processing (Starch, Grain & Produce)",
        "asm1_suitability": "Moderate-High (high particulate carbohydrates XS; seasonal load swings)",
        "flow_mean": 20000.0, "flow_std": 2200.0, "flow_clip": (13000.0, 27000.0),
        "cod_mean": 880.0, "cod_std": 120.0, "cod_clip": (520.0, 1350.0),
        "bod_cod_ratio": (0.45, 0.60),
        "tss_mean": 540.0, "tss_std": 85.0, "tss_clip": (300.0, 850.0),
        "ph_mean": 6.4, "ph_std": 0.25,
        "temp_mean": 24.0, "temp_std": 2.5,
        "fog_mean": 30.0, "tkn_mean": 34.0,
        "shock_every_days": 7, "shock_cod_add": (900.0, 1900.0), "shock_tss_add": (250.0, 550.0),
        "shock_ph_choices": [4.5, 5.2, 9.2],
        "asm1_frac": {"SI": 0.07, "SS": 0.26, "XI": 0.15, "XS": 0.43, "XBH": 0.035, "XBA": 0.005, "XP": 0.05},
        "snh_base": 28.0,
        "eff_bod_base": 29.5,
    },
    "pulp_paper": {
        "display_name": "Pulp & Paper Manufacturing",
        "asm1_suitability": "Limited / Stress-Test Only (high recalcitrant lignin SI/XI not degraded by ASM1)",
        "flow_mean": 26000.0, "flow_std": 1600.0, "flow_clip": (20000.0, 32000.0),
        "cod_mean": 1300.0, "cod_std": 130.0, "cod_clip": (850.0, 1800.0),
        "bod_cod_ratio": (0.25, 0.36),
        "tss_mean": 620.0, "tss_std": 80.0, "tss_clip": (380.0, 920.0),
        "ph_mean": 7.6, "ph_std": 0.35,
        "temp_mean": 33.0, "temp_std": 2.0,
        "fog_mean": 15.0, "tkn_mean": 16.0,
        "shock_every_days": 8, "shock_cod_add": (1400.0, 2800.0), "shock_tss_add": (300.0, 650.0),
        "shock_ph_choices": [4.0, 10.8, 11.8],
        "asm1_frac": {"SI": 0.24, "SS": 0.18, "XI": 0.26, "XS": 0.24, "XBH": 0.025, "XBA": 0.005, "XP": 0.05},
        "snh_base": 12.0,
        "eff_bod_base": 38.6,
    }
}


def compute_realistic_eff_bod(df: pd.DataFrame, base_eff: float, cod_mean: float) -> pd.Series:
    load_factor = (df["COD"] / cod_mean).clip(0.8, 1.4)
    shock_bump = df["shock_event"] * 8.5
    tss_carryover = 0.012 * (df["TSS"] - 250.0).clip(0, None)
    return (base_eff * load_factor + shock_bump + tss_carryover).clip(18.0, 65.0)


def build_industry_dataset(industry_key: str, cfg: dict) -> pd.DataFrame:
    diurnal = 1.0 + 0.08 * np.sin(2 * np.pi * TIME)
    flow = (np.random.normal(cfg["flow_mean"], cfg["flow_std"], N) * diurnal).clip(*cfg["flow_clip"])
    cod = (np.random.normal(cfg["cod_mean"], cfg["cod_std"], N) * diurnal).clip(*cfg["cod_clip"])
    tss = (np.random.normal(cfg["tss_mean"], cfg["tss_std"], N) * diurnal).clip(*cfg["tss_clip"])
    ph = np.random.normal(cfg["ph_mean"], cfg["ph_std"], N)
    temp = np.random.normal(cfg["temp_mean"], cfg["temp_std"], N)

    shock = np.zeros(N, dtype=bool)
    for day in range(2, DAYS, cfg["shock_every_days"]):
        start = day * 1440 + np.random.randint(0, 720)
        duration = np.random.randint(120, 361)
        end = min(start + duration, N)
        shock[start:end] = True

    cod[shock] += np.random.uniform(*cfg["shock_cod_add"], shock.sum())
    tss[shock] += np.random.uniform(*cfg["shock_tss_add"], shock.sum())
    ph[shock] = np.random.choice(cfg["shock_ph_choices"], size=shock.sum()) + np.random.normal(0, 0.25, shock.sum())
    ph = ph.clip(3.0, 12.5)

    bod = cod * np.random.uniform(*cfg["bod_cod_ratio"], N)
    current_do = np.random.normal(2.0, 0.15, N)
    fog = np.random.normal(cfg["fog_mean"], cfg["fog_mean"] * 0.15, N).clip(2.0, None)
    tkn = np.random.normal(cfg["tkn_mean"], cfg["tkn_mean"] * 0.12, N).clip(5.0, None)

    frac = cfg["asm1_frac"]
    bsm1_data = np.column_stack([
        TIME,
        cod * frac["SI"], cod * frac["SS"], cod * frac["XI"], cod * frac["XS"],
        cod * frac["XBH"], cod * frac["XBA"], cod * frac["XP"],
        np.full(N, 0.2), np.full(N, 1.0), np.full(N, cfg["snh_base"]),
        np.full(N, cfg["snh_base"] * 0.2), np.full(N, cfg["snh_base"] * 0.2),
        np.full(N, 7.0), tss, flow, temp, np.zeros((N, 5))
    ])
    bsm1_cols = [
        "time", "SI", "SS", "XI", "XS", "XBH", "XBA", "XP",
        "SO", "SNO", "SNH", "SND", "XND", "SALK",
        "TSS", "Q", "TEMP", "SD1", "SD2", "SD3", "XD4", "XD5"
    ]
    pd.DataFrame(bsm1_data, columns=bsm1_cols).to_csv(
        OUT / f"{industry_key}_bsm1_influent.csv", index=False
    )

    df = pd.DataFrame({
        "industry_type": industry_key,
        "data_provenance": "synthetic_industry_scenario",
        "time_days": TIME,
        "Q_in": flow,
        "COD": cod,
        "BOD": bod,
        "TSS": tss,
        "temperature": temp,
        "pH": ph,
        "current_DO": current_do,
        "FOG_mg_l": fog,
        "TKN_mg_l": tkn,
        "shock_event": shock.astype(int),
        "target_influent_bod5": bod,
        "target_DO": (2.0 + 0.00020 * (cod - cfg["cod_mean"]) + 0.0004 * (flow - cfg["flow_mean"]) / 100.0).clip(1.5, 4.0),
    })
    df["target_aeration"] = df["target_DO"] * (df["Q_in"] / 18000.0) * (df["BOD"] / 300.0).clip(0.6, 2.5)
    df["target_effluent_bod5_indicator"] = compute_realistic_eff_bod(df, cfg["eff_bod_base"], cfg["cod_mean"])

    df.to_csv(OUT / f"{industry_key}_dataset.csv", index=False)
    return df


if __name__ == "__main__":
    train_parts, test_parts = [], []
    metadata = {}

    for key, cfg in INDUSTRY_PROFILES.items():
        print(f"Generating {cfg['display_name']} ({key})...")
        if key == "brewery" and (OUT / "brewery_dataset.csv").exists():
            df = pd.read_csv(OUT / "brewery_dataset.csv")
            df.insert(0, "industry_type", "brewery")
            df.insert(1, "data_provenance", "synthetic_brewery_original")
            df["FOG_mg_l"] = 18.0
            df["TKN_mg_l"] = 30.0
            df["target_influent_bod5"] = df["BOD"]
            df["target_DO"] = (2.0 + 0.00025 * (df["COD"] - 500) + 0.0005 * (df["Q_in"] - 18000) / 100).clip(1.5, 4.0)
            df["target_aeration"] = df["target_DO"] * df["Q_in"] / 18000
            df["target_effluent_bod5_indicator"] = compute_realistic_eff_bod(df, cfg["eff_bod_base"], cfg["cod_mean"])
        else:
            df = build_industry_dataset(key, cfg)

        split = int(len(df) * 0.8)
        train_parts.append(df.iloc[:split])
        test_parts.append(df.iloc[split:])

        metadata[key] = {
            "display_name": cfg["display_name"],
            "asm1_suitability": cfg["asm1_suitability"],
            "rows": int(len(df)),
            "mean_flow_m3_day": float(df["Q_in"].mean()),
            "mean_cod_mg_l": float(df["COD"].mean()),
            "mean_bod_mg_l": float(df["BOD"].mean()),
            "mean_effluent_bod5_mg_l": float(df["target_effluent_bod5_indicator"].mean()),
            "mean_tss_mg_l": float(df["TSS"].mean()),
            "shock_rows": int(df["shock_event"].sum()),
            "provenance": "Synthetic scenario for multi-industry benchmarking; not measured plant data."
        }

    pd.concat(train_parts, ignore_index=True).to_csv(OUT / "multi_industry_ml_train.csv", index=False)
    pd.concat(test_parts, ignore_index=True).to_csv(OUT / "multi_industry_ml_test.csv", index=False)

    with open(OUT / "industry_profiles_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nMULTI-INDUSTRY DATASETS SYNCED SUCCESSFULLY:")
    for k, m in metadata.items():
        print(f" - {m['display_name']}: In_BOD={m['mean_bod_mg_l']:.1f} mg/L -> Eff_BOD5={m['mean_effluent_bod5_mg_l']:.1f} mg/L")