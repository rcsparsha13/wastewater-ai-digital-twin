import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bsm2_python import BSM1OL
from bsm2_python.bsm2.init import reginit_bsm1 as reginit

BASE = Path(__file__).parent
DATA = BASE / "dataset"
MODEL_DIR = BASE / "model"

# ---------------- CONFIGURATION ----------------

WINDOW = 30
CONTROL_INTERVAL = 15

EVAL_START = 95.0
EVAL_END = 100.0
EVAL_DAYS = EVAL_END - EVAL_START

ELECTRICITY_TARIFF = 8.0  # INR per kWh

# Constrained KLa bounds: trims excess aeration during normal load
# while boosting capacity during CIP shock events
MIN_FACTOR = 0.60
MAX_FACTOR = 1.25

# Process & Effluent Constraints (Quality-Based Guardrails)
BOD_SAFETY_LIMIT = 10.0    # mg/L effluent BOD5 indicator guard (statutory limit ~20-30 mg/L)
NH4_SAFETY_LIMIT = 2.8     # mg/L ammonia guard (BSM1 statutory limit is 4.0 mg/L)

DEVICE = torch.device("cpu")

FEATURES = [
    "Q_in", "COD", "BOD", "TSS",
    "temperature", "pH", "current_DO", "shock_event"
]

TARGETS = ["target_DO", "target_aeration"]

# Preserved baseline result from run_bsm1.py (Do not overwrite)
BASELINE_ENERGY_KWH_DAY = 3969.5566666666655

# ---------------- CNN-LSTM ----------------

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(8, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.output(x[:, -1, :])


checkpoint = torch.load(
    MODEL_DIR / "aeration_model.pth",
    map_location=DEVICE,
    weights_only=True
)

model_ml = CNNLSTM().to(DEVICE)
model_ml.load_state_dict(checkpoint["model_state_dict"])
model_ml.eval()

x_scaler = joblib.load(MODEL_DIR / "x_scaler.pkl")
y_scaler = joblib.load(MODEL_DIR / "y_scaler.pkl")

# ---------------- ML ACCURACY ----------------

test_df = pd.read_csv(DATA / "ml_test.csv")

X_test = x_scaler.transform(test_df[FEATURES])
Y_test = y_scaler.transform(test_df[TARGETS])

test_windows = []
test_actual = []

for i in range(0, len(X_test) - WINDOW, 5):
    test_windows.append(X_test[i:i + WINDOW])
    test_actual.append(Y_test[i + WINDOW])

test_windows = np.asarray(test_windows, dtype=np.float32)
test_actual = np.asarray(test_actual, dtype=np.float32)

predictions = []
with torch.no_grad():
    for start in range(0, len(test_windows), 512):
        batch = torch.tensor(test_windows[start:start + 512], dtype=torch.float32).to(DEVICE)
        predictions.append(model_ml(batch).cpu().numpy())

test_predictions = np.concatenate(predictions)
actual_values = y_scaler.inverse_transform(test_actual)
predicted_values = y_scaler.inverse_transform(test_predictions)

ml_metrics = {}
for j, target in enumerate(TARGETS):
    ml_metrics[target] = {
        "MAE": float(mean_absolute_error(actual_values[:, j], predicted_values[:, j])),
        "RMSE": float(np.sqrt(mean_squared_error(actual_values[:, j], predicted_values[:, j]))),
        "R2": float(r2_score(actual_values[:, j], predicted_values[:, j]))
    }

# ---------------- LOAD DATA ----------------

sensor_df = pd.read_csv(DATA / "brewery_dataset.csv")
asm1_data = pd.read_csv(DATA / "brewery_bsm1_influent.csv").to_numpy(dtype=float)

if len(sensor_df) != len(asm1_data):
    raise ValueError("Sensor and ASM1 datasets have different row counts.")
if not np.isfinite(asm1_data).all():
    raise ValueError("ASM1 dataset contains missing or infinite values.")
if not all(col in sensor_df.columns for col in FEATURES):
    raise ValueError("Required CNN-LSTM input columns are missing.")

# ---------------- BATCH AI PREDICTIONS ----------------

scaled_sensor = x_scaler.transform(sensor_df[FEATURES])
control_indices = list(range(WINDOW - 1, len(sensor_df), CONTROL_INTERVAL))

control_windows = np.asarray([
    scaled_sensor[i - WINDOW + 1:i + 1]
    for i in control_indices
], dtype=np.float32)

control_predictions = []
with torch.no_grad():
    for start in range(0, len(control_windows), 512):
        batch = torch.tensor(control_windows[start:start + 512], dtype=torch.float32).to(DEVICE)
        control_predictions.append(model_ml(batch).cpu().numpy())

control_predictions = y_scaler.inverse_transform(np.concatenate(control_predictions))

prediction_lookup = {
    idx: control_predictions[k]
    for k, idx in enumerate(control_indices)
}

# ---------------- BSM1 INITIALIZATION ----------------

DEFAULT_KLA = np.array([
    reginit.KLA1,
    reginit.KLA2,
    reginit.KLA3,
    reginit.KLA4,
    reginit.KLA5
], dtype=float)

model = BSM1OL(
    data_in=asm1_data,
    timestep=None,
    endtime=None,
    evaltime=np.array([EVAL_START, EVAL_END])
)

current_kla = DEFAULT_KLA.copy()
last_factor = 1.0
last_pred_do = np.nan
last_pred_aeration = np.nan

kla_history = []
factor_history = []
pred_do_history = []
pred_aeration_history = []

# ---------------- CONSTRAINED AI-CONTROLLED SIMULATION ----------------

print("\nStarting Constrained AI-Controlled BSM1 simulation...")
print("Simulation duration:", asm1_data[-1, 0], "days")
print("Control interval:", CONTROL_INTERVAL, "minutes")

for i, current_time in enumerate(tqdm(model.simtime)):

    if i in prediction_lookup and i > 0:
        last_pred_do, last_pred_aeration = prediction_lookup[i]
        shock_active = int(sensor_df.iloc[i]["shock_event"]) == 1

        # 1. Feedforward load scaling from CNN-LSTM aeration demand
        # Normal demand (~2.0) trims excess BSM1 aeration to ~0.76x
        ff_factor = 0.76 * (last_pred_aeration / 2.0)

        # 2. Effluent-quality feedback guardrails (Ammonia & Biodegradable BOD5)
        eff_nh4 = float(model.ys_eff_all[i - 1, 9])
        eff_bod5 = 0.65 * float(model.ys_eff_all[i - 1, 1] + model.ys_eff_all[i - 1, 3])

        fb_factor = ff_factor
        if eff_nh4 > NH4_SAFETY_LIMIT or eff_bod5 > BOD_SAFETY_LIMIT:
            fb_factor = max(fb_factor, 1.05)

        if shock_active:
            fb_factor = max(fb_factor, 1.15)

        last_factor = float(np.clip(fb_factor, MIN_FACTOR, MAX_FACTOR))
        current_kla = DEFAULT_KLA.copy()
        current_kla[2:5] *= last_factor

    # Mutate model.klas in-place so BSM1OL.step() updates reactor1..5.kla
    model.klas[:] = current_kla
    model.step(i)

    kla_history.append(current_kla.copy())
    factor_history.append(last_factor)
    pred_do_history.append(last_pred_do)
    pred_aeration_history.append(last_pred_aeration)

    if i == 0:
        model.evaluator.add_new_data("iqi", "iqi")
        model.evaluator.add_new_data("eqi", "eqi")

    if EVAL_START <= current_time <= EVAL_END:
        model.evaluator.update_data("iqi", model.iqi_all[i], current_time)
        model.evaluator.update_data("eqi", model.eqi_all[i], current_time)

model.finish_evaluation(plot=False)

# ---------------- ENERGY (EXACT IWA BSM1 AERATION EQUATION) ----------------

iqi, eqi, mixing_base, pumping_base, _ = model.get_final_performance()
kla_history = np.asarray(kla_history)
time = model.simtime
mask = (time >= EVAL_START) & (time <= EVAL_END)

# Official BSM1 aeration energy: AE = (SO_sat / (1.8 * 1000)) * sum(V_k * KLa_k)
# Aerated reactor volumes V3 = V4 = V5 = 1333 m3, SO_sat = 8.0 mg/L
reactor_volumes = np.array([1000.0, 1000.0, 1333.0, 1333.0, 1333.0])
ae_timeseries_kwh_day = (8.0 / 1800.0) * (kla_history[mask] @ reactor_volumes)

mixing = float(mixing_base)
pumping = float(pumping_base)
aeration = float(ae_timeseries_kwh_day.mean())
total_energy = mixing + pumping + aeration

energy_saved_day = BASELINE_ENERGY_KWH_DAY - total_energy
energy_saved_percent = (energy_saved_day / BASELINE_ENERGY_KWH_DAY) * 100.0
energy_saved_period = energy_saved_day * EVAL_DAYS
money_saved_period = energy_saved_period * ELECTRICITY_TARIFF

# ---------------- EFFLUENT QUALITY ----------------

eff = model.ys_eff_all

# ASM1 indices: 0:SI, 1:SS, 2:XI, 3:XS, 4:XBH, 5:XBA, 6:XP, 7:SO, 8:SNO, 9:SNH
organic_indices = [0, 1, 2, 3, 4, 5, 6]

influent_organic_cod = asm1_data[:, 1:][:, organic_indices].sum(axis=1)
effluent_organic_cod = eff[:, organic_indices].sum(axis=1)

# Physics-grounded ASM1 Biodegradable BOD5 Indicator:
# Derived from readily (SS, idx 1) + slowly (XS, idx 3) biodegradable COD (f_BOD = 0.65)
influent_asm1_bod5 = 0.65 * (asm1_data[:, 2] + asm1_data[:, 4])
effluent_asm1_bod5 = 0.65 * (eff[:, 1] + eff[:, 3])

influent_tss = asm1_data[:, 14]
effluent_tss = eff[:, 13]

influent_flow = asm1_data[:, 15]
effluent_flow = eff[:, 14]

bod_removal = ((influent_asm1_bod5 - effluent_asm1_bod5) / np.maximum(influent_asm1_bod5, 1e-9)) * 100.0
cod_removal = ((influent_organic_cod - effluent_organic_cod) / np.maximum(influent_organic_cod, 1e-9)) * 100.0
tss_removal = ((influent_tss - effluent_tss) / np.maximum(influent_tss, 1e-9)) * 100.0

bod_load_removed = (influent_asm1_bod5 * influent_flow - effluent_asm1_bod5 * effluent_flow) / 1000.0
cod_load_removed = (influent_organic_cod * influent_flow - effluent_organic_cod * effluent_flow) / 1000.0
tss_load_removed = (influent_tss * influent_flow - effluent_tss * effluent_flow) / 1000.0

# ---------------- TIMESERIES EXPORT ----------------

timeseries = pd.DataFrame({
    "time_days": time,
    "predicted_do": pred_do_history,
    "predicted_aeration": pred_aeration_history,
    "control_factor": factor_history,
    "KLa_R1": kla_history[:, 0],
    "KLa_R2": kla_history[:, 1],
    "KLa_R3": kla_history[:, 2],
    "KLa_R4": kla_history[:, 3],
    "KLa_R5": kla_history[:, 4],
    "influent_asm1_bod5_indicator_mg_l": influent_asm1_bod5,
    "effluent_asm1_bod5_indicator_mg_l": effluent_asm1_bod5,
    "influent_asm1_cod_indicator_mg_l": influent_organic_cod,
    "effluent_asm1_cod_indicator_mg_l": effluent_organic_cod,
    "influent_tss_mg_l": influent_tss,
    "effluent_tss_mg_l": effluent_tss,
    "influent_flow_m3_day": influent_flow,
    "effluent_flow_m3_day": effluent_flow,
    "bod_removal_percent": bod_removal,
    "cod_removal_percent": cod_removal,
    "tss_removal_percent": tss_removal,
    "bod_load_removed_kg_day": bod_load_removed,
    "cod_load_removed_kg_day": cod_load_removed,
    "tss_load_removed_kg_day": tss_load_removed,
    "effluent_do_mg_l": eff[:, 7],
    "effluent_ammonia_mg_l": eff[:, 9]
})

timeseries.to_csv(DATA / "ai_bsm1_timeseries.csv", index=False)
eval_df = timeseries.loc[mask]

# ---------------- SUMMARY ----------------

summary = {
    "scenario": "Constrained_AI_Controlled_BSM1",
    "industry_type": "brewery",
    "evaluation_start_day": EVAL_START,
    "evaluation_end_day": EVAL_END,

    "influent_bod_mg_l": float(sensor_df.loc[sensor_df["time_days"].between(EVAL_START, EVAL_END), "BOD"].mean()),
    "influent_cod_mg_l": float(sensor_df.loc[sensor_df["time_days"].between(EVAL_START, EVAL_END), "COD"].mean()),
    "influent_tss_mg_l": float(eval_df["influent_tss_mg_l"].mean()),

    "effluent_bod5_indicator_mg_l": float(eval_df["effluent_asm1_bod5_indicator_mg_l"].mean()),
    "effluent_bod5_status": "ASM1-Derived Biodegradable BOD5 Indicator (0.65 * [SS + XS]); not a 5-day lab assay.",
    "average_bod_removal_percent": float(eval_df["bod_removal_percent"].mean()),
    "average_bod_load_removed_kg_day": float(eval_df["bod_load_removed_kg_day"].mean()),

    "effluent_asm1_cod_indicator_mg_l": float(eval_df["effluent_asm1_cod_indicator_mg_l"].mean()),
    "effluent_tss_mg_l": float(eval_df["effluent_tss_mg_l"].mean()),
    "effluent_do_mg_l": float(eval_df["effluent_do_mg_l"].mean()),
    "effluent_ammonia_mg_l": float(eval_df["effluent_ammonia_mg_l"].mean()),

    "average_cod_removal_percent": float(eval_df["cod_removal_percent"].mean()),
    "average_tss_removal_percent": float(eval_df["tss_removal_percent"].mean()),
    "average_cod_load_removed_kg_day": float(eval_df["cod_load_removed_kg_day"].mean()),
    "average_tss_load_removed_kg_day": float(eval_df["tss_load_removed_kg_day"].mean()),

    "influent_quality_index": float(iqi),
    "effluent_quality_index": float(eqi),

    "mixing_energy_kwh_day": float(mixing),
    "pumping_energy_kwh_day": float(pumping),
    "aeration_energy_kwh_day": float(aeration),
    "total_energy_kwh_day": float(total_energy),
    "baseline_energy_kwh_day": float(BASELINE_ENERGY_KWH_DAY),

    "energy_saved_kwh_day": float(energy_saved_day),
    "energy_saving_percent": float(energy_saved_percent),
    "energy_saved_kwh_evaluation_period": float(energy_saved_period),
    "electricity_tariff_inr_kwh": ELECTRICITY_TARIFF,
    "money_saved_inr_evaluation_period": float(money_saved_period),

    "ml_accuracy": ml_metrics,
    "average_control_factor": float(eval_df["control_factor"].mean()),
    "prediction_count": len(control_predictions),
    "data_note": "Constrained feedforward+feedback KLa controller on synthetic brewery influent."
}

with open(DATA / "ai_bsm1_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

pd.DataFrame([{k: v for k, v in summary.items() if k != "ml_accuracy"}]).to_csv(
    DATA / "ai_bsm1_summary.csv", index=False
)

with open(DATA / "ai_ml_accuracy.json", "w") as f:
    json.dump(ml_metrics, f, indent=4)

print("\n========== CONSTRAINED AI BSM1 RESULTS ==========")
for key, value in summary.items():
    if key != "ml_accuracy":
        print(f"{key}: {value}")