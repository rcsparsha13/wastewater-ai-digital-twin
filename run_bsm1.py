
import os
import numpy as np
import pandas as pd
from bsm2_python import BSM1OL

INPUT = "dataset/brewery_bsm1_influent.csv"
OUTPUT = "dataset/bsm1_baseline_results.csv"

print("Loading brewery influent data...")

if not os.path.exists(INPUT):
    raise FileNotFoundError(f"Input dataset not found: {INPUT}")

df = pd.read_csv(INPUT)
data = df.to_numpy(dtype=float)

print("Input shape:", data.shape)

# Validate the BSM1 input format
if data.ndim != 2 or data.shape[1] != 22:
    raise ValueError(
        f"Expected 22 columns (time + 21 ASM1 components), "
        f"received {data.shape}"
    )

if not np.isfinite(data).all():
    raise ValueError("Input contains NaN or infinite values.")

time_diffs = np.diff(data[:, 0])

if not np.all(time_diffs > 0):
    raise ValueError("Input timestamps must be strictly increasing.")

print("First timestamp:", data[0, 0])
print("Last timestamp:", data[-1, 0])
print("Average timestep (minutes):", np.mean(time_diffs) * 1440)

print("\nStarting BSM1 open-loop baseline simulation...")

# Let BSM1 derive the timestep array from the input timestamps.
# This avoids the final-index mismatch caused by supplying
# a separate timestep value.
model = BSM1OL(
    data_in=data,
    timestep=None,
    endtime=None,
)

print("Simulation time points:", len(model.simtime))
print("Timestep entries:", len(model.timesteps))

if len(model.simtime) != len(model.timesteps):
    raise RuntimeError(
        "Simulation time and timestep arrays have different lengths."
    )

model.simulate(plot=False)

print("\nSimulation completed successfully.")

# Extract model-based performance metrics
performance = model.get_final_performance()

iqi, eqi, mixing, pumping, aeration = performance

total_energy = mixing + pumping + aeration

results = {
    "scenario": "BSM1_baseline",
    "simulation_start_day": float(data[0, 0]),
    "simulation_end_day": float(model.simtime[-1]),
    "evaluation_start_day": float(model.evaltime[0]),
    "evaluation_end_day": float(model.evaltime[1]),
    "influent_quality_index": float(iqi),
    "effluent_quality_index": float(eqi),
    "mixing_energy_kwh_day": float(mixing),
    "pumping_energy_kwh_day": float(pumping),
    "aeration_energy_kwh_day": float(aeration),
    "total_energy_kwh_day": float(total_energy),
}

os.makedirs("dataset", exist_ok=True)

pd.DataFrame([results]).to_csv(OUTPUT, index=False)

print("\n========== BSM1 BASELINE RESULTS ==========")

for key, value in results.items():
    print(f"{key}: {value}")

print("\nSaved results to:", OUTPUT)

