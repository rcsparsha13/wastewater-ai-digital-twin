import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).parent
DEVICE = "cpu"
WINDOW = 30

INDUSTRIES = ["brewery", "dairy", "food_processing", "agro_processing", "pulp_paper"]
BASE_FEATURES = [
    "Q_in", "COD", "BOD", "TSS",
    "temperature", "pH", "current_DO", "shock_event",
    "FOG_mg_l", "TKN_mg_l"
]
IND_COLS = [f"ind_{ind}" for ind in INDUSTRIES]
FEATURES = BASE_FEATURES + IND_COLS


class MultiIndustryCNNLSTM(nn.Module):
    def __init__(self, in_features: int = 15, out_targets: int = 3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_targets)
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.output(x[:, -1, :])


checkpoint = torch.load(
    BASE / "model/multi_industry_model.pth",
    map_location=DEVICE,
    weights_only=True
)

model = MultiIndustryCNNLSTM(len(FEATURES), 3)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

x_scaler = joblib.load(BASE / "model/multi_x_scaler.pkl")
y_scaler = joblib.load(BASE / "model/multi_y_scaler.pkl")


def predict(history: pd.DataFrame, industry_type: str = "brewery"):
    df = history.tail(WINDOW).copy()
    if len(df) < WINDOW:
        raise ValueError("Need at least 30 readings")

    if "FOG_mg_l" not in df.columns:
        df["FOG_mg_l"] = 18.0
    if "TKN_mg_l" not in df.columns:
        df["TKN_mg_l"] = 30.0

    for ind in INDUSTRIES:
        df[f"ind_{ind}"] = 1.0 if ind == industry_type else 0.0

    scaled = x_scaler.transform(df[FEATURES])
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor).numpy()

    result = y_scaler.inverse_transform(output)[0]

    return {
        "industry_type": industry_type,
        "predicted_do": float(result[0]),
        "predicted_aeration": float(result[1]),
        "predicted_effluent_bod5": float(result[2])
    }