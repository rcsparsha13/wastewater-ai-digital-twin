import sys
import time
import pandas as pd
import psycopg2
from predict import predict, INDUSTRIES

DB = {
    "host": "localhost",
    "database": "brewery_ai",
    "user": "brewery_user",
    "password": "brewery123"
}

industry = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in INDUSTRIES else "brewery"
csv_path = f"dataset/{industry}_dataset.csv"

df = pd.read_csv(csv_path).iloc[-300:]

conn = psycopg2.connect(**DB)
cur = conn.cursor()

history = []
print(f"MULTI-INDUSTRY LIVE DATA STREAM STARTED [{industry.upper()}]")

try:
    for _, row in df.iterrows():
        history.append(row)
        fog = float(row["FOG_mg_l"]) if "FOG_mg_l" in row else 18.0
        tkn = float(row["TKN_mg_l"]) if "TKN_mg_l" in row else 30.0

        cur.execute(
            """
            INSERT INTO live_sensors
            (industry_type, time_days, q_in, cod, bod, tss,
             temperature, ph, current_do, fog_mg_l, tkn_mg_l, shock_event)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                industry,
                float(row["time_days"]),
                float(row["Q_in"]),
                float(row["COD"]),
                float(row["BOD"]),
                float(row["TSS"]),
                float(row["temperature"]),
                float(row["pH"]),
                float(row["current_DO"]),
                fog,
                tkn,
                int(row["shock_event"])
            )
        )

        if len(history) >= 30:
            recent_history = pd.DataFrame(history[-30:])
            result = predict(recent_history, industry_type=industry)

            baseline = float(row["current_DO"]) * float(row["Q_in"]) / 18000.0
            ai_energy = float(result["predicted_aeration"]) * 0.78 * float(row["Q_in"]) / 18000.0
            saved = baseline - ai_energy
            cost = saved * 8.0

            cur.execute(
                """
                INSERT INTO ai_predictions
                (industry_type, predicted_do, predicted_aeration, predicted_effluent_bod5,
                 baseline_energy, ai_energy, energy_saved, cost_saved)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    industry,
                    float(result["predicted_do"]),
                    float(result["predicted_aeration"]),
                    float(result["predicted_effluent_bod5"]),
                    float(baseline),
                    float(ai_energy),
                    float(saved),
                    float(cost)
                )
            )

            print(
                f"[{industry}] DO: {result['predicted_do']:.2f} mg/L | "
                f"Eff BOD5: {result['predicted_effluent_bod5']:.2f} mg/L | "
                f"Shock: {int(row['shock_event'])}"
            )

        conn.commit()
        time.sleep(1)

except KeyboardInterrupt:
    print("Stream stopped.")
finally:
    cur.close()
    conn.close()