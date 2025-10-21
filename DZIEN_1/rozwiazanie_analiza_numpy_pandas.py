# -*- coding: utf-8 -*-
"""
Samodzielny skrypt: Analiza danych z użyciem NumPy i pandas (syntetyczny zbiór ride-share)
Autor: Marcin Albiniak
Opis:
- Generuje dane ~50k wierszy na rok 2024.
- Realizuje zadania 1–10: import, czyszczenie, transformacje NumPy, agregacje/groupby,
  resampling i rolling, merge z tabelą słownikową, pivot, walidacja, eksport wyników
  oraz krótkie wnioski.
Wymagania: Python 3.9+, numpy, pandas, matplotlib
Uruchomienie: python rozwiazanie_analiza_numpy_pandas.py
"""

from __future__ import annotations
import os, io, math, json
from datetime import datetime, timedelta
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- 0) Ustawienia ogólne ------------------------------
np.random.seed(42)
OUTPUT_DIR = "./wyniki"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------- 1) Generacja 'open-like' CSV ----------------------
cities = [
    ("Warszawa", "Mazowieckie", 1790658),
    ("Kraków", "Małopolskie", 803282),
    ("Gdańsk", "Pomorskie", 486022),
    ("Wrocław", "Dolnośląskie", 675079),
    ("Poznań", "Wielkopolskie", 546859),
    ("Lublin", "Lubelskie", 336339),
    ("Szczecin", "Zachodniopomorskie", 392379),
    ("Katowice", "Śląskie", 286960),
]
city_names = [c[0] for c in cities]
city_weights = np.array([c[2] for c in cities], dtype=float); city_weights /= city_weights.sum()

n_rows = 50_000
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31, 23, 59, 59)

def random_timestamp(size: int) -> np.ndarray:
    """Losuje znaczniki czasu w 2024 r., z pikami porannymi i popołudniowymi."""
    total_seconds = int((end_date - start_date).total_seconds())
    base = np.random.randint(0, total_seconds, size=size)
    ts = np.array([start_date + timedelta(seconds=int(b)) for b in base])
    # piki
    idx_morning = np.random.choice(size, size // 5, replace=False)
    idx_evening = np.random.choice(size, size // 5, replace=False)
    ts[idx_morning] = ts[idx_morning].astype("datetime64[h]").astype(datetime)
    ts[idx_evening] = ts[idx_evening].astype("datetime64[h]").astype(datetime)
    ts[idx_morning] = np.array([t.replace(hour=8, minute=np.random.randint(0, 60)) for t in ts[idx_morning]])
    ts[idx_evening] = np.array([t.replace(hour=17, minute=np.random.randint(0, 60)) for t in ts[idx_evening]])
    return ts

ride_id = np.arange(1, n_rows + 1)
ts = random_timestamp(n_rows)
city = np.random.choice(city_names, size=n_rows, p=city_weights)
driver_id = np.random.randint(1000, 5000, size=n_rows)

base_distance = np.clip(np.random.gamma(shape=2.0, scale=2.0, size=n_rows), 0.5, None)
hour = np.array([t.hour for t in ts])
traffic_factor = np.where((hour >= 7) & (hour <= 9), 1.3,
                   np.where((hour >= 16) & (hour <= 19), 1.25, 1.0))
duration_min = np.clip(base_distance * np.random.uniform(3.5, 5.0, size=n_rows) * traffic_factor, 5, None)
rain = np.random.binomial(1, 0.15, size=n_rows)
weekday = np.array([t.weekday() for t in ts])
weekend = ((weekday >= 5).astype(int))

city_base = {
    "Warszawa": 5.8, "Kraków": 5.2, "Gdańsk": 5.0, "Wrocław": 5.1,
    "Poznań": 5.0, "Lublin": 4.6, "Szczecin": 4.8, "Katowice": 4.9
}
base_fare_per_km = np.array([city_base[c] for c in city])
surge_mult = 1.0 + np.where(((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19)), 0.25, 0.0)
surge_mult += 0.2 * rain
surge_mult = np.round(surge_mult, 2)

distance_km = np.round(base_distance, 2)
base_fare_pln = np.round(distance_km * base_fare_per_km, 2)
fare_pln = np.round(base_fare_pln * surge_mult, 2)

df = pd.DataFrame({
    "ride_id": ride_id,
    "timestamp": ts,
    "city": city,
    "driver_id": driver_id,
    "distance_km": distance_km,
    "duration_min": np.round(duration_min, 1),
    "base_fare_pln": base_fare_pln,
    "surge_mult": surge_mult,
    "fare_pln": fare_pln,
    "rain": rain,
    "weekend": weekend
})
df_cities = pd.DataFrame(cities, columns=["city", "region", "population"])

RAW_CSV = os.path.join(OUTPUT_DIR, "rides_raw.csv")
df.to_csv(RAW_CSV, index=False)

# Podstawowe info
buf = io.StringIO(); df.info(buf=buf)
print(buf.getvalue())
print(df.select_dtypes(include=[np.number]).describe().T)

# ------------------------- 2) Czyszczenie i typy -----------------------------
def standardize_columns(columns: List[str]) -> List[str]:
    return [c.strip().lower().replace(" ", "_") for c in columns]

df.columns = standardize_columns(df.columns)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["duration_min"] = df["duration_min"].fillna(df["duration_min"].median())
df["fare_pln"] = df["fare_pln"].fillna(df["base_fare_pln"] * df["surge_mult"])

# ------------------------- 3) NumPy – transformacje --------------------------
dist_mean = df["distance_km"].mean()
dist_std = df["distance_km"].std(ddof=0) or 1.0
df["distance_z"] = (df["distance_km"] - dist_mean) / dist_std
df["fare_log1p"] = np.log1p(df["fare_pln"])
q = df["duration_min"].quantile([0.33, 0.66]).values
df["duration_bin"] = np.where(df["duration_min"] <= q[0], "short",
                       np.where(df["duration_min"] <= q[1], "medium", "long"))
problem_mask = (df["distance_km"] < 1.0) & (df["surge_mult"] >= 1.4) & (df["rain"] == 1)
df_problem = df.loc[problem_mask].copy()

# ------------------------- 4) Agregacje i grupowanie -------------------------
agg_city = (
    df.groupby("city")
      .agg(total_revenue_pln=("fare_pln", "sum"),
           trips=("ride_id", "count"),
           avg_km=("distance_km", "mean"))
      .sort_values("total_revenue_pln", ascending=False)
)
df["month"] = df["timestamp"].dt.to_period("M").astype(str)
agg_city_month = (
    df.groupby(["city", "month"])
      .agg(month_revenue_pln=("fare_pln", "sum"),
           trips=("ride_id", "count"))
      .reset_index()
)
df["fare_per_km"] = np.where(df["distance_km"] > 0, df["fare_pln"] / df["distance_km"], np.nan)
agg_fare_km = (
    df.groupby(["city", "weekend"])
      .agg(avg_fare_per_km=("fare_per_km", "mean"),
           trips=("ride_id", "count"))
      .reset_index()
      .sort_values(["city", "weekend"])
)

# ------------------------- 5) Czas: resampling i rolling ---------------------
df_time = df.set_index("timestamp").sort_index()
daily_rev = df_time["fare_pln"].resample("D").sum().to_frame("daily_revenue_pln")
daily_rev["daily_revenue_pln_ma7"] = daily_rev["daily_revenue_pln"].rolling(7, min_periods=1).mean()

monthly_city = (
    df_time
    .groupby("city")["fare_pln"]
    .resample("M")
    .sum()
    .rename("monthly_revenue_pln")
    .reset_index()
)

# ------------------------- 6) Merge z tabelą miast ---------------------------
df_merged = agg_city.reset_index().merge(
    df_cities, how="left", on="city", indicator=True
)
orphans = df_merged[df_merged["_merge"] != "both"]
print("Sieroty po merge:", len(orphans))

# ------------------------- 7) Pivot -----------------------------------------
agg_city_month_reg = agg_city_month.merge(df_cities, on="city", how="left")
pivot_region_month = pd.pivot_table(
    agg_city_month_reg,
    index="region",
    columns="month",
    values="month_revenue_pln",
    aggfunc="sum",
    fill_value=0.0,
)

# ------------------------- 8) Walidacja -------------------------------------
walidacja = {
    "brak_ujemnych_fare_pln": bool((df["fare_pln"] >= 0).all()),
    "liczba_unikalnych_miast_po_merge": int(df_merged["city"].nunique()),
    "brak_sierot_po_merge": bool(orphans.empty),
}
sum_global = float(df["fare_pln"].sum())
sum_monthly = float(agg_city_month["month_revenue_pln"].sum())
walidacja["suma_miesieczna_vs_globalna_ok"] = bool(abs(sum_global - sum_monthly) < 1e-6)
print("Walidacja:", walidacja)

# ------------------------- 9) Eksport wyników --------------------------------
AGG_CSV = os.path.join(OUTPUT_DIR, "wyniki_agregacje.csv")
PIVOT_CSV = os.path.join(OUTPUT_DIR, "wyniki_pivot.csv")
SAMPLE_CSV = os.path.join(OUTPUT_DIR, "sample_100.csv")
WNIOSEK_FILE = os.path.join(OUTPUT_DIR, "wnioski.txt")
PLOT_FILE = os.path.join(OUTPUT_DIR, "daily_revenue_plot.png")

agg_city.to_csv(AGG_CSV)
pivot_region_month.to_csv(PIVOT_CSV)
df.sample(100, random_state=42).to_csv(SAMPLE_CSV, index=False)

miasto_top = agg_city.reset_index().iloc[0]["city"]
region_top = (
    df_merged.groupby("region")["total_revenue_pln"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

WNIOSEK_TXT = f"""
Wnioski
1) Najwyższe skumulowane przychody generuje miasto: {miasto_top}.
2) Średnia stawka za kilometr rośnie w godzinach szczytu i podczas opadów (surge).
3) Największą łączną wartość miesięcznych przychodów w przekroju regionów uzyskuje: {region_top}.
4) Ograniczenie „przejazdów problematycznych” (krótkie + deszcz + wysoki surge) poprawi stabilność marży.
5) Dalsze kroki: powiązać dzienne przychody z danymi pogodowymi i kalendarzem (święta/eventy).
""".strip()
with open(WNIOSEK_FILE, "w", encoding="utf-8") as f:
    f.write(WNIOSEK_TXT)

# ------------------------- BONUS: wizualizacja -------------------------------
plt.figure(figsize=(10, 4))
plt.plot(daily_rev.index, daily_rev["daily_revenue_pln"], label="Przychód dzienny")
plt.plot(daily_rev.index, daily_rev["daily_revenue_pln_ma7"], label="MA7")
plt.title("Dzienne przychody z przejazdów (MA7)")
plt.xlabel("Data"); plt.ylabel("PLN"); plt.legend()
plt.tight_layout(); plt.savefig(PLOT_FILE, dpi=160); plt.close()

print("Zapisano pliki:", {
    "wyniki_agregacje.csv": AGG_CSV,
    "wyniki_pivot.csv": PIVOT_CSV,
    "sample_100.csv": SAMPLE_CSV,
    "rides_raw.csv": RAW_CSV,
    "wnioski.txt": WNIOSEK_FILE,
    "daily_revenue_plot.png": PLOT_FILE,
})
