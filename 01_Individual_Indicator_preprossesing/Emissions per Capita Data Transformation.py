# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

INPUT_FILE = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/05_Scripts/01_Indicators/01.1_ Emissions per GDP/EDGARv8.0_total_GHG_GWP100_AR5_NUTS2_1990_2022.xlsx"
SHEET_SRC  = "GHG by NUTS2 and Sector"

# Ordnername laut deiner Angabe ("ouput" ohne t)
OUTPUT_DIR = Path(r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/05_Scripts/01_Indicators/01.1_ Emissions per GDP/ouput")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "EDGARv8.0_GHG_relevant_sectors_by_NUTS2_2019.xlsx"

RELEVANT_SECTORS = {"Agriculture", "Energy", "Industry", "Transport", "Waste"}
USECOLS = "B:F,AJ"  # B,C,D,E,F,AJ
SHEET_NAME_OUT = "GHG relevant Sectors NUTS2 2019"  # <= 31 Zeichen

# --- Einlesen ---
df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_SRC,
    usecols=USECOLS,
    header=None,
    skiprows=6,  # Daten ab Zeile 7
    engine="openpyxl"
)

df.columns = ["CountryCode","CountryName","NUTS2","NUTS2_Level","Sector","Emissions_AJ"]

# Säubern / Typen
df["Sector"] = df["Sector"].astype(str).str.strip()
for c in ["CountryCode","CountryName","NUTS2","NUTS2_Level"]:
    df[c] = df[c].astype(str).str.strip()
df["Emissions_AJ"] = pd.to_numeric(df["Emissions_AJ"], errors="coerce").fillna(0)
# --- Permanenter Namens-Fix in der Quelle (wir überschreiben NUTS2_Level) ---
permanent_fixes = {
    "Région de Bruxelles-Capitale/ Brussels Hoofdstedelijk Gewest": "Région de Bruxelles-Capitale/Brussels Hoofdstedelijk Gewest",
}
df["NUTS2_Level"] = df["NUTS2_Level"].replace(permanent_fixes)


# Relevante Sektoren filtern
rel = {s.lower() for s in RELEVANT_SECTORS}
df_rel = df[df["Sector"].str.lower().isin(rel)].copy()
if df_rel.empty:
    raise ValueError("Keine Zeilen mit den relevanten Sektoren gefunden.")

# --- Eine Zeile pro Region: Summe über die 5 Sektoren ---
keys = ["CountryCode","CountryName","NUTS2","NUTS2_Level"]
df_out = (df_rel.groupby(keys, as_index=False)["Emissions_AJ"]
          .sum()
          .rename(columns={"Emissions_AJ": "GHG_2019_sum_Agri+Industry+Energy+Industry+Transport+Waste"}))

# Sortierung für Lesbarkeit
df_out = df_out.sort_values(by=["CountryCode","NUTS2"], kind="stable")

# --- Zusätzliche Summe ohne den Sektor 'Transport' ---
# (neue Spalte: "GHG_2019_sum_Agri+Industry+Energy+Industry+Waste")
col_no_transport = "GHG_2019_sum_Agri+Industry+Energy+Industry+Waste"

df_rel_no_transport = df_rel[df_rel["Sector"].str.lower() != "transport"].copy()

sums_no_transport = (
    df_rel_no_transport
    .groupby(keys, as_index=False)["Emissions_AJ"]
    .sum()
    .rename(columns={"Emissions_AJ": col_no_transport})
)

# An df_out anhängen; fehlende Werte = 0 (falls Region nur Transport hatte)
df_out = df_out.merge(sums_no_transport, on=keys, how="left")
df_out[col_no_transport] = df_out[col_no_transport].fillna(0)

# --- EUROSTAT: Merge über Regionsnamen (NUTS2_Level) ---
# Liest A:B ab Zeile 10, extrahiert den Regionsnamen, normalisiert Namen (Diakritika, Zeichen, Case)
# und merged auf df_out["NUTS2_Level"].

import re
import unicodedata

EUROSTAT_GDP_FILE = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/05_Scripts/01_Indicators/01.1_ Emissions per GDP/nama_10r_2gdp__custom_18097128_spreadsheet.xlsx"

def _strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))

def _norm_name(s: str) -> str:
    s = (s or "").replace("\u00A0", " ").replace("\u202f", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- Namens-Fixes in df_out (vor Eurostat-Read) ---
# Harmonisiert Schreibweisen in NUTS2_Level, damit der Merge auf Namen greift.
name_fixes = {
    "Région de Bruxelles-Capitale/ Brussels Hoofdstedelijk Gewest": "Région de Bruxelles-Capitale/Brussels Hoofdstedelijk Gewest",
    "Prov. Brabant Wallon": "Prov. Brabant wallon",
    "Ile-de-France": "Ile de France",
    "Aland": "Åland",
}

# 1) Eurostat: A (Regionen), B (Wert) laden
gdp = pd.read_excel(
    EUROSTAT_GDP_FILE,
    sheet_name="Sheet 1",
    usecols="A:B",
    header=None,
    skiprows=9,    # ab Zeile 10
    dtype=str,
    engine="openpyxl"
)
gdp.columns = ["RegionRaw", "GDP_B_raw"]

# 2) Namen normalisieren + Zahlen parsen (Komma -> Punkt, Whitespaces entfernen)
gdp["RegionName"] = gdp["RegionRaw"].astype(str)
gdp["RegionKey"]  = gdp["RegionName"].apply(_norm_name)

gdp["GDP_B"] = (
    gdp["GDP_B_raw"].astype(str)
      .str.replace("\u202f", "", regex=False)
      .str.replace("\u00A0", "", regex=False)
      .str.replace(" ", "", regex=False)
      .str.replace(",", ".", regex=False)
)
gdp["GDP_B"] = pd.to_numeric(gdp["GDP_B"], errors="coerce")

# Zeilen ohne RegionKey verwerfen
gdp = gdp.dropna(subset=["RegionKey"])

# --- Name-Fixes NUR für den Match (Original in NUTS2_Level bleibt unverändert) ---
name_fixes = {
    "Région de Bruxelles-Capitale/ Brussels Hoofdstedelijk Gewest": "Région de Bruxelles-Capitale/Brussels Hoofdstedelijk Gewest",
    "Prov. Brabant Wallon": "Prov. Brabant wallon",
    "Ile-de-France": "Ile de France",
    "Aland": "Åland",
}

# Linke Seite: RegionKey aus NUTS2_Level bauen (mit Mapping), aber NUTS2_Level selbst NICHT überschreiben
df_out["RegionKey"] = (
    df_out["NUTS2_Level"]
        .astype(str)
        .replace(name_fixes)   # nur fürs Matching umschreiben
        .apply(_norm_name)     # gleiche Normalisierung wie rechts (gdp)
)

# Merge auf den Schlüssel und anschließend aufräumen
df_out = df_out.merge(gdp[["RegionKey", "GDP_B"]], on="RegionKey", how="left")
df_out = df_out.drop(columns=["RegionKey"])


# Optional: schönerer Spaltenname
# df_out = df_out.rename(columns={"GDP_B": "EUROSTAT_B_Value"})

# Debug: Match-Quote ausgeben
matched = df_out["GDP_B"].notna().sum()
total   = len(df_out)
print(f"EUROSTAT name-merge: matched {matched}/{total} Regionen.")
if matched < total:
    # erste Beispiele zeigen, die nicht gematcht wurden
    missing_left = df_out.loc[df_out["GDP_B"].isna(), "NUTS2_Level"].head(10).tolist()
    print("Beispiele ohne Match (NUTS2_Level):", missing_left)

    # --- GHG/BIP (Intensitäten) berechnen ---
import numpy as np

# 1) Spaltennamen robust identifizieren
possible_inclusive = [
    "GHG_2019_sum_Agri+Industry+Energy+Industry+Transport+Waste",
    "GHG_2019_sum_relevant_sectors"  # falls du den generischen Namen genommen hattest
]
incl_col = next((c for c in possible_inclusive if c in df_out.columns), None)
if incl_col is None:
    raise KeyError("Spalte mit GHG-Summe inkl. Transport nicht gefunden. "
                   "Erwarte z.B. 'GHG_2019_sum_Agri+Industry+Energy+Industry+Transport+Waste'.")

no_transport_col = "GHG_2019_sum_Agri+Industry+Energy+Industry+Waste"
if no_transport_col not in df_out.columns:
    raise KeyError("Spalte mit GHG-Summe ohne Transport fehlt: "
                   "'GHG_2019_sum_Agri+Industry+Energy+Industry+Waste'.")

possible_gdp = ["GDP_B", "EUROSTAT_B_Value", "GDP_Eurostat_colB"]
gdp_col = next((c for c in possible_gdp if c in df_out.columns), None)
if gdp_col is None:
    raise KeyError("GDP-Spalte nicht gefunden (erwarte 'GDP_B' o.ä.).")

# 2) Zuverlässig numerisch + Division durch 0 absichern
df_out[gdp_col] = pd.to_numeric(df_out[gdp_col], errors="coerce")

# 3) Neue Spalten berechnen (Einheit: kton/Mio€ ≡ kg/€)
col_incl_ratio = "GHG/GDP Agri+Industry+Energy+Industry+Transport+Waste"
col_no_tr_ratio = "GHG/GDP Agri+Industry+Energy+Industry+Waste"

df_out[col_incl_ratio] = np.where(
    df_out[gdp_col] > 0,
    df_out[incl_col] / df_out[gdp_col],
    np.nan
)

df_out[col_no_tr_ratio] = np.where(
    df_out[gdp_col] > 0,
    df_out[no_transport_col] / df_out[gdp_col],
    np.nan
)

# --- Schreiben ---
try:
    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, sheet_name=SHEET_NAME_OUT, index=False)
except Exception:
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name=SHEET_NAME_OUT, index=False)

print(f"Fertig. Datei gespeichert unter: {OUTPUT_FILE}")
