import pandas as pd
from pathlib import Path



# --- Pfade ---
master_path = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/05_Scripts/02_Master_Framework/output/Master_Framework.xlsx"
)

output_dir = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/output/00_Publication/01_JTF_Regions_Rankings"
)
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "01_JTF_Regions_Ranking.xlsx"



# --- 1. Masterdatei einlesen (Sheet 'Index Data') ---
df = pd.read_excel(master_path, sheet_name="Index Data")
# PT herausfiltern (Portugal entfernen)
df = df[df["Country Code"].astype(str).str.strip() != "PT"]

# --- 2. Relevante Spalten definieren ---
cols_to_keep = [
    # Basisinfos
    "Country Code",
    "Country Name",
    "Corresponding NUTS2 level",
    "NUTS 2 Code",
    "Corresponding NUTS1 level",
    "NUTS 1 Code",
    # Vulnerabilitätsindikatoren (inkl. der neuen)
    "Emission Intensity / GDP for selected sectors",
    "National Exports 2019 € Share High Carbon products",
    "National Energy Mix 2019 Share High Carbon",
    "National Share High Carbon Employment 2019",
    "Share High Carbon Employment NUTS 2 2019",
    "Unemployment rates by sex, age, educational attainment level and NUTS 2 region (%) 2019",
    "Share low educational attainment, level 0-2, 2019, NUTS 2",
    "Share of the Population aged 50 to 64, 2019, NUTS 2",
    "Cost of Capital Average Rank 2019",
    "Share Investment Capacity / GVA, 2019, NUTS 2",
    "Subnational HDI 2019",
    "Regional Development Gap 2019",
    # Flag
    "JTF Region",
    "Population",
    "Total Gross value added at basic prices by NUTS 2 region 2019"
]

# --- 3. Fehlende Spalten prüfen ---
missing = [c for c in cols_to_keep if c not in df.columns]
if missing:
    raise ValueError(f"Folgende Spalten fehlen in 'Index Data': {missing}")

# --- 4. Nur JTF-Regions (JTF Region == 1) behalten ---
jtf_df = df[df["JTF Region"] == 1].copy()

# --- 5. Nur gewünschte Spalten behalten ---
jtf_df = jtf_df[cols_to_keep]

# --- 6. NUTS 2 Code säubern und Doppelte nach NUTS2 entfernen ---
jtf_df["NUTS 2 Code"] = jtf_df["NUTS 2 Code"].astype(str).str.strip()
jtf_df = jtf_df.drop_duplicates(subset=["NUTS 2 Code"], keep="first")

# --- 6b. Indikator-spezifische Excludes (nur für bestimmte Spalten) ---
EXCLUDED_NL_HCE = ["NL11", "NL32", "NL34", "NL36", "NL41", "NL42"]
EXCLUDED_CY_EI = ["CY00"]

# sicherstellen, dass Codes vergleichbar sind
jtf_df["NUTS 2 Code"] = jtf_df["NUTS 2 Code"].astype(str).str.strip().str.upper()

excluded_mask_hce = jtf_df["NUTS 2 Code"].isin(EXCLUDED_NL_HCE)   # nur für HCE-Indikator
excluded_mask_ei  = jtf_df["NUTS 2 Code"].isin(EXCLUDED_CY_EI)    # nur für Emission Intensity


# ---------- MIN–MAX NORMALISIERUNG / RANKING ----------

# Hilfsfunktion für Min–Max (inkl. Richtung)
def minmax_series(s, invert=False, exclude_mask=None):
    s_num = pd.to_numeric(s, errors="coerce")

    if exclude_mask is None:
        s_for_scale = s_num
    else:
        s_for_scale = s_num.mask(exclude_mask)  # excluded -> NaN für Min/Max

    min_val = s_for_scale.min()
    max_val = s_for_scale.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        result = pd.Series(0.0, index=s.index)
    else:
        scaled = (s_num - min_val) / (max_val - min_val)
        if invert:
            scaled = 1 - scaled
        result = scaled

    # Original-NaNs bleiben NaN
    result[s_num.isna()] = pd.NA

    # Excluded -> NA (damit sie nicht “irgendwas” bekommen)
    if exclude_mask is not None:
        result[exclude_mask] = pd.NA

    return result


# 1. Gruppe: höherer Wert = vulnerabler (0 -> niedrigster, 1 -> höchster) und gegenteil bei adaptive capacity
group1_cols = [
    "Emission Intensity / GDP for selected sectors",
    "National Exports 2019 € Share High Carbon products",
    "National Energy Mix 2019 Share High Carbon",
    "Share High Carbon Employment NUTS 2 2019",
    "Unemployment rates by sex, age, educational attainment level and NUTS 2 region (%) 2019",
    "Share of the Population aged 50 to 64, 2019, NUTS 2",
    "Share Investment Capacity / GVA, 2019, NUTS 2",
]

# 2. Gruppe: höherer Wert = weniger vulnerabel (umgedreht; 1 = niedrigster Wert), und gegenteil bei adaptive capacity
group2_cols = [
    "Cost of Capital Average Rank 2019",
    "Subnational HDI 2019",
    "Regional Development Gap 2019",
]

# Ranking-Spalten hinzufügen
for col in group1_cols:
    ranking_col = f"Ranking {col}"

    # Default: kein Exclude
    mask = None

    # NL nur bei HCE-Indikator ausschließen
    if col == "Share High Carbon Employment NUTS 2 2019":
        mask = excluded_mask_hce

    # CY00 nur bei Emission Intensity ausschließen
    if col == "Emission Intensity / GDP for selected sectors":
        mask = excluded_mask_ei

    jtf_df[ranking_col] = minmax_series(jtf_df[col], invert=False, exclude_mask=mask)

for col in group2_cols:
    ranking_col = f"Ranking {col}"
    jtf_df[ranking_col] = minmax_series(jtf_df[col], invert=True, exclude_mask=None)



# ---- Originalspalte duplizieren und umbenennen ----
original_col = "Ranking Share High Carbon Employment NUTS 2 2019"
new_col = "Index High Carbon Employment"

# Prüfen, ob die Spalte existiert
if original_col not in jtf_df.columns:
    raise ValueError(f"Die Spalte '{original_col}' existiert nicht im DataFrame.")

# Neue Spalte ans Ende setzen (Kopie der Originalwerte)
jtf_df[new_col] = jtf_df[original_col].copy()

 
# ---------- DIMENSIONS-INDIZES AUS DEN RANKING-SPALTEN BILDEN ----------



def min_max_normalize(series):
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=series.index)
    return (s - mn) / (mx - mn)


# 1a) Index Economic Exposure
economic_cols = [
    "Ranking Emission Intensity / GDP for selected sectors",
    "Ranking Share High Carbon Employment NUTS 2 2019",
    "Ranking National Energy Mix 2019 Share High Carbon",
]

# Prüfe Spalten
missing_econ = [c for c in economic_cols if c not in jtf_df.columns]
if missing_econ:
    raise ValueError(f"Folgende Ranking-Spalten für 'Index Economic Exposure' fehlen: {missing_econ}")

# Berechnung
# Summe nur, wenn alle Komponenten vorhanden sind (sonst NA)
jtf_df["Index Economic Exposure"] = jtf_df[economic_cols].sum(axis=1, min_count=len(economic_cols))

# Danach normalisieren (NA bleibt NA)
jtf_df["Index Economic Exposure"] = min_max_normalize(jtf_df["Index Economic Exposure"])





# 2) Index Adaptive Capacity (Finanzielle Kapazität)
adaptive_fin_cols = [
    "Ranking Cost of Capital Average Rank 2019",
    "Ranking Share Investment Capacity / GVA, 2019, NUTS 2",
]

missing_adapt_fin = [c for c in adaptive_fin_cols if c not in jtf_df.columns]
if missing_adapt_fin:
    raise ValueError(f"Folgende Ranking-Spalten für 'Index Adaptive Capacity' fehlen: {missing_adapt_fin}")

# Berechnung
jtf_df["Index Adaptive Capacity"] = jtf_df[adaptive_fin_cols].sum(axis=1)
jtf_df["Index Adaptive Capacity"] = min_max_normalize(jtf_df["Index Adaptive Capacity"])


# 3a) Index Socioeconomic Sensitivity
adaptive_soc_cols = [
    "Ranking Subnational HDI 2019",
    "Ranking Regional Development Gap 2019",
    "Ranking Share of the Population aged 50 to 64, 2019, NUTS 2",
]

missing_adapt_soc = [c for c in adaptive_soc_cols if c not in jtf_df.columns]
if missing_adapt_soc:
    raise ValueError(f"Folgende Ranking-Spalten für 'Index Socioeconomic Sensitivity' fehlen: {missing_adapt_soc}")

# Berechnung
jtf_df["Index Socioeconomic Sensitivity"] = jtf_df[adaptive_soc_cols].sum(axis=1)
jtf_df["Index Socioeconomic Sensitivity"] = min_max_normalize(
    jtf_df["Index Socioeconomic Sensitivity"]
)


# ---------- Datei schreiben ----------
jtf_df.to_excel(output_file, index=False, sheet_name="JTF Regions Ranking")


print(f"Fertig! Datei gespeichert unter:\n{output_file}")
