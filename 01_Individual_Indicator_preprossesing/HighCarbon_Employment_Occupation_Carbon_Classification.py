import pandas as pd
from pathlib import Path
from functools import reduce

# === 1. Pfad zur Datei setzen ===
filepath = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/02_Indicators/01_Raw Data/05_High Carbon Employment/NACE x ISCO/ZaussingerSchmidtEgli2025_OccupationClassificationESCOv1.xlsx")

# === 2. Excel-Datei laden ===
df = pd.read_excel(filepath)
print("Spaltennamen:", df.columns.tolist())

# === 3. Relevante Spalten extrahieren ===
df_filtered = df[["preferredLabel", "occupation_category", "ISCOMaingroup"]].copy()

# === 4. Gruppieren und zählen ===
result = (
    df_filtered
    .groupby(["ISCOMaingroup", "occupation_category"])
    .count()
    .reset_index()
    .rename(columns={"preferredLabel": "count"})
)

# === 5. In übersichtliches Format bringen ===
pivot_result = result.pivot(index="ISCOMaingroup", columns="occupation_category", values="count").fillna(0).astype(int)

# === 6. Anteil berechnen ===
viable = "viable-to-decarbonize"
unviable = "unviable-to-decarbonize"

pivot_result["Sum_Occupations"] = pivot_result.sum(axis=1)
pivot_result["High_Carbon_Jobs"] = pivot_result.get(viable, 0) + pivot_result.get(unviable, 0)
pivot_result["Share_High-Carbon-Occupations"] = (pivot_result["High_Carbon_Jobs"] / pivot_result["Sum_Occupations"]).round(3)
pivot_result = pivot_result.drop(columns="High_Carbon_Jobs")

# Ergebnis anzeigen
print(pivot_result)

# === 6b. ISCO-Namen an Eurostat-Formulierungen anpassen ===
rename_isco = {
    "Clerical support workers": "Clerical support workers",
    "Craft related trades workers": "Craft and related trades workers",
    "Elementary occupations": "Elementary occupations",
    "Managers": "Managers",
    "Plant and machine operators, and assemblers": "Plant and machine operators and assemblers",
    "Professionals": "Professionals",
    "Service and sales workers": "Service and sales workers",
    "Skilled agricultural, forestry and fishery workers": "Skilled agricultural, forestry and fishery workers",
    "Technicians and associate professionals": "Technicians and associate professionals"
}
pivot_result.rename(index=rename_isco, inplace=True)
# === 6c. Zusätzliche Zeilen für "No response" und "Armed forces occupations" ===
additional_rows = pd.DataFrame(
    0,
    index=["No response", "Armed forces occupations"],
    columns=pivot_result.columns
)

# Zeilen anhängen
pivot_result = pd.concat([pivot_result, additional_rows])
# === 6d. Setze Share_High-Carbon-Occupations explizit auf 0 für die beiden neuen Zeilen ===
pivot_result.loc["No response", "Share_High-Carbon-Occupations"] = 0
pivot_result.loc["Armed forces occupations", "Share_High-Carbon-Occupations"] = 0

# === Export ===
output_path = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/05_Scripts/01_Indicators/05_High Carbon Employment/output/Share_high_carbon_occupation.xlsx")
output_path.parent.mkdir(parents=True, exist_ok=True)
pivot_result.to_excel(output_path)

print(f"Erfolgreich exportiert nach: {output_path.resolve()}")

# === 7. Eurostat-Daten (2019) aus 12 Sheets zusammenführen ===
eurostat_path = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/02_Indicators/01_Raw Data/05_High Carbon Employment/NACE x ISCO/lfsa_eisn2__custom_17673446_spreadsheet.xlsx")
sheet_names = [f"Sheet {i}" for i in range(1, 13)]
eurostat_dfs = []

for sheet_name in sheet_names:
    sheet = pd.read_excel(eurostat_path, sheet_name=sheet_name, header=None)
    occupation = str(sheet.iloc[8, 2]).strip()
    countries = sheet.iloc[13:51, 0]
    values_2019 = sheet.iloc[13:51, 1]
    valid_rows = countries.notna() & ~countries.astype(str).str.contains(":", na=False)

    df_occ = pd.DataFrame({
        "Country": countries[valid_rows].reset_index(drop=True),
        occupation: values_2019[valid_rows].reset_index(drop=True)
    })
    eurostat_dfs.append(df_occ)

merged_df = reduce(lambda left, right: pd.merge(left, right, on="Country", how="outer"), eurostat_dfs)

print("\nEurostat-Zusammenfassung (2019):")
print(merged_df.head())

# === 8b. Berechne neue Gesamtsumme über alle Occupational-Spalten (C bis M) ===

# Hole die relevanten Spaltennamen (C bis M) → also alle außer "Country"
occupation_columns = merged_df.columns[2:13]  # Spalten C bis M = 11 Spalten

# Konvertiere alle Werte in diesen Spalten in float (falls noch nicht geschehen)
merged_df[occupation_columns] = merged_df[occupation_columns].apply(pd.to_numeric, errors="coerce")

# Summiere über diese Spalten pro Zeile
merged_df["Total_New"] = merged_df[occupation_columns].sum(axis=1)

# === 9. Berechne Share_xxx Spalten durch Matching mit pivot_result ===

# Hole Share-Werte aus Sheet1
share_values = pivot_result["Share_High-Carbon-Occupations"]

# Liste aller Occupation-Spalten (ohne "Country")
occupation_columns = merged_df.columns[1:]

for col in occupation_columns:
    occ_name = col.strip()  # z. B. "Managers"

    if occ_name in share_values.index:
        try:
            # Versuche die Spalte in numerische Werte umzuwandeln
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

            # Multipliziere zeilenweise: Eurostat-Wert * Share-Wert
            merged_df[f"Share_{occ_name}"] = merged_df[col] * share_values[occ_name]
        except Exception as e:
            print(f"❌ Fehler bei Spalte {col}: {e}")
    else:
        print(f"⚠️ Achtung: {occ_name} nicht in Sheet1 vorhanden – wird übersprungen.")
# === 9a. EU-/ISO-Länderkürzel (inkl. EU-konformer Sonderfälle) als neue Spalte A einfügen ===

import re

def _normalize(s: str) -> str:
    s = str(s)
    s = s.replace("–", "-").replace("—", "-")  # Gedankenstriche vereinheitlichen
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# Mapping auf EU-/ISO-Alpha-2 (inkl. Sonderfälle & Aggregate)
code_map_raw = {
    "Austria": "AT",
    "Belgium": "BE",
    "Bosnia and Herzegovina": "BA",
    "Bulgaria": "BG",
    "Croatia": "HR",
    "Cyprus": "CY",
    "Czechia": "CZ",
    "Denmark": "DK",
    "Estonia": "EE",
    "Euro area - 20 countries (from 2023)": "EA20",  # Aggregat
    "European Union - 27 countries (from 2020)": "EU27_2020",  # Aggregat
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Greece": "EL",         # EU nutzt EL
    "Hungary": "HU",
    "Iceland": "IS",
    "Ireland": "IE",
    "Italy": "IT",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Malta": "MT",
    "Montenegro": "ME",
    "Netherlands": "NL",
    "North Macedonia": "MK",
    "Norway": "NO",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Serbia": "RS",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Türkiye": "TR",
    "United Kingdom": "UK",  # EU nutzt UK
}

# Robuster machen: auch Varianten mit anderem Strich zulassen
code_map_raw["Euro area – 20 countries (from 2023)"] = "EA20"
code_map_raw["European Union – 27 countries (from 2020)"] = "EU27_2020"
code_map_raw["Turkey"] = "TR"  # falls Quelle ohne Diakritikum liefert

# Normalisierte Nachschlagetabelle
code_map = {_normalize(k): v for k, v in code_map_raw.items()}

# Codes aus Country ableiten
merged_df["__GeoCode_tmp__"] = merged_df["Country"].apply(lambda x: code_map.get(_normalize(x), None))

# Fehlende Codes melden (nur zur Kontrolle in der Konsole)
_missing = merged_df.loc[merged_df["__GeoCode_tmp__"].isna(), "Country"].dropna().unique().tolist()
if _missing:
    print("⚠️ Keine Zuordnung für:", _missing)

# Neue Spalte A mit Codes einfügen, Rest nach rechts schieben
merged_df.insert(0, "Country Code", merged_df.pop("__GeoCode_tmp__"))

# (Optional) Country-Spalte umbenennen, damit klar ist, dass es der Klarname ist
# merged_df.rename(columns={"Country": "Country_Name"}, inplace=True)
# === 10. Summiere alle Share-Spalten und berechne Anteil an Total_New ===

# Finde alle Spalten, die mit 'Share_' beginnen (aber nicht 'Share_High-Carbon-Occupations')
share_columns = [col for col in merged_df.columns if col.startswith("Share_") and col != "Share_High-Carbon-Occupations"]

# Summiere diese Spalten zeilenweise
merged_df["Sum_Share_Values"] = merged_df[share_columns].sum(axis=1)

# Berechne Anteil an Gesamtbeschäftigung (Total_New)
merged_df["Share_Total_High_Carbon"] = (merged_df["Sum_Share_Values"] / merged_df["Total_New"]).round(3)

# === 8. Als neues Tabellenblatt in bestehende Datei schreiben ===
with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    merged_df.to_excel(writer, sheet_name="Eurostat_2019", index=False)

print("Eurostat-Zusammenfassung 2019 wurde als Tabellenblatt 'Eurostat_2019' hinzugefügt.")
