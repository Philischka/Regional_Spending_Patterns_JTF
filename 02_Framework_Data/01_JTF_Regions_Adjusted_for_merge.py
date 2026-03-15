import os
import pandas as pd

# Pfade
input_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/output/00_Publication/01_JTF_Regions_Rankings/01_JTF_Regions_Ranking.xlsx"
output_dir = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/output/00_Publication/01_JTF_Regions_Rankings/Adjusted_for_merge"
output_path = os.path.join(output_dir, "01_JRF_Region_Ranking_NUTS2_1_0.xlsx")

os.makedirs(output_dir, exist_ok=True)

# Datei einlesen
df = pd.read_excel(input_path)

# Relevante Spalten
base_cols = [
    "Country Code",
    "Country Name",
    "Corresponding NUTS2 level",
    "NUTS 2 Code"
]

index_cols = [
    "Index High Carbon Employment",
    "Index Economic Exposure",
    "Index Adaptive Capacity",
    "Index Socioeconomic Sensitivity", 
    "Ranking Cost of Capital Average Rank 2019",
    "Share High Carbon Employment NUTS 2 2019",
    "Ranking Subnational HDI 2019",
    "Ranking Regional Development Gap 2019",
    "Ranking National Energy Mix 2019 Share High Carbon",
    "Ranking Emission Intensity / GDP for selected sectors",
    "Ranking Share of the Population aged 50 to 64, 2019, NUTS 2",
    "Ranking Share High Carbon Employment NUTS 2 2019",
]


# ---------------------------------------------------
# 1) NUTS-2: pro NUTS2-Code genau eine Zeile (Mittelwert der Indizes)
# ---------------------------------------------------
nuts2 = (
    df.groupby(
        ["Country Code", "Country Name", "Corresponding NUTS2 level", "NUTS 2 Code"],
        as_index=False
    )[index_cols]
    .mean()
)
nuts2["Level"] = "NUTS2"   # optional zur Kennzeichnung

# ---------------------------------------------------
# 2) NUTS-1: Durchschnitt über alle NUTS-2 mit gleichem NUTS 1 Code
# ---------------------------------------------------
if "NUTS 1 Code" not in df.columns:
    raise ValueError("Die Spalte 'NUTS 1 Code' fehlt in der Eingabedatei.")

nuts1_grouped = (
    df.dropna(subset=["NUTS 1 Code"])
      .groupby(["Country Code", "Country Name", "NUTS 1 Code"], as_index=False)[index_cols]
      .mean()
)

# In das gewünschte Format bringen
nuts1 = nuts1_grouped.copy()
nuts1["Corresponding NUTS2 level"] = "NUTS1 aggregate"

# Hier explizit: in 'NUTS 2 Code' steht der NUTS1-Code
nuts1["NUTS 2 Code"] = nuts1["NUTS 1 Code"]
nuts1["Level"] = "NUTS1"

nuts1 = nuts1[base_cols + index_cols + ["Level"]]

# ---------------------------------------------------
# 3) NUTS-0: Durchschnitt über alle NUTS-2 pro Land (Country Code)
# ---------------------------------------------------
nuts0_grouped = (
    df.groupby(["Country Code", "Country Name"], as_index=False)[index_cols]
      .mean()
)

nuts0 = nuts0_grouped.copy()
nuts0["Corresponding NUTS2 level"] = "NUTS0 aggregate"

# Auch hier explizit: in 'NUTS 2 Code' steht der Country Code (NUTS0-Code)
nuts0["NUTS 2 Code"] = nuts0["Country Code"]
nuts0["Level"] = "NUTS0"

nuts0 = nuts0[base_cols + index_cols + ["Level"]]

# ---------------------------------------------------
# 4) Alles zusammenführen
#    -> jeder Code in 'NUTS 2 Code' kommt nur einmal vor,
#       weil wir auf jeder Ebene nach diesem Code gruppiert haben.
# ---------------------------------------------------
result = pd.concat([nuts2, nuts1, nuts0], ignore_index=True)

# Optional: prüfen, ob jeder Code tatsächlich nur einmal vorkommt
dup_codes = result["NUTS 2 Code"][result["NUTS 2 Code"].duplicated()].unique()
if len(dup_codes) > 0:
    print("Warnung: folgende Codes kommen mehrfach vor:", dup_codes)

# ---------------------------------------------------
# 6) NUR am Ende den „NUTS1 aggregate“-Wert ersetzen
# ---------------------------------------------------

# Die Originaldatei erneut einlesen
original_df = pd.read_excel(input_path)

# Extrahiere die relevanten Spalten für den NUTS1 Code und den NUTS1 Level
nuts1_region_names = original_df[["NUTS 1 Code", "Corresponding NUTS1 level"]]

# Optional: Überprüfen, ob es doppelte NUTS 1 Codes gibt
nuts1_region_names = nuts1_region_names.drop_duplicates(subset="NUTS 1 Code", keep="first")

# Durchlaufen der Zeilen im 'result', um den Wert zu ersetzen
# Falls 'Corresponding NUTS2 level' den Wert 'NUTS1 aggregate' enthält
# und der 'NUTS 2 Code' mit einem 'NUTS 1 Code' übereinstimmt,
# dann den 'Corresponding NUTS1 level' aus der Originaldatei übernehmen
for index, row in result.iterrows():
    if row["Corresponding NUTS2 level"] == "NUTS1 aggregate":
        # Match mit NUTS 1 Code durchführen
        matching_nuts1 = nuts1_region_names[nuts1_region_names["NUTS 1 Code"] == row["NUTS 2 Code"]]
        if not matching_nuts1.empty:
            # Wenn ein Match gefunden wird, den "Corresponding NUTS1 level" übernehmen
            result.at[index, "Corresponding NUTS2 level"] = matching_nuts1.iloc[0]["Corresponding NUTS1 level"]

# Optional: Verifikation der Änderungen
print(result[["NUTS 2 Code", "Corresponding NUTS2 level"]].head())
translation_map = {
    "Kärnten": "Carinthia",
    "Niederösterreich": "Lower Austria)",
    "Steiermark": "Styria",
    "Prov. Hainaut": "Province of Hainaut",
    "Yugoiztochen": "Southeastern (BG)",
    "Yugozapaden": "Southwestern (BG)",
    "Cyprus": "Cyprus",
    "Moravskoslezsko": "Moravian-Silesian Region",
    "Severozápad": "Northwest (CZ)",
    "Brandenburg": "Brandenburg",
    "Chemnitz": "Chemnitz",
    "Dresden": "Dresden",
    "Düsseldorf": "Düsseldorf",
    "Köln": "Cologne",
    "Leipzig": "Leipzig",
    "Münster": "Münster",
    "Sachsen-Anhalt": "Saxony-Anhalt",
    "Nordjylland": "North Jutland",
    "Syddanmark": "Southern (DK)",
    "Eesti": "Estonia",
    "Dytiki Makedonia": "Western Macedonia",
    "Kriti": "Crete",
    "Notio Aigaio": "South Aegean",
    "Peloponnisos": "Peloponnese",
    "Voreio Aigaio": "North Aegean",
    "Andalucía": "Andalusia",
    "Aragón": "Aragon",
    "Castilla y León": "Castile and León",
    "Galicia": "Galicia",
    "Illes Balears": "Balearic Islands",
    "Principado de Asturias": "Principality of Asturias",
    "Etelä-Suomi": "Southern (FI)",
    "Länsi-Suomi": "Western (FI)",
    "Pohjois- ja Itä-Suomi": "Northern and Eastern Finland",
    "Alsace": "Alsace",
    "Haute-Normandie": "Upper Normandy",
    "Lorraine": "Lorraine",
    "Nord-Pas de Calais": "Nord–Pas-de-Calais",
    "Pays de la Loire": "Pays de la Loire",
    "Provence-Alpes-Côte d’Azur": "Provence–Alpes–Côte d’Azur",
    "Rhône-Alpes": "Rhône-Alpes",
    "Jadranska Hrvatska": "Adriatic Croatia",
    "Panonska Hrvatska": "Pannonian Croatia",
    "Dél-Dunántúl": "Southern Transdanubia",
    "Észak-Magyarország": "Northern Hungary",
    "Eastern and Midland": "Eastern and Midland (IE)",
    "Puglia": "Apulia",
    "Sardegna": "Sardinia",
    "Vidurio ir vakarų Lietuvos regionas": "Central and Western Lithuania",
    "Luxembourg": "Luxembourg",
    "Latvija": "Latvia",
    "Malta": "Malta",
    "Groningen": "Groningen",
    "Limburg (NL)": "Limburg (NL)",
    "Noord-Brabant": "North Brabant",
    "Noord-Holland": "North Holland",
    "Zeeland": "Zeeland",
    "Zuid-Holland": "South Holland",
    "Dolnośląskie": "Lower Silesia",
    "Małopolskie": "Lesser Poland",
    "Wielkopolskie": "Greater Poland",
    "Łódzkie": "Łódź Region",
    "Śląskie": "Silesia",
    "Centru": "Centre",
    "Sud-Est": "South-East",
    "Sud-Muntenia": "South-Muntenia",
    "Sud-Vest Oltenia": "South-West Oltenia",
    "Vest": "West",
    "Småland med öarna": "Småland and Islands",
    "Övre Norrland": "Upper Norrland",
    "Vzhodna Slovenija": "Eastern (SI)",
    "Stredné Slovensko": "Central (SK)",
    "Východné Slovensko": "Eastern (SK)",
    "Západné Slovensko": "Western (SK)",
    "Ostösterreich": "Eastern (AT)",
    "Südösterreich": "Southern (AT)",
    "Région wallonne": "Walloon Region",
    "Severna i Yugoiztochna Balgariya": "Northern and Southeastern (BG)",
    "Yugozapadna i Yuzhna tsentralna Balgariya": "Southwestern and South-Central (BG)",
    "Česko": "Czechia",
    "Nordrhein-Westfalen": "NRW",
    "Sachsen": "Saxony",
    "Danmark": "Denmark",
    "Nisia Aigaiou, Kriti": "Aegean Islands and Crete",
    "Voreia Ellada": "Northern (EL)",
    "Kentriki Ellada": "Central (EL)",
    "Noroeste": "Northwest (ES)",
    "Noreste": "Northeast (ES)",
    "Centro (ES)": "Centre (ES)",
    "Este": "East (ES)",
    "Sur": "South (ES)",
    "Manner-Suomi": "Mainland (FI)",
    "Normandie": "Normandy",
    "Hauts-de-France": "Hauts-de-France",
    "Grand Est": "Grand Est",
    "Auvergne-Rhône-Alpes": "Auvergne–Rhône–Alpes",
    "Hrvatska": "Croatia",
    "Dunántúl": "Transdanubia",
    "Alföld és Észak": "Great Plain and North (HU)",
    "Ireland": "Ireland",
    "Sud": "South (IT)",
    "Isole": "Islands (IT)",
    "Lietuva": "Lithuania",
    "Noord-Nederland": "Northern Netherlands",
    "West-Nederland": "Western Netherlands",
    "Zuid-Nederland": "South Netherlands",
    "Makroregion południowy": "Southern Macroregion",
    "Makroregion północno-zachodni": "Northwest Macroregion",
    "Makroregion południowo-zachodni": "Southwest Macroregion",
    "Makroregion centralny": "Central Macroregion",
    "Macroregiunea Unu": "Macroregion 1 (RO)",
    "Macroregiunea Doi": "Macroregion 2 (RO)",
    "Macroregiunea Trei": "Macroregion 3 (RO)",
    "Macroregiunea Patru": "Macroregion 4 (RO)",
    "Södra Sverige": "Southern Sweden",
    "Norra Sverige": "Northern Sweden",
    "Slovenija": "Slovenia",
    "Slovensko": "Slovakia"
}
# Übersetzen der Regionsnamen
result["Corresponding NUTS2 level"] = result["Corresponding NUTS2 level"].replace(translation_map)

# Check: welche Namen wurden NICHT übersetzt?
missing = set(result["Corresponding NUTS2 level"]) - set(translation_map.values())
print("Unübersetzte Namen:", missing)

# ---------------------------------------------------
# 4b) Spalte "Total Employment 2019" aus externer Datei hinzufügen
# ---------------------------------------------------
employment_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/02_Indicators/01_Raw Data/05_High Carbon Employment/lfst_r_lfe2en2__custom_17554439_spreadsheet.xlsx"

# Rohdaten ohne Header einlesen, damit Zeilen/Spalten fix indexiert bleiben
emp_raw = pd.read_excel(employment_path, sheet_name="Sheet 1", header=None)

# Ab Zeile 13 (Excel = Index 12): Spalte A (0) = NUTS-Code, Spalte C (2) = Employment 2019
emp = emp_raw.loc[12:, [0, 2]].copy()
emp.columns = ["NUTS 2 Code", "Total Employment 2019"]

# Zeilen ohne NUTS-Code entfernen
emp = emp.dropna(subset=["NUTS 2 Code"])

# NUTS-Codes sicher als String behandeln
emp["NUTS 2 Code"] = emp["NUTS 2 Code"].astype(str)
result["NUTS 2 Code"] = result["NUTS 2 Code"].astype(str)

# Werte liegen in "Thousand persons" vor → Umrechnung in absolute Personen
emp["Total Employment 2019"] = emp["Total Employment 2019"] * 1000

# Left Join (alle 'result'-Zeilen behalten)
result = result.merge(emp, on="NUTS 2 Code", how="left")


# ---------------------------------------------------
# 5) Neue Excel-Datei speichern
# ---------------------------------------------------
result.to_excel(output_path, index=False)

print("Neue Tabelle gespeichert unter:")
print(output_path)
