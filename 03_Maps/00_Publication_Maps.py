import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import re
import colorsys
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.ticker as ticker

mpl.rcParams["hatch.linewidth"] = 1.2  # testweise 1.0–2.0


# =========================================
# 1. Pfade anpassen
# =========================================
excel_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/output/00_Publication/01_JTF_Regions_Rankings/01_JTF_Regions_Ranking.xlsx"
geojson_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/06_Mapping/Map_Files/ref-nuts-2021-20m/NUTS_RG_20M_2021_4326_LEVL_2.geojson"
output_dir = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/06_Mapping/00_Publication"
nuts0_path = "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/06_Mapping/Map_Files/ref-nuts-2021-20m/NUTS_RG_20M_2021_4326_LEVL_0.geojson"

os.makedirs(output_dir, exist_ok=True)
export_xlsx = os.path.join(output_dir, "figure_data_export_maps.xlsx")

# =========================================
# 2. Daten laden
# =========================================
print("Lade Excel-Datei...")
df = pd.read_excel(excel_path)

print("Lade GeoJSON...")
gdf = gpd.read_file(geojson_path)
gdf_countries = gpd.read_file(nuts0_path)

print(df.columns)

# -----------------------------------------
# Filter: Türkei, Island, Grönland raus + Arktis-Ausreißer raus
# (noch in EPSG:4326, daher funktioniert Latitude-Filter)
# -----------------------------------------
gdf["NUTS_ID"] = gdf["NUTS_ID"].astype(str).str.strip().str.upper()
gdf_countries["CNTR_CODE"] = gdf_countries["CNTR_CODE"].astype(str).str.strip().str.upper()

#no data jtf regions
EXCLUDED_PT_ALL = ["PT11", "PT1D", "PT1C"]
EXCLUDED_NL_EE = ["NL11", "NL32", "NL34", "NL36", "NL41", "NL42", "CY00"]
EXCLUDED_NL_HCE = ["NL11", "NL32", "NL34", "NL36", "NL41", "NL42"]

# Türkei/Island/Grönland über NUTS_ID ausschließen
gdf = gdf[
    ~gdf["NUTS_ID"].str.startswith("TR") &
    ~gdf["NUTS_ID"].str.startswith("GL")
].copy()

# Arktis-Ausreißer (Svalbard/Jan Mayen etc.) per Latitude abschneiden
# (Centroid in EPSG:4326 -> y ist Latitude)
gdf = gdf[gdf.geometry.centroid.y < 72].copy()

# Ländergrenzen ebenfalls: Türkei raus + (falls vorhanden) SJ raus
gdf_countries = gdf_countries[~gdf_countries["CNTR_CODE"].isin(["TR", "SJ"])].copy()

# Jetzt projizieren
gdf = gdf.to_crs(3035)
gdf_countries = gdf_countries.to_crs(3035)


nuts_col_geo = "NUTS_ID"

if nuts_col_geo not in gdf.columns:
    raise ValueError(f"Spalte '{nuts_col_geo}' nicht in GeoDataFrame gefunden. Verfügbare Spalten: {gdf.columns}")

if "NUTS 2 Code" not in df.columns:
    raise ValueError(f"Spalte 'NUTS 2 Code' nicht in Excel gefunden. Verfügbare Spalten: {df.columns}")

# Übersee-Regionen entfernen
overseas_nuts2 = ["FRY1", "FRY2", "FRY3", "FRY4", "FRY5", "ES70", "PT20", "PT30"]
gdf = gdf[~gdf[nuts_col_geo].isin(overseas_nuts2)]

print("Mache Merge von GeoDaten und Indizes...")
merged = gdf.merge(df, left_on=nuts_col_geo, right_on="NUTS 2 Code", how="left")
merged = merged.to_crs(3035)

# =========================================
# EXPORT: base + merged data used for maps
# =========================================
with pd.ExcelWriter(export_xlsx, engine="openpyxl") as writer:
    # Excel input (attributes)
    df.to_excel(writer, sheet_name="excel_input", index=False)

    # Geo inputs: keep only attributes (geometry separately is not Excel-friendly)
    gdf.drop(columns="geometry", errors="ignore").to_excel(writer, sheet_name="nuts2_geo_attrs", index=False)
    gdf_countries.drop(columns="geometry", errors="ignore").to_excel(writer, sheet_name="nuts0_geo_attrs", index=False)

    # Merged attributes used for plotting (exclude geometry to keep Excel clean)
    merged.drop(columns="geometry", errors="ignore").to_excel(writer, sheet_name="merged_plot_attrs", index=False)

print("Saved map figure data Excel to:", export_xlsx)

# =========================================
# 3. Farbdefinitionen für die Indikatoren
# =========================================

def _hex_to_rgb01(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

def _rgb01_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c*255)) for c in rgb))

def tweak_hls(hex_color: str, l=None, s=None, h_shift=0.0):
    r, g, b = _hex_to_rgb01(hex_color)
    h, l0, s0 = colorsys.rgb_to_hls(r, g, b)
    h = (h + h_shift) % 1.0
    l = l0 if l is None else l
    s = s0 if s is None else s
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex((r2, g2, b2))

def make_fresh_hex_cmap(base_hex: str, name: str):
    # Start: sehr hell, leicht „airy“
    light = tweak_hls(base_hex, l=0.95, s=0.35)
    # Mid: frisch/gesättigt, aber nicht zu dunkel
    mid   = tweak_hls(base_hex, l=0.65, s=0.75)
    # High: dunkler, aber mit etwas mehr Sättigung für Differenzierung
    high  = tweak_hls(base_hex, l=0.35, s=0.85)
    # Max: exakt dein gewähltes Dunkel (oder du setzt hier high, wenn max auch getweakt sein darf)
    return LinearSegmentedColormap.from_list(name, [light, mid, high, base_hex])

indicator_cmaps = {
    "Economic Exposure": make_fresh_hex_cmap("#0A2A58", "EE_cmap"),
    "Adaptive Capacity": make_fresh_hex_cmap("#0E5B2C", "AC_cmap"),
    "Socioeconomic Sensitivity": make_fresh_hex_cmap("#510A94", "SS_cmap"),
}


# =========================================
# 3b. Zusätzliche (manuelle) Karten-Spalten
# =========================================
manual_cols_cmaps = {
    "Ranking National Energy Mix 2019 Share High Carbon": "Blues",
    "Ranking Emission Intensity / GDP for selected sectors": "Blues"
}

# =========================================
# 4. Helper zum Parsen der Index-Spalten
# =========================================
def parse_index_column(col_name: str):
    if not col_name.startswith("Index "):
        return None
    base = col_name[len("Index "):].strip()
    return base, "01", None


def make_quantile_bins(series, k=10, lower_q=0.025, upper_q=0.975):
    """
    Erzeugt k Klassen-Grenzen basierend auf Quantilen (robust gegen Ausreißer).
    Rückgabe: Liste mit oberen Klassengrenzen für mapclassify (UserDefined bins).
    """
    s = series.dropna()

    # Wenn Serie zu wenig Werte hat, fallback
    if s.nunique() < 2:
        return None

    qs = np.linspace(lower_q, upper_q, k)
    bins = s.quantile(qs).values

    # bins müssen strikt monoton steigend sein
    bins = np.unique(bins)
    if len(bins) < 2:
        return None

    return bins.tolist()


# =========================================
# 5. Alle Index-Spalten durchgehen und Heatmaps erzeugen
# =========================================
coverage_rows = []


index_cols_to_map = [
    "Index Economic Exposure",
    "Index Adaptive Capacity",
    "Index Socioeconomic Sensitivity",
]

for col in index_cols_to_map:

    base_name_clean = col.replace("Index ", "").strip()

    cmap = indicator_cmaps.get(base_name_clean)
    if cmap is None:
        print(f"Überspringe Spalte '{col}' – keine Farbdefinition für '{base_name_clean}'.")
        continue

    method_code = "01"
    method_tag = None


    # Dateinamen aufbauen
    # Dateiname sicher machen (keine /, keine Sonderzeichen)
    indicator_slug = re.sub(r"[^A-Za-z0-9_]+", "_", base_name_clean.replace(" ", "_")).strip("_")

    filename = f"{method_code}_Mapping_{indicator_slug}"
    if method_tag is not None:
        filename += f"_{method_tag}"

    filepath = os.path.join(output_dir, filename + ".pdf")

    print(f"Erstelle Karte für '{col}' → {filepath}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # =========================================
    # Sonderfall: Für "Economic Exposure" Niederlande entfernen
    # =========================================
    plot_df = merged.copy()
    plot_col = col  # we keep plotting the real column

    excluded_ids = list(EXCLUDED_PT_ALL)
    if base_name_clean == "Economic Exposure":
        excluded_ids += EXCLUDED_NL_EE

    mask_excluded = plot_df["NUTS_ID"].isin(excluded_ids)
    plot_df_main = plot_df.loc[~mask_excluded].copy()
    plot_df_excl = plot_df.loc[mask_excluded].copy()



    # =========================================
    # Coverage: wie viele Regionen haben für diese Dimension einen Wert?
    # =========================================
    n_covered_regions = plot_df.loc[plot_df[plot_col].notna(), "NUTS_ID"].nunique()
    n_total_regions_in_map = plot_df["NUTS_ID"].nunique()
    share_covered = (n_covered_regions / n_total_regions_in_map) * 100 if n_total_regions_in_map else 0

    coverage_rows.append({
        "column": col,
        "indicator": base_name_clean,
        "method_code": method_code,
        "method_tag": method_tag if method_tag is not None else "",
        "n_total_regions_in_map": int(n_total_regions_in_map),
        "n_covered_regions": int(n_covered_regions),
        "share_covered_pct": round(share_covered, 2),
    })
    print(f"Coverage for '{col}': {n_covered_regions}/{n_total_regions_in_map} regions ({share_covered:.2f}%)")


    # =========================================
    # NEU: für alle Indikatoren robuste Skalierung mit Perzentilen
    # =========================================
    vmin = merged[col].quantile(0.025)
    vmax = merged[col].quantile(0.975)

    # =========================================
    # Klassierte Legende (Ranges) statt Colorbar
    # =========================================
    bins = make_quantile_bins(plot_df_main[plot_col], k=10, lower_q=0.025, upper_q=0.975)


    plot_df_main.plot(
        column=plot_col,
        ax=ax,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.5,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
        },
        legend=False
    )

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []  # notwendig für manche matplotlib-Versionen

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.03,
        pad=-0.13,
        shrink=0.6   # ← z.B. 60% der ursprünglichen Länge
    )

    cbar.ax.tick_params(labelsize=16)
    cbar.outline.set_visible(False)

    # Schritt 3: Formatierung (eine Nachkommastelle)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()

    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)
    # Overlay: excluded regions as grey hatched
    # Note: hatch support depends on geopandas/matplotlib versions; in most setups this works.
    plot_df_excl.plot(
        ax=ax,
        color="lightgrey",
        edgecolor="#7F7F7F",   # wichtig: hatch nimmt i.d.R. diese Farbe
        linewidth=0.4,
        hatch="////",       # dichter als vorher
        zorder=5
    )

    excluded_patch = mpatches.Patch(
        facecolor="lightgrey",
        edgecolor="#7F7F7F",
        hatch="////",
        label="eligible, no data"
    )

    not_jtf_patch = mpatches.Patch(
        facecolor="lightgrey",                  
        edgecolor="white",
        label="not JTF eligible"
    )

    if base_name_clean == "Adaptive Capacity":
        ax.legend(
            handles=[excluded_patch, not_jtf_patch],
            loc="upper right",
            bbox_to_anchor=(1.06, 0.98),  # ggf. deine neue Position
            frameon=False,
            fontsize=16,
        )


    # Ländergrenzen zeichnen
    gdf_countries.boundary.plot(
        ax=ax,
        linewidth=0.5,
        edgecolor="grey"
    )

    # Dynamischen Ausschnitt aus den Daten bestimmen
    # Fester Europa-Ausschnitt (ohne leere Ränder durch Ausreißer)
    ax.set_xlim(2600000, 6700000) # rechts etwas enger
    ax.set_ylim(1300000, 5500000)


    ax.set_axis_off()

    # Titel
    title = base_name_clean
    if method_tag is not None:
        pretty_method = {
            "national_weightings_0_5": "national weightings 0.5",
            "PCA": "PCA",
            "Entropy": "Entropy"
        }[method_tag]
        title += f" ({pretty_method})"

    #ax.set_title(title, fontsize=14)
    #ax.title.set_position((0.37, 1.02))
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, transparent=True)
    plt.close(fig)

coverage_df = pd.DataFrame(coverage_rows).sort_values(["indicator", "method_code", "method_tag"])
coverage_out = os.path.join(output_dir, "coverage_by_dimension.csv")
coverage_df.to_csv(coverage_out, index=False)
print("\nCoverage summary saved to:", coverage_out)
print(coverage_df[["indicator", "method_code", "method_tag", "n_covered_regions", "n_total_regions_in_map", "share_covered_pct"]].to_string(index=False))


print("Fertig – alle Karten wurden erstellt und gespeichert.")
