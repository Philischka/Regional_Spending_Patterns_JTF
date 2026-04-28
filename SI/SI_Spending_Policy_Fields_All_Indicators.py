import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Pfade ANPASSEN
# ============================================================
vul_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/output/00_Publication/01_JTF_Regions_Rankings/Adjusted_for_merge/01_JRF_Region_Ranking_NUTS2_1_0.xlsx"
flows_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/04_JTF_Finances_Details/Data_used/2021-2027_JTF_Finances_Details_with_NUTS_20251115.csv"

output_dir = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/SI_Policy_Categories_per_Indicator"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Farben und feste Einstellungen
# ============================================================
policy_colors = {
    "Decarbonization": "#0A2A58",
    "Innovation and Economic Transformation": "#0A2A58",
    "Labour and Reskilling": "#0A2A58",
    "Technical Assistance": "#0A2A58",
    "Mobility and Infrastructure": "#0A2A58",
    "Local communities/tourism/culture and natural heritage": "#0A2A58",
    "Social Infrastructure": "#0A2A58",
    "Other": "#0A2A58"
}

selected_categories = [
    "Innovation and Economic Transformation",
    "Decarbonization",
    "Labour and Reskilling",
    "Social Infrastructure"
]

policy_order = [
    "Innovation and Economic Transformation",
    "Decarbonization",
    "Labour and Reskilling",
    "Social Infrastructure"
]


# ============================================================
# Indikatoren: Originalspalte -> x-Achsenlabel
# ============================================================
indicator_map = {
    "Index Economic Exposure": {
        "label": "Index Economic Exposure",
        "color": "#0A2A58"
    },
    "Index Socioeconomic Sensitivity": {
        "label": "Index Sensitivity",
        "color": "#510A94"
    },
    "Index Adaptive Capacity": {
        "label": "IndexAdaptive Capacity",
        "color": "#0E5B2C"
    },
    
    "Ranking Emission Intensity / GDP for selected sectors": {
        "label": "Ranking Emission Intensity",
        "color": "#0A2A58"
    },
    "Ranking National Energy Mix 2019 Share High Carbon": {
        "label": "Ranking Energy Mix",
        "color": "#0A2A58"
    },
    "Ranking Share High Carbon Employment NUTS 2 2019": {
        "label": "Ranking High Carbon Employment",
        "color": "#0A2A58"
    },
    "Ranking Share of the Population aged 50 to 64, 2019, NUTS 2": {
        "label": "Ranking Age Workforce",
        "color": "#510A94"
    },
    "Ranking Subnational HDI 2019": {
        "label": "Ranking Subnational HDI",
        "color": "#510A94"
    },
    "Ranking Regional Development Gap 2019": {
        "label": "Ranking Development Gap",
        "color": "#510A94"
    },
    "Ranking Cost of Capital Average Rank 2019": {
        "label": "Ranking Cost of Capital",
        "color": "#0E5B2C"
    },
    "Ranking Share Investment Capacity / GVA, 2019, NUTS 2": {
        "label": "Ranking Investment Capacity",
        "color": "#0E5B2C"
    }
}


# ============================================================
# Hilfsfunktion für sichere Dateinamen
# ============================================================
def safe_filename(text):
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


# ============================================================
# 1) Daten einlesen und aufbereiten
# ============================================================

# Vulnerability-Daten
vul = pd.read_excel(vul_path)
vul["NUTS 2 Code"] = vul["NUTS 2 Code"].astype(str).str.strip()
vul["Corresponding NUTS2 level"] = vul["Corresponding NUTS2 level"].astype(str).str.strip()

# Alle Indikator-Spalten numerisch machen
for col in indicator_map.keys():
    if col in vul.columns:
        vul[col] = pd.to_numeric(vul[col], errors="coerce")
    else:
        print(f"Warnung: Spalte nicht gefunden in Excel: {col}")

# JTF-Flows
flows = pd.read_csv(flows_path, dtype=str)

for col in [
    "NUTS 2 Code", "NUTS 1 Code", "ms", "dimension_type",
    "Policy Category Title", "eu_amount", "cofinancing_rate"
]:
    if col in flows.columns:
        flows[col] = flows[col].astype(str).str.strip()

# "AT1/AT2" und "HU2/HU3" leeren
if "NUTS 1 Code" in flows.columns:
    flows.loc[flows["NUTS 1 Code"].isin(["AT1/AT2", "HU2/HU3"]), "NUTS 1 Code"] = np.nan

# Nur Intervention Field
flows = flows[flows["dimension_type"] == "Intervention Field"].copy()

# Länder raus
flows = flows[~flows["ms"].isin(["PT", "FR", "CY", "NL"])].copy()

# Spalten umbenennen
flows = flows.rename(columns={
    "NUTS 2 Code": "NUTS2_code",
    "NUTS 1 Code": "NUTS1_code",
    "ms": "MS"
})

# Technical Assistance raus
flows = flows[flows["Policy Category Title"] != "Technical Assistance"].copy()

# Auf vier Policy-Kategorien einschränken
flows = flows[flows["Policy Category Title"].isin(selected_categories)].copy()

# Numerische Beträge
flows["eu_amount"] = pd.to_numeric(flows["eu_amount"], errors="coerce")
flows["cofinancing_rate"] = pd.to_numeric(flows["cofinancing_rate"], errors="coerce")

flows["eu_amount"] = flows["eu_amount"] / flows["cofinancing_rate"]
flows = flows.replace([np.inf, -np.inf], np.nan)
flows = flows.dropna(subset=["eu_amount"])

# ------------------------------------------------------------
# NUTS-Zuordnung (NUTS2 > NUTS1 > MS)
# ------------------------------------------------------------
valid_codes = set(vul["NUTS 2 Code"].tolist())

def pick_region_code(row):
    for col in ["NUTS2_code", "NUTS1_code", "MS"]:
        code = row.get(col, None)
        if pd.notna(code) and code in valid_codes:
            return code
    return np.nan

flows["Region_code"] = flows.apply(pick_region_code, axis=1)
flows = flows.dropna(subset=["Region_code"]).copy()

# ------------------------------------------------------------
# Aggregation nach Region x Policy Category
# ------------------------------------------------------------
region_totals = (
    flows.groupby("Region_code")["eu_amount"]
    .sum()
    .rename("region_total_amount")
    .reset_index()
)

agg = (
    flows.groupby(["Region_code", "Policy Category Title"], as_index=False)["eu_amount"]
    .sum()
)

agg = agg.merge(region_totals, on="Region_code", how="left")
agg["share"] = agg["eu_amount"] / agg["region_total_amount"]
agg["share_pct"] = agg["share"] * 100

# ------------------------------------------------------------
# Bubble-Skalierung (Referenz = Silesia total_amount)
# ------------------------------------------------------------
silesia_row = agg.merge(
    vul[["NUTS 2 Code", "Corresponding NUTS2 level"]],
    left_on="Region_code",
    right_on="NUTS 2 Code",
    how="left"
)
silesia_row = silesia_row[silesia_row["Corresponding NUTS2 level"] == "Silesia"]

if not silesia_row.empty:
    ref_amount = float(silesia_row["eu_amount"].iloc[0])
else:
    ref_amount = float(agg["eu_amount"].max())

def scale_bubble_size(amount, ref, base_size=300):
    if ref <= 0 or np.isnan(ref):
        return base_size
    ratio = amount / ref
    ratio = max(ratio, 0.05)
    return base_size * ratio

agg["bubble_size"] = agg["eu_amount"].apply(lambda x: scale_bubble_size(x, ref=ref_amount))


# ============================================================
# Hilfslisten für Labels / Formatierung
# ============================================================
def format_region_name(region_name: str) -> str:
    if region_name == "Lower Silesia":
        return "Lower\nSilesia"
    elif region_name == "Northern and Southeastern (BG)":
        return "Northern and\nSoutheastern\n(BG)"
    elif region_name == "Moravian-Silesian Region":
        return "Moravian\nSilesian\nRegion"
    elif region_name == "Northwest (CZ)":
        return "Northwest\n(CZ)"
    elif region_name == "Mainland (FI)":
        return "Mainland\n(FI)"
    elif region_name == "South-West Oltenia":
        return "South-West\nOltenia"
    elif region_name == "Province of Hainaut":
        return "Province\nof Hainaut"
    elif region_name == "Central and Western Lithuania":
        return "Central\nand Western\nLithuania"
    elif region_name == "Upper Norrland":
        return "Upper\nNorrland"
    else:
        return region_name


# ============================================================
# Plot-Funktion für einen Indikator
# ============================================================
def make_bubble_plot_for_indicator(indicator_col, x_label, bubble_color, outfile):
    # Merge mit Indikator
    vul_subset = vul[[
        "NUTS 2 Code",
        "Corresponding NUTS2 level",
        indicator_col
    ]].copy()

    vul_subset = vul_subset.rename(columns={
        "NUTS 2 Code": "Region_code",
        "Corresponding NUTS2 level": "Region_name",
        indicator_col: "x_value"
    })

    vul_subset["x_value"] = pd.to_numeric(vul_subset["x_value"], errors="coerce")

    data = agg.merge(vul_subset, on="Region_code", how="inner")

    # Nur NUTS1/2
    data = data[data["Region_code"].str.len() > 2].copy()

    # Formatierte Namen
    data["Region_name_formatted"] = data["Region_name"].apply(format_region_name)

    # Globale Top-3-Regionen nach aktuellem Indikator
    top3_regions = (
        data[["Region_code", "x_value"]]
        .drop_duplicates()
        .dropna(subset=["x_value"])
        .sort_values("x_value", ascending=False)
        .head(3)["Region_code"]
        .tolist()
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

    x_min = data["x_value"].min()
    x_max = data["x_value"].max()
    if np.isfinite(x_min) and np.isfinite(x_max):
        margin = 0.06 * (x_max - x_min) if x_max > x_min else 0.01
        x_limits = (x_min - margin, x_max + margin)
    else:
        x_limits = None

    for ax, policy in zip(axes, policy_order):
        subset = data[data["Policy Category Title"] == policy].copy()
        subset = subset.dropna(subset=["x_value", "share_pct"])

        n_regions = subset["Region_code"].nunique()
        color = bubble_color

        ax.scatter(
            subset["x_value"],
            subset["share_pct"],
            s=subset["bubble_size"],
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            color=color
        )

        if x_limits is not None:
            ax.set_xlim(*x_limits)

        ax.set_title(policy, fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.tick_params(axis="both", labelsize=12)


        # Regressionslinie
        x = subset["x_value"].to_numpy()
        y = subset["share_pct"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)

        stats_text = f"n = {n_regions}"

        if mask.sum() >= 2:
            m, b = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            y_line = m * x_line + b
            ax.plot(x_line, y_line, linewidth=1.0, color="black")

            y_pred = m * x[mask] + b
            ss_res = np.sum((y[mask] - y_pred) ** 2)
            ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            stats_text = (
                f"n = {n_regions}\n"
                f"β = {m:.2f}\n"
                f"R² = {r2:.2f}"
            )
        ax.text(
            0.98, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9
            )
        )



        # Labels
        top5_amount_codes = (
            subset.sort_values("eu_amount", ascending=False)
            .head(2)["Region_code"]
            .tolist()
        )
        top2_share_codes = (
            subset.sort_values("share_pct", ascending=False)
            .head(2)["Region_code"]
            .tolist()
        )

        label_codes = set(top5_amount_codes + top2_share_codes + top3_regions)

        for _, row in subset.iterrows():
            if row["Region_code"] not in label_codes:
                continue

            ax.text(
                row["x_value"],
                row["share_pct"],
                row["Region_name_formatted"],
                fontsize=8,
                ha="center",
                va="center",
                color="black"
            )

        ax.set_xlabel(x_label, fontsize=14)

    axes[0].set_ylabel("Share JTF Allocations (%)", fontsize=14)

    plt.subplots_adjust(bottom=0.25, wspace=0.12)

    # Bubble-Legende
    valid_totals = data["eu_amount"].to_numpy()
    valid_totals = valid_totals[np.isfinite(valid_totals) & (valid_totals > 0)]

    if valid_totals.size > 0:
        min_total = valid_totals.min()
    else:
        min_total = 0.25 * ref_amount

    mid_amount = 0.5 * ref_amount
    legend_amounts = [min_total, mid_amount, ref_amount]
    legend_sizes = [scale_bubble_size(a, ref=ref_amount) for a in legend_amounts]
    legend_labels = [f"{a/1e6:.1f} Mio €" for a in legend_amounts]

    legend_handles = [
        plt.scatter([], [], s=s, edgecolors="black", linewidth=0.5, alpha=0.6, color="#888888")
        for s in legend_sizes
    ]

    leg = fig.legend(
        legend_handles,
        legend_labels,
        title="JTF amount per region & policy",
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=14,
        title_fontsize=14,
        bbox_to_anchor=(0.5, -0.06)
    )

    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(1.0)

    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Alle gewünschten 1x4-PDFs mit Regression erzeugen
# ============================================================

for indicator_col, indicator_info in indicator_map.items():

    x_label = indicator_info["label"]
    bubble_color = indicator_info["color"]

    if indicator_col not in vul.columns:
        print(f"Übersprungen, da Spalte fehlt: {indicator_col}")
        continue

    outfile = os.path.join(
        output_dir,
        f"JTF_Bubbles_1x4_withRegression_{safe_filename(x_label)}.pdf"
    )

    make_bubble_plot_for_indicator(
        indicator_col=indicator_col,
        x_label=x_label,
        bubble_color=bubble_color,
        outfile=outfile
    )

    print(f"Fertig: {outfile}")