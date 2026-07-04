import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


# ============================================================
# Pfade ANPASSEN
# ============================================================
vul_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/output/00_Publication/01_JTF_Regions_Rankings/Adjusted_for_merge/01_JRF_Region_Ranking_NUTS2_1_0.xlsx"  # <- anpassen
flows_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/04_JTF_Finances_Details/Data_used/2021-2027_JTF_Finances_Details_with_NUTS_20251115.csv"  # <- anpassen

output_dir = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/06_Shares_PolicyFields_PerDimension/01_Economic_Exposure/00_New"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Farben und feste Einstellungen
# ============================================================
policy_colors = {
    "Decarbonization": "#0A2A58",                         # green
    "Innovation and Economic Transformation": "#0A2A58",  # blue
    "Labour and Reskilling": "#0A2A58",                   # purple
    "Technical assistance": "#0A2A58",                    # grey
    "Mobility and Infrastructure": "#0A2A58",             # orange
    "Local communities/tourism/culture and natural heritage": "#0A2A58",  # yellow
    "Social Infrastructure": "#0A2A58",
    "Other": "#0A2A58"                                    # grey light
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

policy_display_names = {
    "Decarbonization": "Decarbonisation",
}

# ============================================================
# 1) Daten einlesen und aufbereiten
# ============================================================

# Vulnerability-Daten
vul = pd.read_excel(vul_path)
vul["NUTS 2 Code"] = vul["NUTS 2 Code"].astype(str).str.strip()
vul["Corresponding NUTS2 level"] = vul["Corresponding NUTS2 level"].astype(str).str.strip()

# WICHTIG: Name der Spalte hier ggf. an deinen Excel-Header anpassen
vul["Index Economic Exposure"] = pd.to_numeric(
    vul["Index Economic Exposure"], errors="coerce"
)

# JTF-Flows (Komma als Trennzeichen)
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

# Länder NL, PT, FR raus
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
# Merge mit Economic-Exposure-Index
# ------------------------------------------------------------
vul_subset = vul[["NUTS 2 Code", "Corresponding NUTS2 level", "Index Economic Exposure"]].copy()
vul_subset = vul_subset.rename(columns={
    "NUTS 2 Code": "Region_code",
    "Corresponding NUTS2 level": "Region_name",
    "Index Economic Exposure": "Index_Economic_Exposure"
})
vul_subset["Index_Economic_Exposure"] = pd.to_numeric(
    vul_subset["Index_Economic_Exposure"], errors="coerce"
)

data = agg.merge(vul_subset, on="Region_code", how="inner")

# Nur NUTS1/2 (NUTS0 raus)
data = data[data["Region_code"].str.len() > 2].copy()


# ============================================================
# OLS-Regressionen pro Policy-Kategorie (deskriptiv)
# ============================================================

def regression_summary_statsmodels(data, policies):
    results = []

    for policy in policies:
        subset = data[data["Policy Category Title"] == policy].copy()

        subset = subset.dropna(
            subset=["Index_Economic_Exposure", "share_pct"]
        )

        if len(subset) < 3:
            continue

        X = sm.add_constant(subset["Index_Economic_Exposure"])
        y = subset["share_pct"]

        model = sm.OLS(y, X).fit()

        results.append({
            "Policy Category": policy,
            "N": int(model.nobs),
            "Intercept": model.params["const"],
            "Slope (Economic Exposure)": model.params["Index_Economic_Exposure"],
            "Std. Error (Slope)": model.bse["Index_Economic_Exposure"],
            "p-value (Slope)": model.pvalues["Index_Economic_Exposure"],
            "R_squared": model.rsquared
        })

    return pd.DataFrame(results)

# ------------------------------------------------------------
# Bubble-Skalierung (Referenz = Silesia total_amount)
# ------------------------------------------------------------
silesia_row = data[(data["Region_name"] == "Silesia")]
if not silesia_row.empty:
    ref_amount = float(silesia_row["eu_amount"].iloc[0])
else:
    ref_amount = float(data["eu_amount"].max())  # Fallback

def scale_bubble_size(amount, ref, base_size=300):
    if ref <= 0 or np.isnan(ref):
        return base_size
    ratio = amount / ref
    ratio = max(ratio, 0.05)
    return base_size * ratio

data["bubble_size"] = data["eu_amount"].apply(
    lambda x: scale_bubble_size(x, ref=ref_amount)
)

# ============================================================
# Regressions-Output für Plot und Excel erzeugen
# ============================================================
reg_results = regression_summary_statsmodels(data, policy_order)
# ============================================================
# Export: Figure input data (appendix-ready)
# ============================================================

# Spalten, die wirklich in die Figure eingehen (x, y, size, labels, grouping)
figure_cols = [
    "Region_code",
    "Region_name",
    "Policy Category Title",
    "Index_Economic_Exposure",  # x
    "share_pct",                   # y
    "eu_amount",                # bubble meaning
    "region_total_amount",         # denominator of share
    "bubble_size",                 # plotted size (optional, but reproducibility)
]

fig_data = data[figure_cols].copy()

# Optional: sort for readability
fig_data = fig_data.sort_values(
    ["Policy Category Title", "Index_Economic_Exposure"],
    ascending=[True, False]
)

excel_out = os.path.join(output_dir, "Figure_Data_Bubbleplots_EconomicExposure.xlsx")

with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
    # one sheet with everything used across all panels
    fig_data.to_excel(writer, sheet_name="ALL", index=False)

    # one sheet per policy (matches panels)
    for pol in policy_order:
        tmp = fig_data[fig_data["Policy Category Title"] == pol].copy()
        tmp.to_excel(writer, sheet_name=pol[:31], index=False)  # Excel max 31 chars

print(f"Figure data exported to: {excel_out}")

# ------------------------------------------------------------
# Hilfslisten für Labels / Formatierung
# ------------------------------------------------------------
# Globale Top-3-Regionen nach Economic-Exposure-Index (für Labels)
top3_regions = (
    data[["Region_code", "Index_Economic_Exposure"]]
    .drop_duplicates()
    .sort_values("Index_Economic_Exposure", ascending=False)
    .head(3)["Region_code"]
    .tolist()
)

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

data["Region_name_formatted"] = data["Region_name"].apply(format_region_name)


# ============================================================
# 2) Plot-Funktion
# ============================================================

def make_bubble_plot(add_regression: bool, outfile: str, layout: str = "1x4", policies=None) -> None:
    """
    Erzeugt Bubble-Scatterplots (1x4 oder 2x2)
    mit Index Economic Exposure auf der x-Achse.
    """
    # Welche Policies sollen geplottet werden?
    policy_list = policies if policies is not None else policy_order

   # Layout automatisch passend zu Anzahl Policies
    n = len(policy_list)
    if layout == "1x4":
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]
    elif layout == "2x2":
        # Für 2 Policies ist 1x2 normalerweise sinnvoller, daher hier nur nutzen, wenn du wirklich 2x2 willst.
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
    elif layout == "1x3":
        # Neuer Block für 1x3 Layout
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]
    else:
        raise ValueError("layout muss '1x4' oder '2x2' oder '1x3' sein.")

    # Dynamische x-Achse (kleiner Rand)
    x_min = data["Index_Economic_Exposure"].min()
    x_max = data["Index_Economic_Exposure"].max()
    if np.isfinite(x_min) and np.isfinite(x_max):
        margin = 0.06 * (x_max - x_min) if x_max > x_min else 0.01
        x_limits = (x_min - margin, x_max + margin)
    else:
        x_limits = None

    for ax, policy in zip(axes, policy_list):
        subset = data[data["Policy Category Title"] == policy].copy()
        n_regions = subset["Region_code"].nunique()

        color = policy_colors.get(policy, "#000000")

        # Scatterplot
        sc = ax.scatter(
            subset["Index_Economic_Exposure"],
            subset["share_pct"],
            s=subset["bubble_size"],
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            color=color
        )

        if x_limits is not None:
            ax.set_xlim(*x_limits)

        ax.set_title(policy_display_names.get(policy, policy), fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        # n oben rechts
        # n + Regressionswerte oben rechts
        stats_text = f"n = {n_regions}"

        if add_regression:
            reg_row = reg_results[reg_results["Policy Category"] == policy]

            if not reg_row.empty:
                beta = reg_row["Slope (Economic Exposure)"].iloc[0]
                r2 = reg_row["R_squared"].iloc[0]
                pval = reg_row["p-value (Slope)"].iloc[0]

                # Signifikanzsterne
                if pval < 0.01:
                    stars = "***"
                elif pval < 0.05:
                    stars = "**"
                elif pval < 0.1:
                    stars = "*"
                else:
                    stars = ""

                stats_text = (
                    f"n = {n_regions}\n"
                    f"β = {beta:.2f}{stars}\n"
                    f"R² = {r2:.2f}"
                )

        ax.text(
        0.98, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
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

        # Regressionslinie (optional)
        if add_regression:
            x = subset["Index_Economic_Exposure"].to_numpy()
            y = subset["share_pct"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 2:
                m, b = np.polyfit(x[mask], y[mask], 1)
                x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                y_line = m * x_line + b
                ax.plot(x_line, y_line, linewidth=1.0, color="black")

        # Auswahl der zu beschriftenden Regionen
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

        # Union: Top 5 amount + Top 2 share + globale Top-3 nach Economic Exposure
        label_codes = set(top5_amount_codes + top2_share_codes + top3_regions)

        # Regionennamen in der Mitte der Bubble platzieren (ohne Pfeil)
        for _, row in subset.iterrows():
            region_code = row["Region_code"]
            if region_code not in label_codes:
                continue

            display_name = row["Region_name_formatted"]
            x_val = row["Index_Economic_Exposure"]
            y_val = row["share_pct"]

            ax.text(
                x_val,
                y_val,
                display_name,
                fontsize=8,
                ha="center",
                va="center",
                color="black"
            )


    # Gemeinsame Achsentitel
    axes[0].set_ylabel("Share JTF Spending (%)", fontsize=12)
    for ax in axes:
        ax.set_xlabel("Transition exposure", fontsize=14)

    if layout == "1x4":
        plt.subplots_adjust(bottom=0.25, wspace=0.12)
    else:
        plt.subplots_adjust(bottom=0.20, wspace=0.12, hspace=0.20)

    # ------------------------------------------------------------
    # Bubble-Legende (kleinster Wert, Mitte, Referenz/Silesia)
    # ------------------------------------------------------------
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
        fontsize=13,        # Größe der Legendenlabels
        title_fontsize=13,  # Größe der Überschrift
        bbox_to_anchor=(0.5, -0.06)
    )

    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(1.0)

    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close(fig)

# ============================================================
# 4) ZUSÄTZLICH: nur 2 Scatterplots (Innovation + Decarbonization)
# ============================================================
two_policies = [
    "Innovation and Economic Transformation",
    "Decarbonization"
]

output_file_1x2 = os.path.join(
    output_dir,
    "JTF_EconomicExposure_Bubbles_IndexEconomicExposure_1x2_Innovation_Decarbonization.pdf"
)
output_file_1x2_reg = os.path.join(
    output_dir,
    "JTF_EconomicExposure_Bubbles_IndexEconomicExposure_1x2_Innovation_Decarbonization_withRegression.pdf"
)

# 5) 1x2 ohne Regression
make_bubble_plot(add_regression=False, outfile=output_file_1x2, layout="1x4", policies=two_policies)

# 6) 1x2 mit Regression
make_bubble_plot(add_regression=True, outfile=output_file_1x2_reg, layout="1x4", policies=two_policies)

print(f"Fertig! 1x2 ohne Regression: {output_file_1x2}")
print(f"Fertig! 1x2 mit Regression: {output_file_1x2_reg}")

# ============================================================
# 4) ZUSÄTZLICH: nur 3 Scatterplots (Innovation + Decarbonization + Labour and Reskilling)
# ============================================================
three_policies = [
    "Innovation and Economic Transformation",
    "Decarbonization",
    "Labour and Reskilling"
]

output_file_1x3 = os.path.join(
    output_dir,
    "JTF_EconomicExposure_Bubbles_IndexEconomicExposure_1x3_Innovation_Decarbonization_LabourReskilling.pdf"
)
output_file_1x3_reg = os.path.join(
    output_dir,
    "JTF_EconomicExposure_Bubbles_IndexEconomicExposure_1x3_Innovation_Decarbonization_LabourReskilling_withRegression.pdf"
)

# 5) 1x3 ohne Regression
make_bubble_plot(add_regression=False, outfile=output_file_1x3, layout="1x3", policies=three_policies)

# 6) 1x3 mit Regression
make_bubble_plot(add_regression=True, outfile=output_file_1x3_reg, layout="1x3", policies=three_policies)

print(f"Fertig! 1x3 ohne Regression: {output_file_1x3}")
print(f"Fertig! 1x3 mit Regression: {output_file_1x3_reg}")
# ============================================================
# 3) Plots erzeugen (4 Dateien)
# ============================================================
output_file_1x4 = os.path.join(
    output_dir,
    "JTF_EconomicExposure_Bubbles_IndexEconomicExposure_1x4.pdf"
)
output_file_1x4_reg = os.path.join(
    output_dir,
    "JTF_EconomicExposure_Bubbles_IndexEconomicExposure_1x4_withRegression.pdf"
)
output_file_2x2 = os.path.join(
    output_dir,
    "JTF_EconomicExposure_Bubbles_IndexEconomicExposure_2x2.pdf"
)
output_file_2x2_reg = os.path.join(
    output_dir,
    "JTF_EconomicExposure_Bubbles_IndexEconomicExposure_2x2_withRegression.pdf"
)

# 1) 1x4 ohne Regression
make_bubble_plot(add_regression=False, outfile=output_file_1x4, layout="1x4")

# 2) 1x4 mit Regression
make_bubble_plot(add_regression=True, outfile=output_file_1x4_reg, layout="1x4")

# 3) 2x2 ohne Regression
make_bubble_plot(add_regression=False, outfile=output_file_2x2, layout="2x2")

# 4) 2x2 mit Regression
make_bubble_plot(add_regression=True, outfile=output_file_2x2_reg, layout="2x2")

print(f"Fertig! 1x4 ohne Regression: {output_file_1x4}")
print(f"Fertig! 1x4 mit Regression: {output_file_1x4_reg}")
print(f"Fertig! 2x2 ohne Regression: {output_file_2x2}")
print(f"Fertig! 2x2 mit Regression: {output_file_2x2_reg}")
# ============================================================
# Regressions-Output erzeugen
# ============================================================



print("\nOLS-Regressionsergebnisse (deskriptiv):")
print(reg_results.round(4))

reg_results.to_excel(
    os.path.join(output_dir, "OLS_Regression_Results_EconomicExposure.xlsx"),
    index=False
)
# ============================================================
# PDF-Regressionstabelle (Appendix-ready) für "Innovation and Economic Transformation"
# ============================================================

def p_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""

def format_de(x, digits=2):
    """Format number with comma decimal separator for German thesis style."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{digits}f}".replace(".", ",")


def export_decarbonization_nonlinearity_checks(data, output_dir):
    diagnostics_dir = os.path.join(output_dir, "Decarbonization_nonlinearity_check")
    os.makedirs(diagnostics_dir, exist_ok=True)

    decarb = data[data["Policy Category Title"] == "Decarbonization"].copy()
    decarb = decarb.dropna(subset=["Index_Economic_Exposure", "share_pct", "eu_amount"])
    decarb = decarb.sort_values("Index_Economic_Exposure").copy()

    x = decarb["Index_Economic_Exposure"]
    y = decarb["share_pct"]

    x_linear = sm.add_constant(x)
    linear_model = sm.OLS(y, x_linear).fit()

    decarb["Index_Economic_Exposure_sq"] = decarb["Index_Economic_Exposure"] ** 2
    x_quadratic = sm.add_constant(
        decarb[["Index_Economic_Exposure", "Index_Economic_Exposure_sq"]]
    )
    quadratic_model = sm.OLS(y, x_quadratic).fit()

    model_comparison = pd.DataFrame([
        {
            "model": "linear",
            "n": int(linear_model.nobs),
            "r_squared": linear_model.rsquared,
            "adj_r_squared": linear_model.rsquared_adj,
            "aic": linear_model.aic,
            "bic": linear_model.bic,
            "coef_exposure": linear_model.params["Index_Economic_Exposure"],
            "p_exposure": linear_model.pvalues["Index_Economic_Exposure"],
            "coef_exposure_squared": np.nan,
            "p_exposure_squared": np.nan,
        },
        {
            "model": "quadratic",
            "n": int(quadratic_model.nobs),
            "r_squared": quadratic_model.rsquared,
            "adj_r_squared": quadratic_model.rsquared_adj,
            "aic": quadratic_model.aic,
            "bic": quadratic_model.bic,
            "coef_exposure": quadratic_model.params["Index_Economic_Exposure"],
            "p_exposure": quadratic_model.pvalues["Index_Economic_Exposure"],
            "coef_exposure_squared": quadratic_model.params["Index_Economic_Exposure_sq"],
            "p_exposure_squared": quadratic_model.pvalues["Index_Economic_Exposure_sq"],
        },
    ])

    decarb["exposure_decile"] = pd.qcut(
        decarb["Index_Economic_Exposure"].rank(method="first"),
        q=10,
        labels=range(1, 11),
    ).astype(int)

    decile_means = (
        decarb.groupby("exposure_decile", as_index=False)
        .agg(
            n=("Region_code", "nunique"),
            exposure_min=("Index_Economic_Exposure", "min"),
            exposure_max=("Index_Economic_Exposure", "max"),
            exposure_mean=("Index_Economic_Exposure", "mean"),
            decarbonization_share_mean=("share_pct", "mean"),
            decarbonization_share_median=("share_pct", "median"),
        )
    )

    excel_path = os.path.join(
        diagnostics_dir,
        "decarbonization_nonlinearity_checks.xlsx",
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        decarb.to_excel(writer, sheet_name="plot_data", index=False)
        model_comparison.to_excel(writer, sheet_name="model_comparison", index=False)
        decile_means.to_excel(writer, sheet_name="decile_means", index=False)

    summary_path = os.path.join(
        diagnostics_dir,
        "decarbonization_nonlinearity_summary.txt",
    )
    with open(summary_path, "w") as f:
        f.write("Decarbonization nonlinearity check\n")
        f.write("===================================\n\n")
        f.write("Linear model: share_pct ~ exposure\n")
        f.write(linear_model.summary().as_text())
        f.write("\n\nQuadratic model: share_pct ~ exposure + exposure^2\n")
        f.write(quadratic_model.summary().as_text())
        f.write("\n\nModel comparison\n")
        f.write(model_comparison.to_string(index=False))

    x_grid = np.linspace(x.min(), x.max(), 200)
    linear_pred = linear_model.predict(sm.add_constant(x_grid))
    quadratic_pred = quadratic_model.predict(
        sm.add_constant(
            pd.DataFrame({
                "Index_Economic_Exposure": x_grid,
                "Index_Economic_Exposure_sq": x_grid ** 2,
            })
        )
    )
    lowess_fit = lowess(y, x, frac=0.6, it=0, return_sorted=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        decarb["Index_Economic_Exposure"],
        decarb["share_pct"],
        s=decarb["bubble_size"],
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        color=policy_colors["Decarbonization"],
    )
    ax.plot(x_grid, linear_pred, color="black", linewidth=1.2, label="OLS linear")
    ax.plot(x_grid, quadratic_pred, color="#8B1E3F", linewidth=1.4, linestyle="--", label="OLS quadratic")
    ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="#E07A00", linewidth=2.0, label="LOWESS")

    p_sq = quadratic_model.pvalues["Index_Economic_Exposure_sq"]
    stats_text = (
        f"n = {int(linear_model.nobs)}\n"
        f"Linear adj. R² = {linear_model.rsquared_adj:.3f}\n"
        f"Quadratic adj. R² = {quadratic_model.rsquared_adj:.3f}\n"
        f"p(exposure²) = {p_sq:.3f}{p_stars(p_sq)}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", linewidth=0.8, alpha=0.9),
    )

    ax.set_title("Decarbonisation")
    ax.set_xlabel("Transition exposure")
    ax.set_ylabel("Share JTF Spending (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()

    lowess_pdf = os.path.join(
        diagnostics_dir,
        "decarbonization_share_transition_exposure_lowess_quadratic.pdf",
    )
    plt.savefig(lowess_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        decile_means["exposure_decile"],
        decile_means["decarbonization_share_mean"],
        marker="o",
        linewidth=1.5,
        color=policy_colors["Decarbonization"],
    )
    ax.set_xticks(decile_means["exposure_decile"])
    ax.set_xlabel("Transition exposure decile")
    ax.set_ylabel("Mean decarbonisation share (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    decile_pdf = os.path.join(
        diagnostics_dir,
        "decarbonization_share_by_transition_exposure_decile.pdf",
    )
    plt.savefig(decile_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Decarbonization nonlinearity diagnostics exported to: {diagnostics_dir}")
    print(model_comparison.round(4))

    return {
        "diagnostics_dir": diagnostics_dir,
        "excel_path": excel_path,
        "summary_path": summary_path,
        "lowess_pdf": lowess_pdf,
        "decile_pdf": decile_pdf,
    }

def export_regression_table_pdf(data, policy, outfile_pdf):
    subset = data[data["Policy Category Title"] == policy].copy()
    subset = subset.dropna(subset=["Index_Economic_Exposure", "share_pct"])

    # OLS (bivariat)
    X = sm.add_constant(subset["Index_Economic_Exposure"])
    y = subset["share_pct"]
    model = sm.OLS(y, X).fit()

    b0 = model.params["const"]
    b1 = model.params["Index_Economic_Exposure"]
    se1 = model.bse["Index_Economic_Exposure"]
    p1 = model.pvalues["Index_Economic_Exposure"]
    r2 = model.rsquared
    n = int(model.nobs)

    stars = p_stars(p1)

    # Table content (Paper style: coef with stars; SE in parentheses below)
    table_rows = [
        ["Transition exposure", f"{format_de(b1, 2)}{stars}"],
        ["", f"({format_de(se1, 2)})"],
        ["Constant", f"{format_de(b0, 2)}"],
        ["", ""],
        ["Observations", f"{n}"],
        ["R²", f"{format_de(r2, 3)}"],
        ["Estimation", "OLS (bivariate)"],
    ]

    title = "Table A1. OLS regression – Innovation and Economic Transformation"
    subtitle = "Dependent variable: Share of JTF allocation (%)"

    notes = "Notes: Standard errors in parentheses. *** p < 0.01, ** p < 0.05, * p < 0.1."

    # Create PDF table as a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8.27, 3.6))  # ~A4 width in inches, compact height
    ax.axis("off")

    # Title / subtitle
    ax.text(0.0, 1.06, title, fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 1.00, subtitle, fontsize=14, transform=ax.transAxes)

    # Table
    col_labels = ["Variables", "Share of JTF allocation (%)"]
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        colLoc="left",
        cellLoc="left",
        loc="upper left",
        bbox=[0.0, 0.18, 1.0, 0.72],  # [left, bottom, width, height]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Make header visually distinct
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:  # header row
            cell.set_text_props(fontweight="bold")
            cell.set_linewidth(0.9)

    # Notes at bottom
    ax.text(0.0, 0.06, notes, fontsize=12, transform=ax.transAxes)

    plt.savefig(outfile_pdf, bbox_inches="tight")
    plt.close(fig)

    return model  # optional, for debugging


pdf_outfile = os.path.join(output_dir, "Table_A1_OLS_Innovation_EconomicExposure.pdf")
_ = export_regression_table_pdf(
    data=data,
    policy="Innovation and Economic Transformation",
    outfile_pdf=pdf_outfile
)

print(f"PDF-Regressions-Tabelle gespeichert: {pdf_outfile}")

decarbonization_diagnostics = export_decarbonization_nonlinearity_checks(
    data=data,
    output_dir=output_dir,
)
