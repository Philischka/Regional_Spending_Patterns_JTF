import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter

# -----------------------------
# 1) Paths (as provided)
# -----------------------------
INDICATORS_XLSX = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/output/00_Publication/01_JTF_Regions_Rankings/Adjusted_for_merge/01_JRF_Region_Ranking_NUTS2_1_0.xlsx"
FLOWS_CSV = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/04_JTF_Finances_Details/Data_used/2021-2027_JTF_Finances_Details_with_NUTS_20251115.csv"
OUTPUT_DIR = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/05_Allocations_perDimension"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 2) Columns / indicators
# -----------------------------
REGION_CODE_COL_EXCEL = "NUTS 2 Code"
REGION_NAME_COL_EXCEL = "Corresponding NUTS2 level"

FILTER_DIMENSION_TYPE_COL = "dimension_type"
FILTER_DIMENSION_TYPE_VAL = "Intervention Field"

MS_COL = "ms"
NUTS2_COL_CSV = "NUTS 2 Code"
NUTS1_COL_CSV = "NUTS 1 Code"

COFIN_RATE_COL = "cofinancing_rate"
EU_AMOUNT_COL = "eu_amount"

INDICATORS = [
    "Index Economic Exposure",
    "Index Socioeconomic Sensitivity",
    "Index Adaptive Capacity",
]

COUNTRIES_TO_EXCLUDE_MS = {"NL", "PT", "CY"}

# -----------------------------
# 3) Helpers
# -----------------------------
def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace(" ", "", regex=False)
         .replace({"nan": np.nan, "": np.nan, "None": np.nan}),
        errors="coerce"
    )

def safe_filename(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").replace("__", "_")

def euro_formatter(x, pos):
    if x >= 1e9:
        return f"{x/1e9:.1f} bn €"
    elif x >= 1e6:
        return f"{x/1e6:.0f} m €"
    else:
        return f"{x:.0f} €"

def compute_sizes(values, v_ref, min_size=20, max_size=3000):
    """
    Marker area s (pt^2) is scaled linearly with values, using a fixed reference max v_ref.
    This guarantees consistency between plot and legend.
    """
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v) & (v > 0), v, 0.0)

    if v_ref <= 0:
        return np.full_like(v, min_size, dtype=float)

    s = (v / v_ref) * max_size
    s = np.where(v > 0, np.maximum(s, min_size), min_size)
    return s

def run_ols(d: pd.DataFrame, x_col: str, y_col: str):
    X = sm.add_constant(d[x_col])
    y = d[y_col]
    return sm.OLS(y, X).fit()

def add_size_legend_inside_ul(ax, scatter, reference_values, v_ref, legend_title,
                             min_size=20, max_size=3000):
    ref_vals = np.array(reference_values, dtype=float)
    s_leg = compute_sizes(ref_vals, v_ref=v_ref, min_size=min_size, max_size=max_size)

    face = scatter.get_facecolors()
    color = face[0] if len(face) > 0 else "0.5"

    handles = [
        ax.scatter(
            [], [],
            s=s,
            facecolors="0.75",   # helles Grau innen
            edgecolors="0.5",   # dunkler Rand
            linewidths=0.8,
            alpha=1.0,
            label=euro_formatter(v, None)
        )
        for s, v in zip(s_leg, ref_vals)
    ]


    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        frameon=True,
        scatterpoints=1,

        # spacing / box feel
        borderpad=1.2,       # mehr Innenabstand -> Box wirkt größer
        labelspacing=2.0,    # vertikaler Abstand zwischen Einträgen

        # move text right + widen handle area
        handletextpad=1.4,   # Text weiter weg von den Bubbles
        handlelength=2.2     # mehr "Handle-Breite" -> Box wirkt breiter
    )

    return leg


# -----------------------------
# 4) Load data
# -----------------------------
df_ind = pd.read_excel(INDICATORS_XLSX, dtype=str)
df_flow = pd.read_csv(FLOWS_CSV, dtype=str)

# -----------------------------
# 5) Clean indicators
# -----------------------------
df_ind[REGION_CODE_COL_EXCEL] = df_ind[REGION_CODE_COL_EXCEL].astype(str).str.strip()
df_ind[REGION_NAME_COL_EXCEL] = df_ind[REGION_NAME_COL_EXCEL].astype(str).str.strip()

for col in INDICATORS:
    if col not in df_ind.columns:
        raise KeyError(f"Indicator column missing in Excel: '{col}'")
    df_ind[col] = to_float_series(df_ind[col])

# -----------------------------
# 6) Clean flows
# -----------------------------
for c in [MS_COL, NUTS2_COL_CSV, NUTS1_COL_CSV, FILTER_DIMENSION_TYPE_COL]:
    if c in df_flow.columns:
        df_flow[c] = df_flow[c].astype(str).str.strip()

df_flow[NUTS1_COL_CSV] = df_flow[NUTS1_COL_CSV].replace({"AT1/AT2": "", "HU2/HU3": ""})
df_flow[NUTS1_COL_CSV] = df_flow[NUTS1_COL_CSV].replace({"": np.nan})

df_flow = df_flow[df_flow[FILTER_DIMENSION_TYPE_COL] == FILTER_DIMENSION_TYPE_VAL].copy()
df_flow = df_flow[~df_flow[MS_COL].isin(COUNTRIES_TO_EXCLUDE_MS)].copy()

df_flow[EU_AMOUNT_COL] = to_float_series(df_flow[EU_AMOUNT_COL])
df_flow[COFIN_RATE_COL] = to_float_series(df_flow[COFIN_RATE_COL])

mask_pct = df_flow[COFIN_RATE_COL] > 1
df_flow.loc[mask_pct, COFIN_RATE_COL] = df_flow.loc[mask_pct, COFIN_RATE_COL] / 100.0

df_flow["total_amount"] = df_flow[EU_AMOUNT_COL] / df_flow[COFIN_RATE_COL]

# -----------------------------
# 7) Assign each row to exactly one final code: NUTS2 else NUTS1; drop NUTS0
# -----------------------------
nuts2_valid = df_flow[NUTS2_COL_CSV].notna() & (df_flow[NUTS2_COL_CSV] != "") & (df_flow[NUTS2_COL_CSV].str.lower() != "nan")
nuts1_valid = df_flow[NUTS1_COL_CSV].notna() & (df_flow[NUTS1_COL_CSV] != "") & (df_flow[NUTS1_COL_CSV].str.lower() != "nan")

df_flow["region_code_final"] = np.where(
    nuts2_valid, df_flow[NUTS2_COL_CSV],
    np.where(nuts1_valid, df_flow[NUTS1_COL_CSV], np.nan)
)

df_flow = df_flow[df_flow["region_code_final"].notna()].copy()
df_flow["region_code_final"] = df_flow["region_code_final"].astype(str).str.strip()
df_flow = df_flow[df_flow["region_code_final"].str.len().isin([3, 4])].copy()

flows_by_region = (
    df_flow.groupby("region_code_final", as_index=False)
           .agg(
               eu_amount_sum=(EU_AMOUNT_COL, "sum"),
               total_amount_sum=("total_amount", "sum")
           )
)

# -----------------------------
# 8) Merge
# -----------------------------
df = df_ind.rename(columns={REGION_CODE_COL_EXCEL: "region_code", REGION_NAME_COL_EXCEL: "region_name"}).copy()
df["region_code"] = df["region_code"].astype(str).str.strip()

df = df.merge(flows_by_region, left_on="region_code", right_on="region_code_final", how="left")
df.drop(columns=["region_code_final"], inplace=True)

df["match_level"] = np.where(df["region_code"].str.len() == 4, "NUTS2",
                      np.where(df["region_code"].str.len() == 3, "NUTS1", np.nan))


MANUAL_LABEL_BREAKS = {
    "Lower Silesia": "Lower\nSilesia",
    "Northern and Southeastern (BG)": "Northern and\nSoutheastern\n(BG)",
    "Moravian-Silesian Region": "Moravian\nSilesian\nRegion",
    "Mainland (FI)": "Mainland\n(FI)",
    "Northwest (CZ)": "Northwest\n(CZ)",
    "Province of Hainaut": "Province\nof Hainaut",
    "Central and Western Lithuania": "Central\nand Western\nLithuania",
    "South-West Oltenia": "South-West\nOltenia",
    "Eastern (SI)": "Eastern\n(SI)",
}
# -----------------------------
# 8b) Export plot data to Excel (inputs used for figures)
# -----------------------------
export_xlsx = os.path.join(OUTPUT_DIR, "figure_data_export.xlsx")

plot_cols = ["region_code", "region_name", "match_level"] + INDICATORS + ["eu_amount_sum", "total_amount_sum"]
df_plot = df[plot_cols].copy()

with pd.ExcelWriter(export_xlsx, engine="openpyxl") as writer:
    df_plot.to_excel(writer, sheet_name="plot_data_merged", index=False)
    flows_by_region.to_excel(writer, sheet_name="flows_by_region", index=False)
    df_ind.to_excel(writer, sheet_name="indicators_raw_cleaned", index=False)
    df_flow.to_excel(writer, sheet_name="flows_raw_filtered", index=False)

print("Saved figure data Excel to:", export_xlsx)

# -----------------------------
# 9) Plot + OLS (consistent size scaling, legend outside, raise y-limit)
# -----------------------------

INDICATOR_STYLE = {
    "Index Economic Exposure": {"label": "Economic Exposure", "color": "#0A2A58"},
    "Index Socioeconomic Sensitivity": {"label": "Sensitivity", "color": "#510A94"},
    "Index Adaptive Capacity": {"label": "Adaptive Capacity", "color": "#0E5B2C"},
}


def bubbleplot_ax(
    ax,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    ols_results: list,
    legend_values: list,
    size_ref_value: float,
    color: str,
    show_size_legend: bool,
    show_y_axis: bool,
    y_lim: tuple,
    y_pad_ratio: float = 0.18,
    min_size: int = 10,
    max_size: int = 1800
):
    d = data[[x_col, y_col, "region_code", "region_name"]].copy().dropna(subset=[x_col, y_col])
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col])]

    if d.empty:
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        return

    sizes = compute_sizes(d[y_col].to_numpy(), v_ref=size_ref_value, min_size=min_size, max_size=max_size)

    sc = ax.scatter(
        d[x_col], d[y_col],
        s=sizes,
        alpha=0.55,
        facecolors=color,
        edgecolors="0.2",
        linewidths=0.6,
        clip_on=False
    )



    # Labels: extremes
    top_x = d.nlargest(6, x_col)
    top_y = d.nlargest(6, y_col)
    d_lab = pd.concat([top_x, top_y], axis=0).drop_duplicates(subset=["region_code"])

    for _, r in d_lab.iterrows():
        label = str(r["region_name"])
        label = MANUAL_LABEL_BREAKS.get(label, label)
        ax.annotate(
            text=label,
            xy=(r[x_col], r[y_col]),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            alpha=0.85
        )

    # OLS
    model = run_ols(d, x_col, y_col)
    ols_results.append({
        "x_variable": x_col,
        "y_variable": y_col,
        "beta": float(model.params.get(x_col, np.nan)),
        "intercept": float(model.params.get("const", np.nan)),
        "std_error": float(model.bse.get(x_col, np.nan)),
        "p_value": float(model.pvalues.get(x_col, np.nan)),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs)
    })


    # R^2 label (top-right, small)
    ax.text(
        0.94, 0.98,
        f"$R^2$ = {model.rsquared:.2f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=12,
        alpha=0.9
    )

    x_pred = np.linspace(d[x_col].min(), d[x_col].max(), 200)
    y_pred = model.params["const"] + model.params[x_col] * x_pred
    ax.plot(x_pred, y_pred, linestyle="--", color="0.2")


    # Axis formatting
    ax.yaxis.set_major_formatter(FuncFormatter(euro_formatter))
    ax.set_ylim(*y_lim)

    # ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=16)
    # X-axis always ends at 1
    # X axis: data range slightly beyond 1 for breathing room,
    # but ticks/labels still end at 1.0
    # X axis: asymmetric breathing room (less on the left, more on the right)
    x_pad_left = 0.02
    x_pad_right = 0.05
    ax.set_xlim(-x_pad_left, 1 + x_pad_right)
    ax.set_xticks(np.linspace(0, 1, 6))  # 0.0 ... 1.0
    ax.tick_params(axis="x", labelsize=14)


    # Add a bit of horizontal padding inside the axes (keeps x-axis ending at 1)
    ax.margins(x=0.03)

    



    # Never show y-axis label
    ax.set_ylabel("")

    if not show_y_axis:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)


    ax.grid(True, alpha=0.4)

    if show_size_legend:
        add_size_legend_inside_ul(
            ax,
            scatter=sc,
            reference_values=legend_values,
            v_ref=size_ref_value,
            legend_title=f"Bubble size = {ylabel}",
            min_size=min_size,
            max_size=max_size
        )


# -----------------------------
# 10) Create plots + save OLS
#     Use fixed size reference per y-metric so legend matches plot exactly.
# -----------------------------
ols_results = []

# Reference maximum for size scaling:
# Use the global max of each y-variable across all regions (consistent across the 3 plots per metric).
EU_SIZE_REF = float(np.nanmax(df["eu_amount_sum"].to_numpy()))
TOTAL_SIZE_REF = float(np.nanmax(df["total_amount_sum"].to_numpy()))

def make_panel_3_indicators(y_col, y_label, size_ref_value, out_path_pdf):
    # Gemeinsame y-Limits für alle 3 Panels (identische Skalierung)
    y_max = float(np.nanmax(df[y_col].to_numpy()))
    y_lim = (0, y_max * (1 + 0.22))

    fig, axes = plt.subplots(ncols=3, figsize=(18, 6), sharey=True)

    for i, ind in enumerate(INDICATORS):
        style = INDICATOR_STYLE.get(ind, {"label": ind, "color": "0.5"})
        show_first = (i == 0) and (ind == "Index Economic Exposure")

        bubbleplot_ax(
            ax=axes[i],
            data=df,
            x_col=ind,
            y_col=y_col,
            title=style["label"],
            xlabel=style["label"],
            ylabel=y_label,
            ols_results=ols_results,
            legend_values=[1e8, 5e8, 1e9],
            size_ref_value=size_ref_value,
            color=style["color"],
            show_size_legend=show_first,   # nur Economic Exposure
            show_y_axis=show_first,        # nur Economic Exposure
            y_lim=y_lim,
            y_pad_ratio=0.22,
            min_size=10,
            max_size=1800
        )

    # Sauberes Alignment ohne extra rechten Rand (Legende ist im Plot)
    fig.tight_layout()
    fig.savefig(out_path_pdf, format="pdf")
    plt.close(fig)

def make_single_indicator_plot(x_col, y_col, y_label, size_ref_value, out_path_pdf):
    style = INDICATOR_STYLE.get(x_col, {"label": x_col, "color": "0.5"})

    # gleiche y-Limits wie in den Panels (konsistent)
    y_max = float(np.nanmax(df[y_col].to_numpy()))
    y_lim = (0, y_max * (1 + 0.22))

    fig, ax = plt.subplots(figsize=(6.5, 6))  # ähnlich hoch wie Panel, aber single

    is_econ = (x_col == "Index Economic Exposure")

    bubbleplot_ax(
        ax=ax,
        data=df,
        x_col=x_col,
        y_col=y_col,
        title=style["label"],
        xlabel=style["label"],
        ylabel=y_label,
        ols_results=ols_results,
        legend_values=[1e8, 5e8, 1e9],
        size_ref_value=size_ref_value,
        color=style["color"],
        show_size_legend=is_econ,   # Legende nur bei Economic Exposure
        show_y_axis=is_econ,        # Y-Achse nur bei Economic Exposure (Ticks)
        y_lim=y_lim,
        y_pad_ratio=0.22,
        min_size=10,
        max_size=1800
    )

    fig.tight_layout()
    fig.savefig(out_path_pdf, format="pdf")
    plt.close(fig)

# --- Aufrufe: ein Panel pro y-Metrik ---
ols_results = []

EU_SIZE_REF = float(np.nanmax(df["eu_amount_sum"].to_numpy()))
TOTAL_SIZE_REF = float(np.nanmax(df["total_amount_sum"].to_numpy()))

out_eu = os.path.join(OUTPUT_DIR, "bubble_panel_3indicators_Y_eu_amount_sum.pdf")
make_panel_3_indicators("eu_amount_sum", "EU amount", EU_SIZE_REF, out_eu)

out_total = os.path.join(OUTPUT_DIR, "bubble_panel_3indicators_Y_total_amount_sum.pdf")
make_panel_3_indicators("total_amount_sum", "Total amount", TOTAL_SIZE_REF, out_total)

# --- Single plots: EU amount ---
for ind in INDICATORS:
    out_single = os.path.join(OUTPUT_DIR, f"single_{safe_filename(ind)}_Y_eu_amount_sum.pdf")
    make_single_indicator_plot(ind, "eu_amount_sum", "EU amount", EU_SIZE_REF, out_single)

# --- Single plots: Total amount ---
for ind in INDICATORS:
    out_single = os.path.join(OUTPUT_DIR, f"single_{safe_filename(ind)}_Y_total_amount_sum.pdf")
    make_single_indicator_plot(ind, "total_amount_sum", "Total amount", TOTAL_SIZE_REF, out_single)


print("Done. Saved 2 panel PDFs to:", OUTPUT_DIR)


# -----------------------------
# 11) Save OLS results (CSV + full summaries)
# -----------------------------
ols_df = pd.DataFrame(ols_results)
ols_summary_csv = os.path.join(OUTPUT_DIR, "OLS_summary_table.csv")
ols_df.to_csv(ols_summary_csv, index=False)
print("Saved OLS summary table to:", ols_summary_csv)

# Full summaries as txt
for row in ols_results:
    x_col = row["x_variable"]
    y_col = row["y_variable"]
    d = df[[x_col, y_col]].dropna()
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col])]
    if d.empty:
        continue
    model = run_ols(d, x_col, y_col)
    fname = f"OLS_{safe_filename(y_col)}_on_{safe_filename(x_col)}.txt"
    with open(os.path.join(OUTPUT_DIR, fname), "w") as f:
        f.write(model.summary().as_text())
print("Saved OLS full summaries (.txt) to:", OUTPUT_DIR)

# -----------------------------
# 12) Audit export
# -----------------------------
audit_path = os.path.join(OUTPUT_DIR, "matched_regions_audit.csv")
df[["region_code", "region_name", "match_level"] + INDICATORS + ["eu_amount_sum", "total_amount_sum"]].to_csv(audit_path, index=False)
print("Saved audit table to:", audit_path)


