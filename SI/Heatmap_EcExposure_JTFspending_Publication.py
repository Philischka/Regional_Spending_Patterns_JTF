# GROUPED HEATMAP: Regions grouped by Economic Exposure no exports index
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

def clip_values(values, percentile=95):
    """Return max cap based on percentile to reduce influence of extreme outliers."""
    finite_vals = [v for v in values if np.isfinite(v)]
    if len(finite_vals) == 0:
        return 1.0
    return np.nanpercentile(finite_vals, percentile)

# ----------------------------------------------------------------
# INPUT PATHS (ANPASSEN!)
# ----------------------------------------------------------------
vuln_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/output/00_Publication/01_JTF_Regions_Rankings/Adjusted_for_merge/01_JRF_Region_Ranking_NUTS2_1_0.xlsx"
flows_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/04_JTF_Finances_Details/Data_used/2021-2027_JTF_Finances_Details_with_NUTS_20251115.csv"

out_dir = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/06_Shares_PolicyFields_PerDimension/01_Economic_Exposure/01_Heatmap"

# OUTPUT
grouped_dir = out_dir
os.makedirs(grouped_dir, exist_ok=True)

# ================================================================
# 1. LOAD VULNERABILITY DATA
# ================================================================
col_code = "NUTS 2 Code"
col_share = "Index Economic Exposure"

vuln = pd.read_excel(vuln_path)
vuln[col_share] = pd.to_numeric(vuln[col_share], errors="coerce")
vuln["high_carbon_jobs"] = vuln[col_share]

vuln_small = vuln[[col_code, col_share, "high_carbon_jobs"]].copy()

# ================================================================
# 2. LOAD JTF FLOWS AND KEEP 'OTHER' AGAIN
# ================================================================
flows = pd.read_csv(flows_path)
flows = flows[flows["dimension_type"] == "Intervention Field"].copy()

flows["NUTS 1 Code"] = flows["NUTS 1 Code"].replace({"AT1/AT2": np.nan, "HU2/HU3": np.nan})
flows = flows[~flows["ms"].isin(["PT", "FR"])].copy()

flows["Policy Category Title"] = flows["Policy Category Title"].astype(str).str.strip().str.title()

# REMOVE ONLY "Technical Assistance"
flows = flows[flows["Policy Category Title"].str.lower() != "technical assistance"].copy()

for col in ["cofinancing_rate", "eu_amount"]:
    flows[col] = pd.to_numeric(flows[col], errors="coerce")

flows["total_amount"] = flows["eu_amount"] / flows["cofinancing_rate"]

# ================================================================
# 3. MATCH REGIONS (NUTS2 → NUTS1 → MS)
# ================================================================
m1 = flows.merge(vuln_small, left_on="NUTS 2 Code", right_on=col_code, how="left")
matched = m1[~m1["high_carbon_jobs"].isna()].copy()
unmatched = m1[m1["high_carbon_jobs"].isna()].copy()
unmatched = unmatched.drop(columns=[col_code, col_share, "high_carbon_jobs"])

m2 = unmatched.merge(vuln_small, left_on="NUTS 1 Code", right_on=col_code, how="left")
matched = pd.concat([matched, m2[~m2["high_carbon_jobs"].isna()]], ignore_index=True)
unmatched = m2[m2["high_carbon_jobs"].isna()].copy()
unmatched = unmatched.drop(columns=[col_code, col_share, "high_carbon_jobs"])

m3 = unmatched.merge(vuln_small, left_on="ms", right_on=col_code, how="left")
matched = pd.concat([matched, m3[~m3["high_carbon_jobs"].isna()]], ignore_index=True)

matched = matched.rename(columns={col_code: "region_code"})

matched = matched[matched["region_code"].astype(str).str.len() > 2]
matched = matched[matched["high_carbon_jobs"] > 0]

# ================================================================
# 4. AGGREGATE FLOWS BY REGION × POLICY CATEGORY
# ================================================================
group = (
    matched.groupby(["region_code", "Policy Category Title", col_share], as_index=False)
    .agg(total_amount_sum=("total_amount", "sum"))
)

# ================================================================
# 5. REGION TOTALS + PERCENT SHARES PER POLICY
# ================================================================
region_totals = (
    group.groupby("region_code", as_index=False)
    .agg(region_total=("total_amount_sum", "sum"))
)

group = group.merge(region_totals, on="region_code", how="left")
group["pct_share"] = 100 * group["total_amount_sum"] / group["region_total"]

# ================================================================
# 6. GROUP REGIONS INTO QUANTILE BUCKETS (25%-Version)
# ================================================================
region_shares = (
    group[["region_code", col_share]].drop_duplicates()
)

quantiles = region_shares[col_share].quantile([0.25, 0.5, 0.75, 0.9])

def assign_group(x):
    if x <= quantiles.iloc[0]:
        return "0–25%"
    elif x <= quantiles.iloc[1]:
        return "25–50%"
    elif x <= quantiles.iloc[2]:
        return "50–75%"
    elif x <= quantiles.iloc[3]:
        return "75–90%"
    else:
        return "90–100%"

region_shares["group"] = region_shares[col_share].apply(assign_group)

group = group.merge(region_shares[["region_code", "group"]], on="region_code", how="left")

# ================================================================
# 7. DEFINE POLICY CATEGORY ORDER
# ================================================================
policy_order_7 = [
    "Innovation And Economic Transformation",
    "Decarbonization",
    "Labour And Reskilling",
    "Social Infrastructure",
    "Local Communities/Tourism/Culture And Natural Heritage",
    "Mobility And Infrastructure",
    "Other"
]
# Mehrzeilige Labels für die Tabellenüberschriften
col_label_map = {
    "Innovation And Economic Transformation": "Innovation And\nEconomic\nTransformation",
    "Decarbonization": "Decarbon-\nization",
    "Labour And Reskilling": "Labour And\nReskilling",
    "Social Infrastructure": "Social\nInfrastructure",
    "Local Communities/Tourism/Culture And Natural Heritage":
        "Local Communi-\nties/Tourism/\nCulture & Nat.\nHeritage",
    "Mobility And Infrastructure": "Mobility And\nInfrastructure",
    "Other": "Other",
    "Avg. # Policies": "Avg. #\nPolicies",
    "Avg Total € (M)": "Avg Total\n€ (M)"
}
# ensure missing categories exist
for cat in policy_order_7:
    if cat not in group["Policy Category Title"].unique():
        group.loc[len(group)] = [None, cat, np.nan, 0, None, "FILLER"]

# ================================================================
# 8. MEAN + MEDIAN PER GROUP & POLICY (in one cell)
# ================================================================
table = (
    group.groupby(["group", "Policy Category Title"])
    .agg(
        mean_pct=("pct_share", "mean"),
        avg_total=("total_amount_sum", "mean")
    )
    .reset_index()
)

def format_cell(mean_pct, avg_total):
    if np.isnan(mean_pct) and np.isnan(avg_total):
        return "Mean: 0%\nAvg €: 0 M"
    
    avg_million = avg_total / 1_000_000
    avg_million_rounded = round(avg_million, 1)

    return f"Mean: {mean_pct:.1f}%\nAvg €: {avg_million_rounded} M"

table["cell"] = table.apply(
    lambda r: format_cell(r["mean_pct"], r["avg_total"]), axis=1
)

pivot = table.pivot(index="group", columns="Policy Category Title", values="cell")
pivot = pivot.reindex(index=["0–25%", "25–50%", "50–75%", "75–90%", "90–100%"])
pivot = pivot[policy_order_7]

pivot = pivot.fillna("Mean: 0%\nAvg €: 0 M")

# ================================================================
# 9. NUMBER OF POLICY CATEGORIES PER REGION & GROUP
# ================================================================
policy_presence = (
    group.assign(has_amount=group["total_amount_sum"] > 0)
    .groupby(["region_code", "Policy Category Title"])
    .agg(has_amount=("has_amount", "sum"))
)

policy_presence = (
    policy_presence.reset_index()
    .pivot(index="region_code", columns="Policy Category Title", values="has_amount")
    .fillna(0)
)

policy_presence["num_invested"] = (policy_presence > 0).sum(axis=1)

policy_group = (
    policy_presence.merge(region_shares, on="region_code")
    .groupby("group", as_index=False)
    .agg(avg_num_policies=("num_invested", "mean"))
)

policy_group["avg_num_policies"] = policy_group["avg_num_policies"].round(2)

# ============================================================
# NEW: Average Total Amount (Millionen €) pro Gruppe
# ============================================================
avg_total_group = (
    group.groupby("group", as_index=False)
    .agg(avg_total_million=("total_amount_sum", lambda x: x.mean() / 1_000_000))
)

avg_total_group["avg_total_million"] = avg_total_group["avg_total_million"].round(1)

pivot["Avg. # Policies"] = policy_group.set_index("group")["avg_num_policies"]
pivot["Avg. # Policies"] = pivot["Avg. # Policies"].fillna(0)

pivot["Avg Total € (M)"] = avg_total_group.set_index("group")["avg_total_million"]
pivot["Avg Total € (M)"] = pivot["Avg Total € (M)"].fillna(0)

# ================================================================
# 10. CREATE HEATMAP (25%-Gruppen)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))

cell_text = []
for row in pivot.index:
    row_vals = pivot.loc[row, policy_order_7].tolist()
    row_vals += [
        f"{pivot.loc[row, 'Avg. # Policies']:.2f}",
        f"{pivot.loc[row, 'Avg Total € (M)']:.1f}"
    ]
    cell_text.append(row_vals)

columns_for_table = policy_order_7 + ["Avg. # Policies", "Avg Total € (M)"]
col_display = [col_label_map.get(c, c) for c in columns_for_table]

tbl = ax.table(
    cellText=cell_text,
    rowLabels=pivot.index.tolist(),
    colLabels=col_display,          # -> mehrzeilige Labels
    cellLoc="center",
    loc="center"
)


ax.axis("off")

for key, cell in tbl.get_celld().items():
    row, col = key
    cell.set_edgecolor("black")
    if row == 0:
        cell.set_height(0.3)
        cell.set_fontsize(10)
        cell.set_text_props(weight="bold")
    else:
        cell.set_height(0.2)

for key, cell in tbl.get_celld().items():
    cell.set_width(0.1)
    cell.set_height(0.18)

tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

# ============================================================
# COLORING (25%-Heatmap)
# ============================================================

header_cells = [(col, cell) for (row, col), cell in tbl.get_celld().items() if row == 0]
header_cells = sorted(header_cells, key=lambda x: x[0])
col_pos = {pivot.columns[j]: header_cells[j][0] for j in range(len(pivot.columns))}

# 1) Blaue Skala für Policy-Felder (avg_total)
purple_vals = table["avg_total"].values
purple_min = np.nanmin(purple_vals) if np.isfinite(purple_vals).any() else 0
purple_cap = clip_values(purple_vals, percentile=95)

def purple_color_clipped(val):
    if val is None or np.isnan(val) or val <= 0 or purple_cap == purple_min:
        return "#ffffff"
    val_c = min(val, purple_cap)
    norm = (val_c - purple_min) / (purple_cap - purple_min + 1e-9)
    return plt.cm.Blues(norm)

def purple_color(val):
    if val is None or np.isnan(val) or val <= 0 or purple_cap == purple_min:
        return "#ffffff"
    norm = (val - purple_min) / (purple_cap - purple_min + 1e-9)
    return plt.cm.Blues(norm)

avg_matrix = table.pivot(index="group", columns="Policy Category Title", values="avg_total")

# 2) Graue Skala für Summary-Spalten
summary_cols = ["Avg. # Policies", "Avg Total € (M)"]
grey_vals = pivot[summary_cols].values.astype(float)
grey_min = np.nanmin(grey_vals) if np.isfinite(grey_vals).any() else 0
grey_max = np.nanmax(grey_vals) if np.isfinite(grey_vals).any() else 1

def grey_color(val):
    if val is None or np.isnan(val) or grey_max == grey_min:
        return "#ffffff"
    norm = (val - grey_min) / (grey_max - grey_min + 1e-9)
    return plt.cm.Greys(norm)

def set_text_color(cell, facecolor):
    rgba = mcolors.to_rgba(facecolor)
    r, g, b, a = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cell.get_text().set_color("white" if luminance < 0.5 else "black")

# 4) Policy-Zellen einfärben (blau)
policy_cols = policy_order_7
for row_idx, group_label in enumerate(pivot.index):
    for cat in policy_cols:
        if cat not in pivot.columns:
            continue
        table_col = col_pos[cat]
        cell = tbl[(row_idx + 1, table_col)]

        val = np.nan
        if group_label in avg_matrix.index and cat in avg_matrix.columns:
            val = avg_matrix.loc[group_label, cat]

        face = purple_color(val)
        cell.set_facecolor(face)
        set_text_color(cell, face)

# 5) Summary-Spalten einfärben (grau)
for row_idx, group_label in enumerate(pivot.index):
    for col_name in summary_cols:
        table_col = col_pos[col_name]
        cell = tbl[(row_idx + 1, table_col)]
        val = float(pivot.loc[group_label, col_name])
        face = grey_color(val)
        cell.set_facecolor(face)
        set_text_color(cell, face)

plt.tight_layout()
plt.savefig(os.path.join(grouped_dir, "grouped_policy_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 10b. CREATE HEATMAP (25%-Gruppen) – Farbe = Mean Share
# ================================================================
fig_s, ax_s = plt.subplots(figsize=(12, 6))

# gleiche Struktur wie oben
columns_for_table = policy_order_7 + ["Avg. # Policies", "Avg Total € (M)"]
col_display = [col_label_map.get(c, c) for c in columns_for_table]

cell_text = []
for row in pivot.index:
    row_vals = pivot.loc[row, policy_order_7].tolist()
    row_vals += [
        f"{pivot.loc[row, 'Avg. # Policies']:.2f}",
        f"{pivot.loc[row, 'Avg Total € (M)']:.1f}"
    ]
    cell_text.append(row_vals)

tbl_s = ax_s.table(
    cellText=cell_text,
    rowLabels=pivot.index.tolist(),
    colLabels=col_display,
    cellLoc="center",
    loc="center"
)

ax_s.axis("off")

# Rahmen, Höhe, Schrift
for (row, col), cell in tbl_s.get_celld().items():
    cell.set_edgecolor("black")
    if row == 0:
        cell.set_height(0.3)
        cell.set_fontsize(10)
        cell.set_text_props(weight="bold")
    else:
        cell.set_height(0.2)

# Breite
for _, cell in tbl_s.get_celld().items():
    cell.set_width(0.14)

tbl_s.auto_set_font_size(False)
tbl_s.set_fontsize(8)

# Spaltenpositionen rekonstruieren (wie oben)
header_cells_s = [(col, cell) for (row, col), cell in tbl_s.get_celld().items() if row == 0]
header_cells_s = sorted(header_cells_s, key=lambda x: x[0])
col_pos_s = {columns_for_table[j]: header_cells_s[j][0] for j in range(len(columns_for_table))}

# NEU: Farbe basiert auf MEAN SHARE (mean_pct)
vals_share = table["mean_pct"].values
share_min = np.nanmin(vals_share) if np.isfinite(vals_share).any() else 0
share_cap = clip_values(vals_share, percentile=95)

def blue_color_share(val):
    if val is None or np.isnan(val) or share_cap == share_min:
        return "#ffffff"
    # val ist in %, typischerweise 0–100
    val_c = min(val, share_cap)
    norm = (val_c - share_min) / (share_cap - share_min + 1e-9)
    return plt.cm.Blues(norm)

# Pivot für MEAN SHARE statt AVG TOTAL
avg_matrix_share = table.pivot(index="group",
                               columns="Policy Category Title",
                               values="mean_pct")

# Summary-Spalten weiterhin grau
summary_cols = ["Avg. # Policies", "Avg Total € (M)"]
grey_vals = pivot[summary_cols].values.astype(float)
grey_min = np.nanmin(grey_vals) if np.isfinite(grey_vals).any() else 0
grey_max = np.nanmax(grey_vals) if np.isfinite(grey_vals).any() else 1

def grey_color(val):
    if val is None or np.isnan(val) or grey_max == grey_min:
        return "#ffffff"
    norm = (val - grey_min) / (grey_max - grey_min + 1e-9)
    return plt.cm.Greys(norm)

# Policy-Zellen färben nach MEAN SHARE
for row_idx, group_label in enumerate(pivot.index):
    for cat in policy_order_7:
        table_col = col_pos_s[cat]
        cell = tbl_s[(row_idx + 1, table_col)]

        val = np.nan
        if group_label in avg_matrix_share.index and cat in avg_matrix_share.columns:
            val = avg_matrix_share.loc[group_label, cat]

        face = blue_color_share(val)
        cell.set_facecolor(face)
        set_text_color(cell, face)

# Summary-Spalten wie gehabt grau
for row_idx, group_label in enumerate(pivot.index):
    for col_name in summary_cols:
        table_col = col_pos_s[col_name]
        cell = tbl_s[(row_idx + 1, table_col)]
        val = float(pivot.loc[group_label, col_name])
        face = grey_color(val)
        cell.set_facecolor(face)
        set_text_color(cell, face)

plt.tight_layout()
plt.savefig(os.path.join(grouped_dir, "grouped_policy_heatmap_share.png"),
            dpi=300, bbox_inches="tight")
plt.close()


# ================================================================
# 11. 10%-GRUPPEN (DECILES) – ZWEITE HEATMAP
# ================================================================
region_shares10 = (
    group[["region_code", col_share]].drop_duplicates()
)

qs10 = region_shares10[col_share].quantile(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)

def assign_group_10(x):
    if x <= qs10.iloc[0]:
        return "0–10%"
    elif x <= qs10.iloc[1]:
        return "10–20%"
    elif x <= qs10.iloc[2]:
        return "20–30%"
    elif x <= qs10.iloc[3]:
        return "30–40%"
    elif x <= qs10.iloc[4]:
        return "40–50%"
    elif x <= qs10.iloc[5]:
        return "50–60%"
    elif x <= qs10.iloc[6]:
        return "60–70%"
    elif x <= qs10.iloc[7]:
        return "70–80%"
    elif x <= qs10.iloc[8]:
        return "80–90%"
    else:
        return "90–100%"

region_shares10["group10"] = region_shares10[col_share].apply(assign_group_10)
group10 = group.merge(region_shares10[["region_code", "group10"]],
                      on="region_code", how="left")

table10 = (
    group10.groupby(["group10", "Policy Category Title"])
    .agg(
        mean_pct=("pct_share", "mean"),
        avg_total=("total_amount_sum", "mean")
    )
    .reset_index()
)

def format_cell_10(mean_pct, avg_total):
    if np.isnan(mean_pct) and np.isnan(avg_total):
        return "Mean: 0%\nAvg €: 0 M"
    
    avg_million = avg_total / 1_000_000
    avg_million_rounded = round(avg_million, 1)

    return f"Mean: {mean_pct:.1f}%\nAvg €: {avg_million_rounded} M"

table10["cell"] = table10.apply(
    lambda r: format_cell_10(r["mean_pct"], r["avg_total"]), axis=1
)

pivot10 = table10.pivot(index="group10",
                        columns="Policy Category Title",
                        values="cell")

pivot10 = pivot10.reindex([
    "0–10%", "10–20%", "20–30%", "30–40%", "40–50%",
    "50–60%", "60–70%", "70–80%", "80–90%", "90–100%"
])

pivot10 = pivot10[policy_order_7]
pivot10 = pivot10.fillna("Mean: 0%\nAvg €: 0 M")

# NUMBER OF POLICIES PER REGION & GROUP10
policy_presence10 = (
    group10.assign(has_amount=group10["total_amount_sum"] > 0)
    .groupby(["region_code", "Policy Category Title"])
    .agg(has_amount=("has_amount", "sum"))
)

policy_presence10 = (
    policy_presence10.reset_index()
    .pivot(index="region_code", columns="Policy Category Title", values="has_amount")
    .fillna(0)
)

policy_presence10["num_invested"] = (policy_presence10 > 0).sum(axis=1)

policy_group10 = (
    policy_presence10.merge(region_shares10[["region_code", "group10"]], on="region_code")
    .groupby("group10", as_index=False)
    .agg(avg_num_policies=("num_invested", "mean"))
)

policy_group10["avg_num_policies"] = policy_group10["avg_num_policies"].round(2)

avg_total_group10 = (
    group10.groupby("group10", as_index=False)
    .agg(avg_total_million=("total_amount_sum", lambda x: x.mean() / 1_000_000))
)

avg_total_group10["avg_total_million"] = avg_total_group10["avg_total_million"].round(1)

pivot10["Avg. # Policies"] = policy_group10.set_index("group10")["avg_num_policies"]
pivot10["Avg. # Policies"] = pivot10["Avg. # Policies"].fillna(0)

pivot10["Avg Total € (M)"] = avg_total_group10.set_index("group10")["avg_total_million"]
pivot10["Avg Total € (M)"] = pivot10["Avg Total € (M)"].fillna(0)

# ================================================================
# 12. CREATE HEATMAP (10%-Gruppen)
# ================================================================
fig10, ax10 = plt.subplots(figsize=(12, 6))

cell_text10 = []
for row in pivot10.index:
    row_vals = pivot10.loc[row, policy_order_7].tolist()
    row_vals += [
        f"{pivot10.loc[row, 'Avg. # Policies']:.2f}",
        f"{pivot10.loc[row, 'Avg Total € (M)']:.1f}"
    ]
    cell_text10.append(row_vals)

columns_for_table10 = policy_order_7 + ["Avg. # Policies", "Avg Total € (M)"]

columns_for_table10 = policy_order_7 + ["Avg. # Policies", "Avg Total € (M)"]
col_display10 = [col_label_map.get(c, c) for c in columns_for_table10]

tbl10 = ax10.table(
    cellText=cell_text10,
    rowLabels=pivot10.index.tolist(),
    colLabels=col_display10,        # -> mehrzeilige Labels
    cellLoc="center",
    loc="center"
)

ax10.axis("off")

# Erste Schleife: Rahmen, Höhe & Schrift
for key, cell in tbl10.get_celld().items():
    row, col = key
    cell.set_edgecolor("black")
    if row == 0:
        cell.set_height(0.3)     # Header
        cell.set_fontsize(10)
        cell.set_text_props(weight="bold")
    else:
        cell.set_height(0.1)    # höhere Zellen für 10er-Gruppen

# Zweite Schleife: nur Breite setzen, Höhe NICHT mehr anfassen
for key, cell in tbl10.get_celld().items():
    cell.set_width(0.17)
    # keine cell.set_height(...) mehr hier!


tbl10.auto_set_font_size(False)
tbl10.set_fontsize(8)

# COLORING (10%-Heatmap)
header_cells10 = [(col, cell) for (row, col), cell in tbl10.get_celld().items() if row == 0]
header_cells10 = sorted(header_cells10, key=lambda x: x[0])
col_pos10 = {columns_for_table10[j]: header_cells10[j][0] for j in range(len(columns_for_table10))}

purple_vals10 = table10["avg_total"].values
purple_min10 = np.nanmin(purple_vals10) if np.isfinite(purple_vals10).any() else 0
purple_cap10 = clip_values(purple_vals10, percentile=95)

def purple_color10(val):
    if val is None or np.isnan(val) or val <= 0 or purple_cap10 == purple_min10:
        return "#ffffff"
    norm = (val - purple_min10) / (purple_cap10 - purple_min10 + 1e-9)
    return plt.cm.Blues(norm)

avg_matrix10 = table10.pivot(index="group10", columns="Policy Category Title", values="avg_total")

summary_cols10 = ["Avg. # Policies", "Avg Total € (M)"]
grey_vals10 = pivot10[summary_cols10].values.astype(float)
grey_min10 = np.nanmin(grey_vals10) if np.isfinite(grey_vals10).any() else 0
grey_max10 = np.nanmax(grey_vals10) if np.isfinite(grey_vals10).any() else 1

def grey_color10(val):
    if val is None or np.isnan(val) or grey_max10 == grey_min10:
        return "#ffffff"
    norm = (val - grey_min10) / (grey_max10 - grey_min10 + 1e-9)
    return plt.cm.Greys(norm)

def set_text_color10(cell, facecolor):
    rgba = mcolors.to_rgba(facecolor)
    r, g, b, a = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cell.get_text().set_color("white" if luminance < 0.5 else "black")

policy_cols10 = policy_order_7
for row_idx, group_label in enumerate(pivot10.index):
    for cat in policy_cols10:
        if cat not in pivot10.columns:
            continue
        table_col = col_pos10[cat]
        cell = tbl10[(row_idx + 1, table_col)]

        val = np.nan
        if group_label in avg_matrix10.index and cat in avg_matrix10.columns:
            val = avg_matrix10.loc[group_label, cat]

        face = purple_color10(val)
        cell.set_facecolor(face)
        set_text_color10(cell, face)

for row_idx, group_label in enumerate(pivot10.index):
    for col_name_p in summary_cols10:
        table_col_p = col_pos10[col_name_p]
        val_p = float(pivot10.loc[group_label, col_name_p])
        cell_p = tbl10[(row_idx + 1, table_col_p)]
        face_p = grey_color10(val_p)
        cell_p.set_facecolor(face_p)
        set_text_color10(cell_p, face_p)

plt.tight_layout()
plt.savefig(os.path.join(grouped_dir, "grouped_policy_heatmap_10pct.png"), dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 12b. CREATE HEATMAP (10%-Gruppen) – Farbe = Mean Share
# ================================================================
fig10s, ax10s = plt.subplots(figsize=(12, 12))

columns_for_table10 = policy_order_7 + ["Avg. # Policies", "Avg Total € (M)"]
col_display10 = [col_label_map.get(c, c) for c in columns_for_table10]

cell_text10 = []
for row in pivot10.index:
    row_vals = pivot10.loc[row, policy_order_7].tolist()
    row_vals += [
        f"{pivot10.loc[row, 'Avg. # Policies']:.2f}",
        f"{pivot10.loc[row, 'Avg Total € (M)']:.1f}"
    ]
    cell_text10.append(row_vals)

tbl10s = ax10s.table(
    cellText=cell_text10,
    rowLabels=pivot10.index.tolist(),
    colLabels=col_display10,
    cellLoc="center",
    loc="center"
)

ax10s.axis("off")

# Rahmen & Höhe
for (row, col), cell in tbl10s.get_celld().items():
    cell.set_edgecolor("black")
    if row == 0:
        cell.set_height(0.2)
        cell.set_fontsize(10)
        cell.set_text_props(weight="bold")
    else:
        cell.set_height(0.15)

# Breite
for _, cell in tbl10s.get_celld().items():
    cell.set_width(0.14)

tbl10s.auto_set_font_size(False)
tbl10s.set_fontsize(9)

# Spaltenpositionen
header_cells10s = [(col, cell) for (row, col), cell in tbl10s.get_celld().items() if row == 0]
header_cells10s = sorted(header_cells10s, key=lambda x: x[0])
col_pos10s = {columns_for_table10[j]: header_cells10s[j][0] for j in range(len(columns_for_table10))}

# NEU: Farbe nach MEAN SHARE
vals_share10 = table10["mean_pct"].values
share_min10 = np.nanmin(vals_share10) if np.isfinite(vals_share10).any() else 0
share_cap10 = clip_values(vals_share10, percentile=95)

def blue_color_share10(val):
    if val is None or np.isnan(val) or share_cap10 == share_min10:
        return "#ffffff"
    val_c = min(val, share_cap10)
    norm = (val_c - share_min10) / (share_cap10 - share_min10 + 1e-9)
    return plt.cm.Blues(norm)

avg_matrix10_share = table10.pivot(index="group10",
                                   columns="Policy Category Title",
                                   values="mean_pct")

summary_cols10 = ["Avg. # Policies", "Avg Total € (M)"]
grey_vals10 = pivot10[summary_cols10].values.astype(float)
grey_min10 = np.nanmin(grey_vals10) if np.isfinite(grey_vals10).any() else 0
grey_max10 = np.nanmax(grey_vals10) if np.isfinite(grey_vals10).any() else 1

def grey_color10(val):
    if val is None or np.isnan(val) or grey_max10 == grey_min10:
        return "#ffffff"
    norm = (val - grey_min10) / (grey_max10 - grey_min10 + 1e-9)
    return plt.cm.Greys(norm)

def set_text_color10(cell, facecolor):
    rgba = mcolors.to_rgba(facecolor)
    r, g, b, a = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cell.get_text().set_color("white" if luminance < 0.5 else "black")

# Policy-Zellen nach MEAN SHARE
for row_idx, group_label in enumerate(pivot10.index):
    for cat in policy_order_7:
        table_col = col_pos10s[cat]
        cell = tbl10s[(row_idx + 1, table_col)]

        val = np.nan
        if group_label in avg_matrix10_share.index and cat in avg_matrix10_share.columns:
            val = avg_matrix10_share.loc[group_label, cat]

        face = blue_color_share10(val)
        cell.set_facecolor(face)
        set_text_color10(cell, face)

# Summary-Spalten grau
for row_idx, group_label in enumerate(pivot10.index):
    for col_name_p in summary_cols10:
        table_col_p = col_pos10s[col_name_p]
        val_p = float(pivot10.loc[group_label, col_name_p])
        cell_p = tbl10s[(row_idx + 1, table_col_p)]
        face_p = grey_color10(val_p)
        cell_p.set_facecolor(face_p)
        set_text_color10(cell_p, face_p)

plt.tight_layout()
plt.savefig(os.path.join(grouped_dir, "grouped_policy_heatmap_10pct_share.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 13. TABELLE MIT DEZILEN UND DURCHSCHNITTLICHEN POLICIES
# ================================================================

# Wir erstellen eine neue Tabelle mit den Dezilen und der durchschnittlichen Anzahl an investierten Policy-Feldern

table_deciles = policy_group10[["group10", "avg_num_policies"]].copy()
table_deciles["avg_num_policies"] = table_deciles["avg_num_policies"].round(2)

# Wir fügen eine Funktion hinzu, um die Zellen basierend auf dem Wert in der rechten Spalte zu färben
def colorize_right_column(value):
    # Farbskala basierend auf dem Wert der rechten Spalte
    return plt.cm.Blues((value - table_deciles["avg_num_policies"].min()) / 
                        (table_deciles["avg_num_policies"].max() - table_deciles["avg_num_policies"].min()))

# Tabelle zur Darstellung
fig_deciles, ax_deciles = plt.subplots(figsize=(6, 4))

# Zelltext vorbereiten
cell_text_deciles = []
for row in table_deciles.itertuples():
    cell_text_deciles.append([row.group10, f"{row.avg_num_policies:.2f}"])

# Tabelle erzeugen
tbl_deciles = ax_deciles.table(
    cellText=cell_text_deciles,
    colLabels=["Decile", "Avg. # Policies"],
    cellLoc="center",
    loc="center"
)

# Farben nur für die rechte Spalte setzen
for (row, col), cell in tbl_deciles.get_celld().items():
    if row > 0:  # Kopfzeile überspringen
        if col == 1:  # Nur die rechte Spalte (Index 1)
            value = table_deciles.iloc[row - 1]["avg_num_policies"]
            facecolor = colorize_right_column(value)
            cell.set_facecolor(facecolor)
            set_text_color(cell, facecolor)

# Achse ausblenden
ax_deciles.axis("off")

# Schriftgröße und Zellhöhen anpassen
for key, cell in tbl_deciles.get_celld().items():
    cell.set_edgecolor("black")
    if key[0] == 0:  # Kopfzeile
        cell.set_fontsize(10)
        cell.set_text_props(weight="bold")
    else:
        cell.set_fontsize(8)
    cell.set_height(0.25)

# Layout anpassen und speichern
plt.tight_layout()
plt.savefig(os.path.join(grouped_dir, "deciles_avg_policies_table_colored_right.png"), dpi=300, bbox_inches="tight")
plt.close()
# ============================================
# PRINT: Regionen nach 10%-Gruppen
# ============================================
regions_by_group10 = (
    region_shares10[["region_code", col_share, "group10"]]
    .sort_values(["group10", col_share])
)

print("\n=== Regionen nach 10%-Gruppen (nach Economic-Exposure-Index sortiert) ===")
for grp, df_grp in regions_by_group10.groupby("group10"):
    print(f"\nGruppe {grp}:")
    for _, row in df_grp.iterrows():
        share_pct = row[col_share] * 100
        print(f"  {row['region_code']}: {share_pct:.2f} %")
