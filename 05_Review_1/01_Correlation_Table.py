from pathlib import Path
import html
import shutil
import subprocess

import pandas as pd


REGION_DATA_PATH = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/"
    "output/00_Publication/01_JTF_Regions_Rankings/Adjusted_for_merge/"
    "01_JRF_Region_Ranking_NUTS2_1_0.xlsx"
)
FINANCE_DATA_PATH = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/"
    "04_JTF_Finances_Details/Data_used/2021-2027_JTF_Finances_Details_with_NUTS_20251115.csv"
)
OUTPUT_DIR = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/"
    "07_Visualizations/00_Publication/08_Review/01_Correlation_Table_GVA_Population"
)
OUTPUT_FILE = OUTPUT_DIR / "01_Correlation_Table_GVA_Population_JTF.xlsx"
COMBINED_TABLE_HTML = OUTPUT_DIR / "01_Correlation_Table_Pearson_Spearman.html"
COMBINED_TABLE_SVG = OUTPUT_DIR / "01_Correlation_Table_Pearson_Spearman.svg"
COMBINED_TABLE_PNG = OUTPUT_DIR / "01_Correlation_Table_Pearson_Spearman.png"

EXCLUDED_COUNTRIES = {"NL", "PT", "CY"}
INTERVENTION_FIELD = "Intervention Field"

# Variables for which correlations are reported in the SI table.
VARIABLES = [
    "Index High Carbon Employment",
    "Index Economic Exposure",
    "Index Adaptive Capacity",
    "Index Socioeconomic Sensitivity",
    "Population",
    "GVA",
]

DISPLAY_VARIABLE_NAMES = {
    "Index Economic Exposure": "Index Transition Exposure",
}

REFERENCE_VARIABLES = {
    "Population": "Population",
    "GVA": "GVA",
    "JTF allocation (EU amount)": "JTF allocation",
}

# Lower values are kept first when duplicate allocation source codes occur.
MATCH_PRIORITY = {
    "NUTS2": 1,
    "NUTS1": 2,
    "NUTS1 parent": 3,
    "No match": 99,
}


def to_numeric(series):
    # Convert potentially formatted strings to numeric values.
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def clean_code(series):
    # Standardize regional codes for matching.
    return series.astype("string").str.strip().str.upper()


def split_codes(value):
    # Split cells that contain more than one NUTS code, such as "HU2/HU3".
    if pd.isna(value):
        return []
    return [
        code.strip().upper()
        for code in str(value).replace(";", "/").replace(",", "/").split("/")
        if code.strip()
    ]


def read_region_data():
    # Load the vulnerability dataset and keep only NUTS1 and NUTS2 observations.
    regions = pd.read_excel(REGION_DATA_PATH)
    required_cols = [
        "Country Code",
        "Country Name",
        "Corresponding NUTS2 level",
        "NUTS 2 Code",
        "Level",
        *VARIABLES,
    ]
    missing = [col for col in required_cols if col not in regions.columns]
    if missing:
        raise KeyError("Missing columns in region data: " + ", ".join(missing))

    # Exclude NUTS0 observations and countries not considered in this robustness check.
    regions = regions[regions["Level"].isin(["NUTS2", "NUTS1"])].copy()
    regions["NUTS 2 Code"] = clean_code(regions["NUTS 2 Code"])
    regions = regions[~regions["NUTS 2 Code"].str[:2].isin(EXCLUDED_COUNTRIES)].copy()

    for col in VARIABLES:
        regions[col] = to_numeric(regions[col])

    return regions


def read_finance_data():
    # Load the JTF finance data and keep only Intervention Field observations.
    finances = pd.read_csv(FINANCE_DATA_PATH)
    required_cols = [
        "NUTS 2 Code",
        "NUTS 1 Code",
        "ms",
        "dimension_type",
        "eu_amount",
        "Policy Category Title",
    ]
    missing = [col for col in required_cols if col not in finances.columns]
    if missing:
        raise KeyError("Missing columns in finance data: " + ", ".join(missing))

    finances = finances[finances["dimension_type"] == INTERVENTION_FIELD].copy()
    finances["ms"] = clean_code(finances["ms"])
    finances = finances[~finances["ms"].isin(EXCLUDED_COUNTRIES)].copy()

    # Clean regional codes and remove grouped NUTS1 cells that should not be used for matching.
    finances["NUTS 2 Code"] = clean_code(finances["NUTS 2 Code"])
    finances["NUTS 1 Code"] = clean_code(finances["NUTS 1 Code"])
    finances.loc[finances["NUTS 1 Code"].isin(["AT1/AT2", "HU2/HU3"]), "NUTS 1 Code"] = pd.NA

    finances["eu_amount_num"] = to_numeric(finances["eu_amount"])

    return finances


def allocation_map(finances, code_col):
    # Aggregate EU allocations by the requested NUTS code column.
    return (
        finances.assign(match_code=finances[code_col].apply(split_codes))
        .explode("match_code")
        .dropna(subset=["match_code"])
        .groupby("match_code", as_index=True)["eu_amount_num"]
        .sum()
        .to_dict()
    )


def build_allocation_maps(finances):
    # Build separate lookup tables for exact NUTS2 matches and NUTS1 matches.
    return {
        "NUTS2": allocation_map(finances, "NUTS 2 Code"),
        "NUTS1": allocation_map(finances, "NUTS 1 Code"),
    }


def match_allocation(region_code, allocation_maps):
    # Match first by exact NUTS2 code, then by NUTS1 code, then by parent NUTS1 code.
    code = str(region_code).strip().upper()
    parent_nuts1_code = code[:3] if len(code) >= 3 else code

    if code in allocation_maps["NUTS2"]:
        return allocation_maps["NUTS2"][code], "NUTS2", f"NUTS2:{code}"
    if code in allocation_maps["NUTS1"]:
        return allocation_maps["NUTS1"][code], "NUTS1", f"NUTS1:{code}"
    if parent_nuts1_code in allocation_maps["NUTS1"]:
        return allocation_maps["NUTS1"][parent_nuts1_code], "NUTS1 parent", f"NUTS1:{parent_nuts1_code}"
    return pd.NA, "No match", pd.NA


def add_allocations(regions, finances):
    # Attach JTF allocations to the regional vulnerability dataset.
    allocation_maps = build_allocation_maps(finances)
    matches = regions["NUTS 2 Code"].apply(lambda code: match_allocation(code, allocation_maps))

    matched = regions.copy()
    matched["JTF allocation"] = [match[0] for match in matches]
    matched["Allocation match level"] = [match[1] for match in matches]
    matched["Allocation source code"] = [match[2] for match in matches]
    matched["JTF allocation"] = to_numeric(matched["JTF allocation"])
    excluded = matched[matched["Allocation match level"].eq("No match")].copy()
    matched = matched[~matched["Allocation match level"].eq("No match")].copy()

    # Ensure that each allocation source is used only once in the correlation analysis.
    matched["Match priority"] = matched["Allocation match level"].map(MATCH_PRIORITY)
    matched = matched.sort_values(
        by=["Allocation source code", "Match priority", "Level", "NUTS 2 Code"],
        kind="stable",
    )
    duplicate_source = matched["Allocation source code"].duplicated(keep="first")
    excluded_duplicate_sources = matched[duplicate_source].copy()
    excluded_duplicate_sources["Exclusion reason"] = "Duplicate allocation source code"
    matched = matched[~duplicate_source].copy()
    matched = matched.drop(columns=["Match priority"])

    return matched, excluded, excluded_duplicate_sources


def make_correlation_table(data, method):
    # Create the correlation table for the selected method, e.g. Pearson or Spearman.
    rows = []
    for variable in VARIABLES:
        row = {"Variable": DISPLAY_VARIABLE_NAMES.get(variable, variable)}
        for output_name, source_col in REFERENCE_VARIABLES.items():
            if variable == source_col:
                row[output_name] = 1.0
                continue
            valid = data[[variable, source_col]].dropna()
            row[output_name] = valid[variable].corr(valid[source_col], method=method)
        rows.append(row)
    return pd.DataFrame(rows)


def make_observation_count_table(data):
    # Report the number of complete observations behind each pairwise correlation.
    rows = []
    for variable in VARIABLES:
        row = {"Variable": DISPLAY_VARIABLE_NAMES.get(variable, variable)}
        for output_name, source_col in REFERENCE_VARIABLES.items():
            if variable == source_col:
                row[output_name] = int(data[variable].notna().sum())
            else:
                row[output_name] = len(data[[variable, source_col]].dropna())
        rows.append(row)
    return pd.DataFrame(rows)


def make_combined_correlation_table(pearson, spearman):
    # Combine Pearson and Spearman correlations in one publication-ready table.
    combined = pearson[["Variable"]].copy()
    for col in REFERENCE_VARIABLES:
        combined[f"Pearson: {col}"] = pearson[col]
    for col in REFERENCE_VARIABLES:
        combined[f"Spearman: {col}"] = spearman[col]
    return combined


def render_combined_table_svg(table, output_path):
    # Render the combined correlation table as dependency-free SVG.
    display = table.copy()
    for col in display.columns:
        if col != "Variable":
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")

    width = 1250
    margin = 28
    title_height = 42
    group_header_height = 34
    header_height = 54
    row_height = 42
    height = margin * 2 + title_height + group_header_height + header_height + row_height * len(display)
    variable_width = 360
    value_width = (width - margin * 2 - variable_width) / 6
    col_widths = [variable_width] + [value_width] * 6

    x_positions = [margin]
    for col_width in col_widths[:-1]:
        x_positions.append(x_positions[-1] + col_width)

    headers = [
        "Variable",
        "Population",
        "GVA",
        "JTF allocation",
        "Population",
        "GVA",
        "JTF allocation",
    ]

    def rect(x, y, w, h, fill, stroke="#c8c8c8"):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'

    def text(x, y, value, size=16, weight="400", anchor="start", fill="#111111"):
        escaped = html.escape(str(value))
        return (
            f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escaped}</text>'
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#ffffff", "#ffffff"),
        text(margin, margin + 24, "Correlation table: Pearson and Spearman coefficients", size=22, weight="700"),
    ]

    y = margin + title_height
    elements.append(rect(x_positions[0], y, col_widths[0], group_header_height, "#f2f2f2"))
    elements.append(rect(x_positions[1], y, sum(col_widths[1:4]), group_header_height, "#e8eef7"))
    elements.append(rect(x_positions[4], y, sum(col_widths[4:7]), group_header_height, "#e9f4ec"))
    elements.append(text(x_positions[1] + sum(col_widths[1:4]) / 2, y + 23, "Pearson", size=15, weight="700", anchor="middle"))
    elements.append(text(x_positions[4] + sum(col_widths[4:7]) / 2, y + 23, "Spearman", size=15, weight="700", anchor="middle"))

    y += group_header_height
    for idx, header in enumerate(headers):
        elements.append(rect(x_positions[idx], y, col_widths[idx], header_height, "#f7f7f7"))
        anchor = "start" if idx == 0 else "middle"
        x_text = x_positions[idx] + 12 if idx == 0 else x_positions[idx] + col_widths[idx] / 2
        elements.append(text(x_text, y + 33, header, size=14, weight="700", anchor=anchor))

    y += header_height
    for row_idx, (_, row) in enumerate(display.iterrows()):
        fill = "#ffffff" if row_idx % 2 == 0 else "#fbfbfb"
        for col_idx, col_name in enumerate(display.columns):
            elements.append(rect(x_positions[col_idx], y, col_widths[col_idx], row_height, fill))
            anchor = "start" if col_idx == 0 else "middle"
            x_text = x_positions[col_idx] + 12 if col_idx == 0 else x_positions[col_idx] + col_widths[col_idx] / 2
            elements.append(text(x_text, y + 27, row[col_name], size=13, anchor=anchor))
        y += row_height

    elements.append("</svg>")
    output_path.write_text("\n".join(elements), encoding="utf-8")


def convert_svg_to_png(svg_path, png_path):
    # Use macOS Quick Look to render the SVG table as PNG when available.
    qlmanage = shutil.which("qlmanage")
    if not qlmanage:
        print(f"PNG export skipped because qlmanage is not available. SVG saved to:\n{svg_path}")
        return False

    subprocess.run(
        [qlmanage, "-t", "-s", "2500", "-o", str(png_path.parent), str(svg_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    quicklook_png = png_path.parent / f"{svg_path.name}.png"
    if quicklook_png.exists():
        if png_path.exists():
            png_path.unlink()
        quicklook_png.rename(png_path)
        return True

    print(f"PNG export did not produce the expected file. SVG saved to:\n{svg_path}")
    return False


def render_combined_table_html(table, output_path):
    # Render the combined table as HTML for a reliable headless-browser PNG export.
    display = table.copy()
    for col in display.columns:
        if col != "Variable":
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")

    rows = []
    for _, row in display.iterrows():
        cells = [f"<td class='variable'>{html.escape(str(row['Variable']))}</td>"]
        for col in display.columns[1:]:
            cells.append(f"<td>{html.escape(str(row[col]))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 32px;
    background: #ffffff;
    font-family: Arial, Helvetica, sans-serif;
    color: #111111;
  }}
  .wrap {{
    width: 1480px;
  }}
  h1 {{
    margin: 0 0 22px 0;
    font-size: 30px;
    line-height: 1.2;
  }}
  table {{
    border-collapse: collapse;
    width: auto;
    table-layout: fixed;
    font-size: 17px;
  }}
  th, td {{
    border: 1px solid #c8c8c8;
    padding: 13px 12px;
    text-align: center;
    vertical-align: middle;
  }}
  thead tr:first-child th {{
    background: #e8eef7;
    font-size: 19px;
    font-weight: 700;
  }}
  thead tr:first-child th:first-child {{
    background: #f2f2f2;
  }}
  thead tr:nth-child(2) th {{
    background: #f7f7f7;
    font-weight: 700;
  }}
  tbody tr:nth-child(even) {{
    background: #fbfbfb;
  }}
  .variable {{
    width: 260px;
    text-align: left;
  }}
  .value {{
    width: 150px;
  }}
</style>
</head>
<body>
<div class="wrap">
<h1>Correlation table: Pearson and Spearman coefficients</h1>
<table>
  <colgroup>
    <col style="width: 260px">
    <col span="6" style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th class="variable"></th>
      <th class="value" colspan="3">Pearson</th>
      <th class="value" colspan="3">Spearman</th>
    </tr>
    <tr>
      <th class="variable">Variable</th>
      <th class="value">Population</th>
      <th class="value">GVA</th>
      <th class="value">JTF allocation</th>
      <th class="value">Population</th>
      <th class="value">GVA</th>
      <th class="value">JTF allocation</th>
    </tr>
  </thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>
</div>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def convert_html_to_png(html_path, png_path):
    # Use local Chrome in headless mode to create a full PNG screenshot of the table.
    chrome_candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    chrome = next((path for path in chrome_candidates if path and Path(path).exists()), None)
    if not chrome:
        print(f"PNG export skipped because Chrome/Chromium is not available. HTML saved to:\n{html_path}")
        return False

    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            "--window-size=1600,820",
            f"--screenshot={png_path}",
            html_path.as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return png_path.exists()


def main():
    # Check that both input files are available before running the analysis.
    if not REGION_DATA_PATH.exists():
        raise FileNotFoundError(f"Region data not found: {REGION_DATA_PATH}")
    if not FINANCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Finance data not found: {FINANCE_DATA_PATH}")

    regions = read_region_data()
    finances = read_finance_data()
    matched, excluded, excluded_duplicate_sources = add_allocations(regions, finances)

    # Pearson is used as the main SI table; Spearman is exported as a robustness sheet.
    pearson = make_correlation_table(matched, method="pearson")
    spearman = make_correlation_table(matched, method="spearman")
    combined = make_combined_correlation_table(pearson, spearman)
    observation_counts = make_observation_count_table(matched)
    match_summary = (
        matched["Allocation match level"]
        .value_counts(dropna=False)
        .rename_axis("Allocation match level")
        .reset_index(name="Regions")
    )

    # Write the main table and diagnostic sheets to one Excel workbook.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        pearson.to_excel(writer, sheet_name="Correlation Table", index=False)
        spearman.to_excel(writer, sheet_name="Spearman Robustness", index=False)
        combined.to_excel(writer, sheet_name="Combined Pearson Spearman", index=False)
        observation_counts.to_excel(writer, sheet_name="Observation Counts", index=False)
        match_summary.to_excel(writer, sheet_name="Match Summary", index=False)
        matched.to_excel(writer, sheet_name="Matched Data", index=False)
        excluded.to_excel(writer, sheet_name="Excluded No NUTS Match", index=False)
        excluded_duplicate_sources.to_excel(writer, sheet_name="Excluded Duplicate Sources", index=False)

    render_combined_table_html(combined, COMBINED_TABLE_HTML)
    png_created = convert_html_to_png(COMBINED_TABLE_HTML, COMBINED_TABLE_PNG)
    render_combined_table_svg(combined, COMBINED_TABLE_SVG)

    print(f"Correlation table saved to:\n{OUTPUT_FILE}")
    if png_created:
        print(f"Combined Pearson/Spearman PNG saved to:\n{COMBINED_TABLE_PNG}")
    print("\nMatch summary:")
    print(match_summary.to_string(index=False))
    print(f"\nExcluded because no NUTS1/NUTS2 allocation match was found: {len(excluded)}")
    print(f"Excluded because allocation source was already used: {len(excluded_duplicate_sources)}")
    print("\nPearson correlation table:")
    print(pearson.to_string(index=False))


if __name__ == "__main__":
    main()
