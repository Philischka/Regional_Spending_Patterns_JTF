from pathlib import Path
import os
import re

import pandas as pd


INDICATOR_PATH = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/"
    "05_Scripts/02_Master_Framework/output/Master_Framework.xlsx"
)
INDICATOR_SHEET = "Index Data"
FINANCE_PATH = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/"
    "04_JTF_Finances_Details/Data_used/2021-2027_JTF_Finances_Details_with_NUTS_20251115.csv"
)
OUTPUT_DIR = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/"
    "07_Visualizations/00_Publication/08_Review/02_Box_Plots_all_regions"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep matplotlib cache inside the output folder to avoid permission issues.
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / "mpl_config"))

import matplotlib.pyplot as plt


EXCLUDED_COUNTRIES = {"NL", "PT", "CY"}
INTERVENTION_FIELD = "Intervention Field"

SCORE_COLUMNS = {
    "Index Transition Exposure Review": {
        "plot_column": "Index Transition Exposure Review",
        "title": "Transition Exposure",
        "ylabel": "Transition Exposure",
        "invert": False,
    },
    "Adaptive Capacity Review": {
        "plot_column": "Adaptive Capacity Review (inverted)",
        "title": "Adaptive Capacity",
        "ylabel": "Adaptive Capacity",
        "invert": True,
    },
    "Index Sensitivity Review": {
        "plot_column": "Index Sensitivity Review",
        "title": "Sensitivity",
        "ylabel": "Sensitivity",
        "invert": False,
    },
}

BASE_COLUMNS = [
    "Country Code",
    "Country Name",
    "Corresponding NUTS2 level",
    "NUTS 2 Code",
    "JTF Region",
]


def to_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def clean_code(series):
    return series.astype("string").str.strip().str.upper()


def split_codes(value):
    if pd.isna(value):
        return []
    return [
        code.strip().upper()
        for code in str(value).replace(";", "/").replace(",", "/").split("/")
        if code.strip()
    ]


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def read_indicator_data():
    df = pd.read_excel(INDICATOR_PATH, sheet_name=INDICATOR_SHEET)

    required = BASE_COLUMNS + list(SCORE_COLUMNS)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError("Missing columns in indicator data: " + ", ".join(missing))

    df = df[required].copy()
    df["Country Code"] = clean_code(df["Country Code"])
    df["NUTS 2 Code"] = clean_code(df["NUTS 2 Code"])
    df["JTF Region"] = to_numeric(df["JTF Region"])

    # Keep only NUTS1 and NUTS2 codes, remove NUTS0/country-level rows, and exclude NL/PT/CY.
    code_length = df["NUTS 2 Code"].str.len()
    df = df[code_length.isin([3, 4])].copy()
    df = df[~df["Country Code"].isin(EXCLUDED_COUNTRIES)].copy()
    df = df[df["JTF Region"].isin([0, 1])].copy()

    # The master sheet contains repeated NUTS2 codes at NUTS3 row level.
    # The selected scores are regional-level values, so each NUTS code should enter only once.
    df = df.drop_duplicates(subset=["NUTS 2 Code"], keep="first").copy()

    for col, settings in SCORE_COLUMNS.items():
        df[col] = to_numeric(df[col])
        if settings["invert"]:
            df[settings["plot_column"]] = 1 - df[col]
        else:
            df[settings["plot_column"]] = df[col]

    return df


def read_finance_data():
    finances = pd.read_csv(FINANCE_PATH)

    required = [
        "NUTS 2 Code",
        "NUTS 1 Code",
        "ms",
        "dimension_type",
        "cofinancing_rate",
        "eu_amount",
        "Policy Category Title",
    ]
    missing = [col for col in required if col not in finances.columns]
    if missing:
        raise KeyError("Missing columns in finance data: " + ", ".join(missing))

    finances = finances[finances["dimension_type"] == INTERVENTION_FIELD].copy()
    finances["ms"] = clean_code(finances["ms"])
    finances = finances[~finances["ms"].isin(EXCLUDED_COUNTRIES)].copy()

    finances["NUTS 2 Code"] = clean_code(finances["NUTS 2 Code"])
    finances["NUTS 1 Code"] = clean_code(finances["NUTS 1 Code"])
    finances.loc[finances["NUTS 1 Code"].isin(["AT1/AT2", "HU2/HU3"]), "NUTS 1 Code"] = pd.NA

    finances["eu_amount_num"] = to_numeric(finances["eu_amount"])

    return finances


def allocation_map(finances, code_col):
    return (
        finances.assign(match_code=finances[code_col].apply(split_codes))
        .explode("match_code")
        .dropna(subset=["match_code"])
        .groupby("match_code", as_index=True)
        .agg(
            allocation_eu_amount=("eu_amount_num", "sum"),
        )
        .to_dict("index")
    )


def build_allocation_maps(finances):
    return {
        "NUTS2": allocation_map(finances, "NUTS 2 Code"),
        "NUTS1": allocation_map(finances, "NUTS 1 Code"),
        "MS": finances.groupby("ms", as_index=True)
        .agg(allocation_eu_amount=("eu_amount_num", "sum"))
        .to_dict("index"),
    }


def match_allocation(region_code, allocation_maps):
    code = str(region_code).strip().upper()
    parent_nuts1 = code[:3] if len(code) >= 3 else code
    member_state = code[:2]

    if code in allocation_maps["NUTS2"]:
        return allocation_maps["NUTS2"][code], "NUTS2", f"NUTS2:{code}"
    if code in allocation_maps["NUTS1"]:
        return allocation_maps["NUTS1"][code], "NUTS1", f"NUTS1:{code}"
    if parent_nuts1 in allocation_maps["NUTS1"]:
        return allocation_maps["NUTS1"][parent_nuts1], "NUTS1 parent", f"NUTS1:{parent_nuts1}"
    if member_state in allocation_maps["MS"]:
        return allocation_maps["MS"][member_state], "MS", f"MS:{member_state}"
    return {"allocation_eu_amount": pd.NA}, "No match", pd.NA


def attach_finance_matches(indicators, finances):
    allocation_maps = build_allocation_maps(finances)
    matches = indicators["NUTS 2 Code"].apply(lambda code: match_allocation(code, allocation_maps))

    out = indicators.copy()
    out["Matched allocation (eu_amount)"] = [match[0]["allocation_eu_amount"] for match in matches]
    out["Allocation match level"] = [match[1] for match in matches]
    out["Allocation source code"] = [match[2] for match in matches]
    return out


def plot_boxplot(data, plot_col, title, ylabel):
    plot_data = data[["JTF Region", plot_col]].dropna().copy()
    non_jtf = plot_data.loc[plot_data["JTF Region"].eq(0), plot_col]
    jtf = plot_data.loc[plot_data["JTF Region"].eq(1), plot_col]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    box = ax.boxplot(
        [non_jtf, jtf],
        tick_labels=[f"Not JTF eligible\n(n={len(non_jtf)})", f"JTF eligible\n(n={len(jtf)})"],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 2.0},
        boxprops={"linewidth": 1.5, "edgecolor": "#333333"},
        whiskerprops={"linewidth": 1.4, "color": "#333333"},
        capprops={"linewidth": 1.4, "color": "#333333"},
    )

    for patch, color in zip(box["boxes"], ["#D9D9D9", "#5C8D89"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=22, labelpad=12)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=18, pad=8)
    ax.tick_params(axis="y", labelsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    slug = safe_filename(title)
    pdf_path = OUTPUT_DIR / f"Boxplot_{slug}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main():
    if not INDICATOR_PATH.exists():
        raise FileNotFoundError(f"Indicator input not found: {INDICATOR_PATH}")
    if not FINANCE_PATH.exists():
        raise FileNotFoundError(f"Financial flow input not found: {FINANCE_PATH}")

    indicators = read_indicator_data()
    finances = read_finance_data()
    plot_data = attach_finance_matches(indicators, finances)

    output_files = []
    for _, settings in SCORE_COLUMNS.items():
        output_files.append(
            plot_boxplot(
                plot_data,
                plot_col=settings["plot_column"],
                title=settings["title"],
                ylabel=settings["ylabel"],
            )
        )

    summary = (
        plot_data.groupby("JTF Region", dropna=False)
        .size()
        .rename("Regions")
        .reset_index()
    )
    match_summary = (
        plot_data["Allocation match level"]
        .value_counts(dropna=False)
        .rename_axis("Allocation match level")
        .reset_index(name="Regions")
    )

    export_xlsx = OUTPUT_DIR / "boxplot_data_all_regions.xlsx"
    with pd.ExcelWriter(export_xlsx, engine="openpyxl") as writer:
        plot_data.to_excel(writer, sheet_name="plot_data", index=False)
        summary.to_excel(writer, sheet_name="jtf_group_summary", index=False)
        match_summary.to_excel(writer, sheet_name="finance_match_summary", index=False)

    print(f"Saved boxplots to:\n{OUTPUT_DIR}")
    for pdf_path in output_files:
        print(f"- {pdf_path.name}")
    print(f"\nSaved plot data to:\n{export_xlsx}")
    print("\nJTF group summary:")
    print(summary.to_string(index=False))
    print("\nFinance match summary:")
    print(match_summary.to_string(index=False))


if __name__ == "__main__":
    main()
