
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from matplotlib.ticker import PercentFormatter
import matplotlib.lines as mlines
from adjustText import adjust_text 
from matplotlib.backends.backend_pdf import PdfPages




# ---------------------------------------------------------
# Pfade
# ---------------------------------------------------------
csv_path = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/04_JTF_Finances_Details/Data_used/2021-2027_JTF_Finances_Details_with_NUTS_20251115.csv")

xlsx_path = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/05_Scripts/02_Master_Framework/output/JTF Regions/Adjusted for merge/01_JRF_Region_Ranking_NUTS2_1_0.xlsx")

# Output: Adaptive Capacity
output_dir_cofin = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/07_Adaptive_Capacity_Cofinancing/Output_cofinancingrate")
output_dir_ownamount = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/07_Adaptive_Capacity_Cofinancing/Output_averageownamount")
output_dir_avgownamount = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/07_Adaptive_Capacity_Cofinancing/Output_averageownamount")
output_dir_stacked = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/07_Adaptive_Capacity_Cofinancing/03_Bar_Chart")
output_dir_cofin_ownamount = Path(
    "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/07_Adaptive_Capacity_Cofinancing/04_Cofinance&ownamount"
)

for d in [
    output_dir_cofin,
    output_dir_ownamount,
    output_dir_avgownamount,
    output_dir_stacked,
    output_dir_cofin_ownamount,  # <- NEU
]:
    d.mkdir(parents=True, exist_ok=True)


# Output: Cost of Capital
output_dir_coc = Path("/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/07_Adaptive_Capacity_Cofinancing/05_Only_CostofCapital")
output_dir_coc_bar = output_dir_coc / "01_Bar_Chart_avCofinance"
output_dir_coc_bubbles = output_dir_coc / "Bubbles"  # <<< NEU

for d in [
    output_dir_cofin,
    output_dir_ownamount,
    output_dir_avgownamount,
    output_dir_coc,
    output_dir_coc_bar,
    output_dir_coc_bubbles,  # <<< NEU
]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Hilfsfunktion: LOWESS-Scatterplot mit optionalen Labels
# ---------------------------------------------------------
def scatter_with_lowess(df, x_col, y_col, title, x_label, y_label, save_path, frac=0.3, label_col=None):
    """
    Erstellt einen Scatterplot mit LOWESS-Fit.
    Wenn label_col angegeben ist, werden die Werte dieser Spalte als Textlabels an die Punkte geschrieben.
    """
    cols = [x_col, y_col]
    if label_col is not None:
        cols.append(label_col)

    plot_df = df[cols].dropna(subset=[x_col, y_col])
    if plot_df.empty:
        print(f"Keine Daten für Plot {title}.")
        return

    x = plot_df[x_col].values
    y = plot_df[y_col].values

    lowess_result = lowess(y, x, frac=frac, return_sorted=True)
    x_lowess = lowess_result[:, 0]
    y_lowess = lowess_result[:, 1]

    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    plt.plot(x_lowess, y_lowess, linewidth=2)

    if label_col is not None:
        for _, row in plot_df.iterrows():
            plt.annotate(
                str(row[label_col]),
                (row[x_col], row[y_col]),
                fontsize=6,
                alpha=0.7,
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot gespeichert unter: {save_path}")
# ---------------------------------------------------------
# Neue Funktion: Bubble-Scatterplot (Punktgröße = sum_own_amount)
# ---------------------------------------------------------

def scatter_bubble_ownamount(
    df,
    x_col,
    y_col,
    size_col,
    title,
    x_label,
    y_label,
    save_path,
    label_col=None,
    size_scale=2000,
    n_bins=5,
    label_mode="center",   # NEU: wie Labels angeordnet werden
    label_top_n=15,        # NEU: für Top-N-Variante
    nuts_mapping=None, 
):

    """
    Bubble scatterplot:
    - x-axis: x_col
    - y-axis: y_col
    - bubble size & color: size_col (own financing amount)
    - right-hand legend:
        * top: color ranges (equal-sized points)
        * bottom: bubble size examples (different sizes)
    """

    cols = [x_col, y_col, size_col]
    if label_col:
        cols.append(label_col)

    # Filtere NUTS0 (Länder) raus, nur NUTS1 und NUTS2 behalten
    plot_df = df[cols].dropna()

    # Filtere NUTS0-Regionen (die Länder-Codes, z. B. "EL") raus
    plot_df = plot_df[plot_df[label_col].str.len() > 2]  # Nur NUTS1 und NUTS2 behalten

    if plot_df.empty:
        print(f"No data available for: {title}")
        return

    x = plot_df[x_col].values
    y = plot_df[y_col].values
    raw = plot_df[size_col].values

    min_val = raw.min()
    max_val = raw.max()

    # -------- helper for "nice" rounding --------
    def nice_round(value):
        order = int(np.floor(np.log10(max_val))) - 1
        base = 10 ** max(order, 3)  # at least 1,000
        return np.round(value / base) * base

    # -------- 1. bins / ranges for colors --------
    edges = np.linspace(min_val, max_val, n_bins + 1)
    edges_rounded = [nice_round(v) for v in edges]

    bin_indices = np.digitize(raw, edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    cmap = plt.cm.Greens
    colors_bins = [cmap((i + 0.5) / n_bins) for i in range(n_bins)]

    plot_df["bin_idx"] = bin_indices
    plot_df["bin_color"] = plot_df["bin_idx"].apply(lambda i: colors_bins[i])

    # continuous bubble sizes
    sizes = (raw / max_val) * size_scale

    plt.figure(figsize=(10, 7))

    # -------- main scatter --------
    plt.scatter(
        x,
        y,
        s=sizes,
        c=plot_df["bin_color"],
        alpha=0.85,
        edgecolors="black",
        linewidth=0.4,
    )

        # --------------------------
        # --------------------------
    # Labels für die Punkte
    # --------------------------
    texts = []

    if label_col:
        # Modus 1: klassische Labels direkt im Punkt
        if label_mode == "center":
            for _, row in plot_df.iterrows():
                plt.text(
                    row[x_col],
                    row[y_col],
                    str(row[label_col]),
                    fontsize=7,
                    ha="center",
                    va="center",
                )

        # Modus 2: leichte Offsets (abwechselnd hoch/runter)
        elif label_mode == "offset":
            for i, (_, row) in enumerate(plot_df.iterrows()):
                dx = 0.01                           # horizontaler Offset
                dy = 0.01 if i % 2 == 0 else -0.01  # vertikal alternierend
                plt.text(
                    row[x_col] + dx,
                    row[y_col] + dy,
                    str(row[label_col]),
                    fontsize=7,
                    ha="left",
                    va="center",
                )

        # Modus 3: nur Top-N Regionen (nach size_col, z.B. sum_own_amount)
        elif label_mode == "topN":
            # Wie viele extrem kleine Eigenanteile zusätzlich beschriftet werden sollen
           # n_low = 5   # kannst du bei Bedarf anpassen

            # Top-N mit größter Eigenfinanzierung
            top_df = plot_df.nlargest(label_top_n, size_col)

            # N kleinste Eigenfinanzierungen
           # low_df = plot_df.nsmallest(n_low, size_col)

            # Spezifische Regionen, die immer beschriftet werden sollen
            specific_regions = [
                "FRL0", "IE06", "NL34", "SE33", "EL42", "DK0", "ES24"
            ]

            # Auswahl der spezifischen Regionen
            specific_df = plot_df[plot_df[label_col].isin(specific_regions)]

            # Kombination von Top-N, Low-N und spezifischen Regionen
            label_df = pd.concat([top_df, specific_df]).drop_duplicates(subset=[label_col])
                        # Regionen, die oberhalb der Bubble beschriftet werden sollen
            regions_above_1 = ["Denmark", "South Aegean", "Zeeland"]
            regions_above_2 = ["Aragon"]
            regions_above_3 = ["Brandenburg", "Lower Silesia"]
            regions_above_4 = ["Greater Poland"]
            regions_above_5 = ["Northwest (CZ)", "South-West Oltenia", "NRW", "Groningen"]
            regions_above_7 = ["Mainland (FI)"]
            regions_above_8 = ["Upper Norrland", "Eastern and Midland (IE)"]
            regions_above_9 = ["Silesia"]

            name_wrapped_map = {
                "IE06": "Eastern and\nMidland (IE)",
                "BG3": "Northern and\nSoutheastern\n(BG)",
                "EL42": "South\nAegean",
                "DEE0": "Saxony-\nAnhalt", 
                "SE33": "Upper\nNorrland",
            }

            # Alle relevanten Labels hinzufügen
            # Alle relevanten Labels hinzufügen
            # Alle relevanten Labels hinzufügen
            # Alle relevanten Labels hinzufügen
            for _, row in label_df.iterrows():

                region_code = row[label_col]
                region_name = nuts_mapping.get(region_code, region_code)

                x_pos = row[x_col]
                y_pos = row[y_col]

                # Text ggf. umbrechen – über den CODE
                label_text = name_wrapped_map.get(region_code, region_name)

                # ----------------------------------------------------------
                # SPEZIALFALL: Provence–Alpes–Côte d’Azur (über NUTS-Code)
                # ----------------------------------------------------------
                if region_code == "FRL0":
                    plt.annotate(
                        label_text,
                        xy=(x_pos, y_pos),
                        xycoords="data",
                        xytext=(30, -30),          # unten rechts
                        textcoords="offset points",
                        fontsize=7,
                        ha="left",
                        va="center",
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=0.6,
                            color="grey",
                            shrinkA=4,
                            shrinkB=2
                        )
                    )
                    continue  # Standard-Logik überspringen

                # ----------------------------------------------------------
                # SPEZIALFALL: Province of Hainaut (auch über NUTS-Code)
                # ----------------------------------------------------------
                if region_code == "BE32":  # falls das der Code ist – ggf. anpassen
                    plt.annotate(
                        label_text,
                        xy=(x_pos, y_pos),
                        xycoords="data",
                        xytext=(-30, -20),         # unten links
                        textcoords="offset points",
                        fontsize=7,
                        ha="right",
                        va="center",
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=0.6,
                            color="grey",
                            shrinkA=4,
                            shrinkB=2
                        )
                    )
                    continue

                # ----------------------------------------------------------
                # Ab hier deine bisherige Offset-Logik
                # ----------------------------------------------------------
                dx = 0
                dy = 0
                ha = "center"
                va = "center"

                if region_name in regions_above_1:
                    dy = 0.01
                    ha = "center"
                    va = "bottom"
                elif region_name in regions_above_2:
                    dy = 0.013
                    ha = "center"
                    va = "bottom"
                elif region_name in regions_above_3:
                    dy = 0.026
                    ha = "center"
                    va = "bottom"
                elif region_name in regions_above_4:
                    dy = 0.02
                    ha = "left"
                    va = "bottom"
                elif region_name in regions_above_5:
                    dy = 0.023
                    ha = "center"
                    va = "bottom"
                elif region_name in regions_above_7:
                    dy = -0.023
                    ha = "center"
                    va = "top"
                elif region_name in regions_above_8:
                    dy = -0.015
                    ha = "center"
                    va = "top"
                elif region_name in regions_above_9:
                    dy = -0.013
                    ha = "center"
                    va = "top"

                plt.text(
        x_pos + dx,
        y_pos + dy,
        label_text,
        fontsize=7,
        ha=ha,
        va=va,
    )

        # Modus 4: automatische Entzerrung mit adjustText
        elif label_mode == "adjust":
            try:
                from adjustText import adjust_text   # Voraussetzung: pip install adjustText
                for _, row in plot_df.iterrows():
                    t = plt.text(
                        row[x_col],
                        row[y_col],
                        str(row[label_col]),
                        fontsize=7,
                        ha="center",
                        va="center",
                    )
                    texts.append(t)

                adjust_text(
                    texts,
                    arrowprops=dict(
                        arrowstyle="-",
                        lw=0.5,
                        color="gray",
                        alpha=0.7,
                    ),
                )
            except Exception as e:
                print("adjustText nicht verfügbar, fallback auf center-labels:", e)
                for _, row in plot_df.iterrows():
                    plt.text(
                        row[x_col],
                        row[y_col],
                        str(row[label_col]),
                        fontsize=7,
                        ha="center",
                        va="center",
                    )



    # ======================================================
        # ======================================================
    # Legend: color ranges (equal-sized dots)
    #          + bubble sizes (different sizes)
    #          → combined in ONE box
    # ======================================================

    # ---- helper for compact € values ----
    def format_eur(x):
        if x >= 1e9:
            return f"{x/1e9:.1f}B €"
        elif x >= 1e6:
            return f"{x/1e6:.0f}M €"
        elif x >= 1e3:
            return f"{x/1e3:.0f}k €"
        else:
            return f"{x:.0f} €"

    # --------------------------
    # 1) Farbranges
    # --------------------------
        # --------------------------
    # 1) Farbranges
    # --------------------------
    color_handles = []
    for i in range(n_bins):
        low = edges_rounded[i]
        high = edges_rounded[i + 1]

        if i == 0:
            label = f"≤ {format_eur(high)}"
        elif i == n_bins - 1:
            label = f"≥ {format_eur(low)}"
        else:
            label = f"{format_eur(low)}–{format_eur(high)}"

        h = plt.scatter(
            [], [], 
            s=80, 
            c=[colors_bins[i]], 
            edgecolors="black", 
            linewidth=0.4, 
            label=label
        )
        color_handles.append(h)

    # --------------------------
    # 2) Bubblegröße (median + max)
    # --------------------------
    med_val = nice_round(np.median(raw))

    size_levels = [med_val]   # <-- nur der Median


    size_handles = []
    for v in size_levels:
        size_rep = (v / max_val) * size_scale
        h = plt.scatter(
            [], [], 
            s=size_rep, 
            facecolors="none", 
            edgecolors="black", 
            linewidth=0.6,
            label=f"{format_eur(v)}"
        )
        size_handles.append(h)

    # --------------------------
    # 3) Kombinierte Legende
    # --------------------------
    combined_handles = (
        color_handles
        + [mlines.Line2D([], [], color="none", label="")]  # spacer
        + size_handles
    )

    legend = plt.legend(
        handles=combined_handles,
        title="Regional Co-Financing Amount",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.99),
        borderaxespad=1.0,
        frameon=True,
        borderpad=1.2,        # inner padding of the box
        labelspacing=1.2,     # vertical space between entries
        handlelength=1.8,     # length of the legend bullet marker
        handletextpad=1.2,    # space between bullet & label
        fontsize=10
    )

    # expand legend box by increasing the frame padding
    frame = legend.get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor("black")
    



    # -------- axes & save --------
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Bubble plot saved to: {save_path}")


#------------------------------------------
# Bubble Farbe an JTF share anpassen
#--------------------------------
def scatter_bubble_jtfshare(
    df,
    x_col,
    y_col,
    size_col,
    color_col,          # 🆕 eigene Spalte für Farbe (hier: jtf_share)
    title,
    x_label,
    y_label,
    save_path,
    label_col=None,
    size_scale=2000,
    n_bins=5,
    label_top_n=15,
    nuts_mapping=None,
    add_trendline=False, # <<< NEU
    save_path_trendline=None,   # <<< NEU
):
    """
    Bubble-Scatterplot:
    - x-axis: x_col (Adaptive Capacity)
    - y-axis: y_col (EU Co-Financing Share)
    - bubble size: size_col (regionale Eigenfinanzierung, sum_own_amount)
    - bubble color: color_col (regionaler Anteil am gesamten JTF, jtf_share)
    """

    cols = [x_col, y_col, size_col, color_col]
    if label_col:
        cols.append(label_col)

    plot_df = df[cols].dropna()
    # Nur NUTS1 & NUTS2 (wie beim anderen Plot)
    if label_col:
        plot_df = plot_df[plot_df[label_col].str.len() > 2]

    if plot_df.empty:
        print(f"No data available for: {title}")
        return

    x = plot_df[x_col].values
    y = plot_df[y_col].values
    raw_size = plot_df[size_col].values         # für Bubblegröße
    raw_color = plot_df[color_col].values       # für Farben (JTF-Anteil)

    # -------- helper für "nice" rounding (für Größen-Legende) --------
    max_val = raw_size.max()

    def nice_round(value):
        order = int(np.floor(np.log10(max_val))) - 1
        base = 10 ** max(order, 3)  # mindestens 1.000
        return np.round(value / base) * base

        # -------- 1. Farbbins auf Basis von jtf_share mit Top-5-Sonderklasse --------
    min_c = raw_color.min()
    max_c = raw_color.max()

    n_top = 5  # Anzahl der Regionen mit höchstem JTF-Anteil in eigener (dunkelster) Klasse

    if len(raw_color) > n_top:
        # Schwellenwert: kleinster Wert innerhalb der Top-5
        cutoff = np.sort(raw_color)[-n_top]
        is_top = raw_color >= cutoff
    else:
        # Falls es weniger als 5 Regionen gibt, nicht trennen
        is_top = np.zeros_like(raw_color, dtype=bool)

    # Restliche Regionen (ohne Top-5) für feinere Bins
    rest_values = raw_color[~is_top]
    if rest_values.size > 0:
        rest_min = rest_values.min()
        rest_max = rest_values.max()
    else:
        rest_min = min_c
        rest_max = max_c

    # Wir verwenden (n_bins - 1) Klassen für den "Rest" und 1 Klasse für die Top-5
    n_bins_rest = n_bins - 1
    edges_rest = np.linspace(rest_min, rest_max, n_bins_rest + 1)

    # Start: alle als Top-5 klassifizieren (letzte Klasse)
    bin_indices = np.full_like(raw_color, fill_value=n_bins - 1, dtype=int)

    # Für alle Nicht-Top-5 richtige Bins zuweisen
    rest_idx = np.where(~is_top)[0]
    rest_bin = np.digitize(rest_values, edges_rest, right=True) - 1
    rest_bin = np.clip(rest_bin, 0, n_bins_rest - 1)
    bin_indices[rest_idx] = rest_bin

    # Farben (dunkelste Klasse = Top-5)
    cmap = plt.cm.Greens
    colors_bins = [cmap((i + 0.5) / n_bins) for i in range(n_bins)]

    plot_df["bin_idx"] = bin_indices
    plot_df["bin_color"] = plot_df["bin_idx"].apply(lambda i: colors_bins[i])


    # kontinuierliche Bubblegrößen (weiterhin own_amount)
    sizes = (raw_size / max_val) * size_scale

    plt.figure(figsize=(10, 7))

    # -------- Haupt-Scatter --------
    plt.scatter(
        x,
        y,
        s=sizes,
        c=plot_df["bin_color"],
        alpha=0.85,
        edgecolors="black",
        linewidth=0.4,
    )

    # ======================================================
    # Labels (Top-15 etc.) – gleiche Logik wie im anderen Plot
    # ======================================================
    if label_col and nuts_mapping is not None:
        # Top-N nach size_col (own_amount)
        top_df = plot_df.nlargest(label_top_n, size_col)

        # Spezifische Regionen, die immer beschriftet werden sollen (weiterhin über Codes!)
        specific_regions = [
            "FRL0", "IE06", "NL34", "SE33", "EL42", "DK0", "ES24"
        ]

        # Auswahl der spezifischen Regionen
        specific_df = plot_df[plot_df[label_col].isin(specific_regions)]

        # Kombination von Top-N und spezifischen Regionen
        label_df = pd.concat([top_df, specific_df]).drop_duplicates(subset=[label_col])

        # Regionen, die oberhalb / unterhalb der Bubble beschriftet werden sollen
        regions_above_1 = ["Denmark", "South Aegean", "Zeeland"]
        regions_above_2 = ["Aragon"]
        regions_above_3 = ["Brandenburg", "Lower Silesia"]
        regions_above_4 = ["Greater Poland", "Province of Hainaut"]
        regions_above_5 = ["Northwest (CZ)", "South-West Oltenia", "NRW", "Groningen"]
        regions_above_8 = ["Upper Norrland", "Eastern and Midland (IE)"]
        regions_above_9 = ["Silesia"]

        # Manuelle Umbrüche – über REGIONSCODES gesteuert
        name_wrapped_map = {
            "IE06": "Eastern and\nMidland (IE)",
            "BG3": "Northern and\nSoutheastern\n(BG)",
            "EL42": "South\nAegean",
            "DEE0": "Saxony-\nAnhalt",
            "SE33": "Upper\nNorrland",
        }

        # Alle relevanten Labels hinzufügen
               # Alle relevanten Labels hinzufügen
        for _, row in label_df.iterrows():

            region_code = row[label_col]
            region_name = nuts_mapping.get(region_code, region_code)

            x_pos = row[x_col]
            y_pos = row[y_col]

            # Text ggf. umbrechen – über den CODE
            label_text = name_wrapped_map.get(region_code, region_name)

            # ----------------------------------------------------------
            # SPEZIALFÄLLE MIT PFEILEN (über NUTS-CODE, robust)
            # ----------------------------------------------------------

            # Provence–Alpes–Côte d’Azur (FRL0) → schräg unten rechts
            if region_code == "FRL0":
                print("Annotate with arrow:", region_code, region_name)  # Debug
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),
                    xycoords="data",
                    xytext=(30, -20),          # 30 Punkte rechts, 30 Punkte runter
                    textcoords="offset points",
                    fontsize=7,
                    ha="left",
                    va="center",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=4,
                        shrinkB=2,
                    ),
                )
                continue  # Nichts weiter für diese Region


            # Brandenburg → schräg unten rechts
            if region_code == "DE40":
                print("Annotate with arrow:", region_code, region_name)  # Debug
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),
                    xycoords="data",
                    xytext=(30, -20),          # 30 Punkte rechts, 30 Punkte runter
                    textcoords="offset points",
                    fontsize=7,
                    ha="left",
                    va="center",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=4,
                        shrinkB=2,
                    ),
                )
                continue  # Nichts weiter für diese Region

                        # Greater Poland → kurzer Pfeil nach unten rechts, Text oben links
            if region_name == "Greater Poland":
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),          # Mittelpunkt der Bubble
                    xycoords="data",
                    xytext=(-25, 20),           # leicht links & oben
                    textcoords="offset points",
                    fontsize=7,
                    ha="right",                 # Text rechtsbündig (zeigt auf die Bubble)
                    va="bottom",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=4,
                        shrinkB=2,
                    ),
                )
                continue


                        # Greater Poland → kurzer Pfeil nach unten rechts, Text oben links
            if region_name == "South-West Oltenia":
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),          # Mittelpunkt der Bubble
                    xycoords="data",
                    xytext=(-15, 15),           # leicht links & oben
                    textcoords="offset points",
                    fontsize=7,
                    ha="right",                 # Text rechtsbündig (zeigt auf die Bubble)
                    va="bottom",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=4,
                        shrinkB=2,
                    ),
                )
                continue

                        # Northwest (CZ) → kurzer Pfeil gerade nach unten, Text direkt über der Bubble
            if region_name == "Northwest (CZ)":
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),          # Mittelpunkt der Bubble
                    xycoords="data",
                    xytext=(0, 30),             # direkt über der Bubble
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=4,
                        shrinkB=2,
                    ),
                )
                continue
                        # Northwest (CZ) → kurzer Pfeil gerade nach unten, Text direkt über der Bubble
            if region_name == "Zeeland":
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),          # Mittelpunkt der Bubble
                    xycoords="data",
                    xytext=(0, 20),             # direkt über der Bubble
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=5,
                        shrinkB=2,
                    ),
                )
                continue



                        # Lower Silesia → kurzer Pfeil nach unten links, Text oben rechts
            if region_name == "Lower Silesia":
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),          # Mittelpunkt der Bubble
                    xycoords="data",
                    xytext=(10, 25),            # leicht rechts & oben
                    textcoords="offset points",
                    fontsize=7,
                    ha="left",                  # Text linksbündig (zeigt Richtung Bubble)
                    va="bottom",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=4,
                        shrinkB=2,
                    ),
                )
                continue



            # Province of Hainaut (bitte ggf. NUTS2-Code anpassen, z. B. "BE32")
            if region_name == "Province of Hainaut":
                print("Annotate with arrow:", region_code, region_name)  # Debug
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),
                    xycoords="data",
                    xytext=(-30, -20),         # 30 Punkte links, 20 Punkte runter
                    textcoords="offset points",
                    fontsize=7,
                    ha="right",
                    va="center",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=4,
                        shrinkB=2,
                    ),
                )
                continue

            # Province of Hainaut (bitte ggf. NUTS2-Code anpassen, z. B. "BE32")
            if region_name == "Mainland (FI)":
                print("Annotate with arrow:", region_code, region_name)  # Debug
                plt.annotate(
                    label_text,
                    xy=(x_pos, y_pos),
                    xycoords="data",
                    xytext=(-15, -30),         # 30 Punkte links, 20 Punkte runter
                    textcoords="offset points",
                    fontsize=7,
                    ha="right",
                    va="center",
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=0.6,
                        color="grey",
                        shrinkA=4,
                        shrinkB=2,
                    ),
                )
                continue

            # ----------------------------------------------------------
            # Standard-Label-Logik (ohne Pfeil) – wie vorher
            # ----------------------------------------------------------
            dx = 0
            dy = 0
            ha = "center"
            va = "center"

            # 1) leicht oberhalb
            if region_name in regions_above_1:
                dy = 0.01
                ha = "center"
                va = "bottom"

            elif region_name in regions_above_2:
                dy = 0.013
                ha = "center"
                va = "bottom"

            elif region_name in regions_above_3:
                dy = 0.026
                ha = "center"
                va = "bottom"

            elif region_name in regions_above_4:
                dy = 0.02
                ha = "left"
                va = "bottom"

            elif region_name in regions_above_5:
                dy = 0.023
                ha = "center"
                va = "bottom"

            elif region_name in regions_above_8:
                dy = -0.015
                ha = "center"
                va = "top"

            elif region_name in regions_above_9:
                dy = -0.013
                ha = "center"
                va = "top"

            # Standard-Text ohne Pfeil
            plt.text(
                x_pos + dx,
                y_pos + dy,
                label_text,
                fontsize=7,
                ha=ha,
                va=va,
            )



            


   # ======================================================
    # Legende: Farben = JTF-Anteile, Größen = own_amount
    # ======================================================
    # Anteil der Top-n Regionen (für Legendenlabel)
    top_values = raw_color[is_top]
    top_share_min = top_values.min()
    top_share_max = top_values.max()

    # Funktion zum Formatieren der Prozentwerte
    def format_pct(x):
        return f"{x*100:.2f} %"

    color_handles = []
    for i in range(n_bins):
        if i < n_bins_rest:
            low = edges_rest[i]
            high = edges_rest[i + 1]

            if i == 0:
                label = f"≤ {format_pct(high)}"
            else:
                label = f"{format_pct(low)}–{format_pct(high)}"
        else:
            # Hier wird die letzte Gruppe angepasst
            label = f"≥ {format_pct(top_share_min)}"  # Angepasst für die höchste Gruppe

        h = plt.scatter(
            [], [],
            s=80,
            c=[colors_bins[i]],
            edgecolors="black",
            linewidth=0.4,
            label=label,
        )
        color_handles.append(h)


    # Größen-Legende: z.B. Median als Beispiel
    # THREE SIZE LEVELS: small – median – large
    small_val = nice_round(np.percentile(raw_size, 10))   # 10th percentile
    med_val   = nice_round(np.median(raw_size))           # 50th percentile (existing)
    large_val = nice_round(np.percentile(raw_size, 90))   # 90th percentile

    size_levels = [small_val, med_val, large_val]


    size_handles = []
    for v in size_levels:
        size_rep = (v / max_val) * size_scale
        h = plt.scatter(
            [], [],
            s=size_rep,
            facecolors="none",
            edgecolors="black",
            linewidth=0.6,
            label=f"{v/1e6:.0f}M €",   # z.B. 40M €
        )
        size_handles.append(h)

    import matplotlib.lines as mlines

    # Dummy-Handles für interne Überschriften
    header_color = plt.scatter([], [], s=0, label="Share of JTF")
    header_size  = plt.scatter([], [], s=0, label="Regional Amounts")

    # Spacer (eine leere Zeile)
    spacer = mlines.Line2D([], [], color="none", label=" ")  # ein Leerzeichen als Label


    # Reihenfolge in der Legende:
    # 1. Überschrift Farben
    # 2. Farbpunkte
    # 3. Spacer
    # 4. Überschrift Größen
    # 5. Größen-Bubbles
    # kleiner, feinregulierbarer Abstand zwischen Medium & Large
    # HIER stellst du den Abstand ein → s = 0.1 bis 30 (je nach gewünschtem Abstand)
    spacer_sizes = mlines.Line2D([], [], color="none", label=" ")


    legend_handles = (
        [header_color] +
        color_handles +
        [spacer] +
        [header_size] +
        [
            size_handles[0],      # small
            size_handles[1],      # medium
            spacer_sizes,         # FEINER Abstand
            size_handles[2],      # large
        ]
    )


    legend = plt.legend(
        handles=legend_handles,
        title="",                    # kein globaler Titel mehr
        loc="upper left",
        bbox_to_anchor=(1.02, 0.99),
        borderaxespad=1.0,
        frameon=True,
        borderpad=1.8,
        labelspacing=0.7,
        handlelength=1.8,
        handletextpad=1.2,
        fontsize=10,
    )

    # Rahmen wie bisher
    frame = legend.get_frame()
    frame.set_linewidth(0.8)
    frame.set_edgecolor("black")

    # Überschriften fett setzen
    texts = legend.get_texts()
    if texts:
        # Index 0 = "Share of JTF (color)"
        texts[0].set_weight("bold")
        # Index der zweiten Überschrift = 1 (Header) + len(color_handles) + 1 (Spacer)
        idx_size_header = 1 + len(color_handles) + 1
        if idx_size_header < len(texts):
            texts[idx_size_header].set_weight("bold")

        # ======================================================
    # OLS Trendlinie (optional)
    # ======================================================
    if add_trendline:
        import statsmodels.api as sm

        # OLS-Modell: y ~ beta0 + beta1 * x
        X = sm.add_constant(x)       # fügt konstante Spalte hinzu
        model = sm.OLS(y, X).fit()

        # x-Werte für die Linie generieren
        x_line = np.linspace(x.min(), x.max(), 200)
        X_line = sm.add_constant(x_line)
        y_line = model.predict(X_line)

        # Trendlinie plotten
        plt.plot(
            x_line,
            y_line,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label="OLS trendline",
        )

        plt.text(
            0.03,
            0.97,
            f"$R^2$ = {model.rsquared:.2f}",
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2.5),
        )

        # (optional) Summary in der Konsole
        print("\nOLS Regression Summary:")
        print(model.summary())

        # ---- Save OLS regression summary to a text file ----
        summary_text = model.summary().as_text()

        summary_path = str(save_path).replace(".pdf", "_OLS_summary.txt")

        with open(summary_path, "w") as f:
            f.write(summary_text)

        print(f"OLS summary saved to: {summary_path}")


    # Achsen & Speichern
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Bubble plot (JTF color) saved to: {save_path}")

# ---------------------------------------------------------
# Neue Funktion: Gestapeltes Balkendiagramm EU vs. Eigenanteil
# ---------------------------------------------------------
def plot_stacked_bar_cofinancing(df, title, save_path):
    # wir brauchen beide: avg_cofinancing_rate UND adaptive_capacity_index
    df_plot = df.dropna(subset=["avg_cofinancing_rate", "adaptive_capacity_index"]).copy()
    if df_plot.empty:
        print(f"Keine Daten für gestapeltes Balkendiagramm: {title}")
        return

    df_plot["eu_share"] = df_plot["avg_cofinancing_rate"]
    df_plot["own_share"] = 1 - df_plot["avg_cofinancing_rate"]

    # Sortierung nach bereits gemapptem Adaptive-Capacity-Index
    df_plot = df_plot.sort_values("adaptive_capacity_index")


    x = np.arange(len(df_plot))
    region_labels = df_plot["region_label"].astype(str).values

    fig, ax = plt.subplots(figsize=(max(8, len(df_plot) * 0.25), 6))

    ax.bar(x, df_plot["own_share"], label="Eigenanteil", color="#cc7000")
    ax.bar(x, df_plot["eu_share"], bottom=df_plot["own_share"], label="EU-Anteil", color="#ffb347")

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=90, fontsize=6)

    ax.set_ylabel("Anteil an Gesamtfinanzierung")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Gestapeltes Balkendiagramm gespeichert unter: {save_path}")
# ---------------------------------------------------------
# Neue Funktion: Gestapeltes Balkendiagramm auf Länderebene
# ---------------------------------------------------------
def plot_stacked_bar_cofinancing_by_country(df, title, save_path):
    """
    Erwartet ein DataFrame mit Spalten:
    - ms (Ländercode)
    - avg_cofinancing_rate

    Aggregiert, falls nötig, auf Länderebene und zeichnet
    einen 100%-Stacked-Bar pro Land.
    """
    # Sicherstellen, dass nur gültige Werte drin sind
    df_plot = df.dropna(subset=["ms", "avg_cofinancing_rate"]).copy()

    # Falls df noch auf Regionenebene ist, hier auf Länderebene mitteln
    df_plot = (
        df_plot
        .groupby("ms", dropna=False)
        .agg(avg_cofinancing_rate=("avg_cofinancing_rate", "mean"))
        .reset_index()
    )

    if df_plot.empty:
        print(f"Keine Daten für gestapeltes Balkendiagramm (Länder): {title}")
        return

    df_plot["eu_share"] = df_plot["avg_cofinancing_rate"]
    df_plot["own_share"] = 1 - df_plot["avg_cofinancing_rate"]

    # Länder alphabetisch sortieren
    df_plot = df_plot.sort_values("ms")

    x = np.arange(len(df_plot))
    country_labels = df_plot["ms"].astype(str).values

    fig, ax = plt.subplots(figsize=(max(8, len(df_plot) * 0.4), 6))

    # Unterer Balken: Eigenanteil (dunkleres Orange)
    ax.bar(x, df_plot["own_share"], label="Eigenanteil", color="#cc7000")

    # Oberer Balken: EU-Anteil (helleres Orange)
    ax.bar(x, df_plot["eu_share"], bottom=df_plot["own_share"], label="EU-Anteil", color="#ffb347")

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(country_labels, rotation=45, ha="right")

    ax.set_ylabel("Anteil an Gesamtfinanzierung")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Gestapeltes Balkendiagramm (Länder) gespeichert unter: {save_path}")
# ---------------------------------------------------------
# Neue Funktion: Gestapeltes Balkendiagramm auf Länderebene,
# sortiert nach nationaler Adaptive Capacity (aus Excel)
# ---------------------------------------------------------
def plot_stacked_bar_cofinancing_by_country_ac(df, title, save_path):
    """
    Erwartet ein DataFrame mit:
    - ms (Ländercode)
    - avg_cofinancing_rate
    - country_ac_index (nationaler AC-Wert aus Excel)

    Aggregiert auf Länderebene und sortiert nach
    nationaler Adaptive Capacity (niedrig -> hoch).
    """

    df_plot = df.dropna(subset=["ms", "avg_cofinancing_rate", "country_ac_index"]).copy()

    df_plot = (
        df_plot
        .groupby("ms", dropna=False)
        .agg(
            avg_cofinancing_rate=("avg_cofinancing_rate", "mean"),
            avg_adaptive_capacity=("country_ac_index", "mean")
        )
        .reset_index()
    )

    if df_plot.empty:
        print(f"Keine Daten für Länder-AC-Balkendiagramm: {title}")
        return

    df_plot["eu_share"] = df_plot["avg_cofinancing_rate"]
    df_plot["own_share"] = 1 - df_plot["avg_cofinancing_rate"]

    # Sortierung: niedrigste nationale AC links, höchste rechts
    df_plot = df_plot.sort_values("avg_adaptive_capacity")

    x = np.arange(len(df_plot))
    country_labels = df_plot["ms"].astype(str).values

    fig, ax = plt.subplots(figsize=(max(8, len(df_plot) * 0.4), 6))

    ax.bar(x, df_plot["own_share"], label="Eigenanteil", color="#cc7000")
    ax.bar(x, df_plot["eu_share"], bottom=df_plot["own_share"], label="EU-Anteil", color="#ffb347")

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(country_labels, rotation=45, ha="right")

    ax.set_ylabel("Anteil an Gesamtfinanzierung")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Länder-Balkendiagramm (nach nationaler Adaptive Capacity sortiert) gespeichert unter: {save_path}")

# ---------------------------------------------------------
# Neue Funktion: Gestapeltes Balkendiagramm, sortiert nach Cost of Capital Rank
# ---------------------------------------------------------
def plot_stacked_bar_cofinancing_coc(df, title, save_path):
    """
    Erwartet ein DataFrame mit Spalten:
    - avg_cofinancing_rate
    - cost_of_capital_rank
    - region_label

    Sortiert die Regionen nach cost_of_capital_rank und zeichnet
    einen 100%-Stacked-Bar: unten Eigenanteil, oben EU-Anteil.
    """
    df_plot = df.dropna(subset=["avg_cofinancing_rate", "cost_of_capital_rank"]).copy()
    if df_plot.empty:
        print(f"Keine Daten für gestapeltes Balkendiagramm (Cost of Capital): {title}")
        return

    df_plot["eu_share"] = df_plot["avg_cofinancing_rate"]
    df_plot["own_share"] = 1 - df_plot["avg_cofinancing_rate"]

    # Sortierung nach Cost of Capital Rank (standard: aufsteigend)
    df_plot = df_plot.sort_values("cost_of_capital_rank")

    x = np.arange(len(df_plot))
    region_labels = df_plot["region_label"].astype(str).values

    fig, ax = plt.subplots(figsize=(max(8, len(df_plot) * 0.25), 6))

    # Unterer Balken: Eigenanteil (dunkleres Orange)
    ax.bar(x, df_plot["own_share"], label="Eigenanteil", color="#cc7000")

    # Oberer Balken: EU-Anteil (helleres Orange)
    ax.bar(x, df_plot["eu_share"], bottom=df_plot["own_share"], label="EU-Anteil", color="#ffb347")

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=90, fontsize=6)

    ax.set_ylabel("Anteil an Gesamtfinanzierung")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Gestapeltes Balkendiagramm (Cost of Capital) gespeichert unter: {save_path}")


# ---------------------------------------------------------
# Helper: Zuordnung eines Index über region_label
# ---------------------------------------------------------
def make_index_with_fallback(df_vul, colname):
    """
    Baut ein Dictionary:
    - Code (NUTS2, NUTS1 oder Country) -> Index
    und liefert eine Funktion, die pro Zeile von df_reg
    den passenden Wert anhand von 'region_label' zuordnet.
    """

    code_to_index = (
        df_vul[["NUTS 2 Code", colname]]
        .dropna(subset=["NUTS 2 Code"])
        .groupby("NUTS 2 Code")[colname]
        .mean()
        .to_dict()
    )

    def getter(row):
        label = row.get("region_label", None)
        if pd.notna(label) and label in code_to_index:
            return code_to_index[label]
        return np.nan

    return getter


def main():
    # ---------------------------------------------------------
    # 1) Daten einlesen
    # ---------------------------------------------------------
    df_fin = pd.read_csv(csv_path)

    # Lade die Excel-Datei mit den Regionen-Namen
    df_vul = pd.read_excel(xlsx_path)

        # Lade die Excel-Datei, um die "Corresponding NUTS2 level"-Spalte zu bekommen
    df_nuts_names = pd.read_excel(xlsx_path)

    # Erstelle ein Mapping von NUTS-Codes zu den zugehörigen Bezeichnungen
    nuts_mapping = df_nuts_names.set_index("NUTS 2 Code")["Corresponding NUTS2 level"].to_dict()



    # SUPER WICHTIG: zuerst nach dimension_type == "Intervention Field" filtern
    if "dimension_type" not in df_fin.columns:
        raise ValueError("Spalte 'dimension_type' fehlt in der CSV-Datei.")
    df_fin = df_fin[df_fin["dimension_type"] == "Intervention Field"].copy()

    df_fin.columns = df_fin.columns.str.strip()

    # ---------------------------------------------------------
    # Neue Spalte: Proportion of JTF
    # ---------------------------------------------------------

    # eu_amount numerisch interpretieren (falls noch als String)
    df_fin["eu_amount"] = pd.to_numeric(df_fin["eu_amount"], errors="coerce")

    # Gesamtsumme der EU-Beträge (bereits nur Intervention Field, weil df_fin vorher gefiltert wurde)
    total_eu_if = df_fin["eu_amount"].sum()

    # Anteil jeder Zeile an der Gesamtsumme
    df_fin["Proportion of JTF"] = df_fin["eu_amount"] / total_eu_if

    df_vul = pd.read_excel(xlsx_path)
    df_vul.columns = df_vul.columns.str.strip()

    # ---------------------------------------------------------
    # Pflichtspalten prüfen
    # ---------------------------------------------------------
    required_fin_cols = ["NUTS 2 Code", "NUTS 1 Code", "ms", "eu_amount", "total_amount", "cofinancing_rate"]
    for col in required_fin_cols:
        if col not in df_fin.columns:
            raise ValueError(f"Spalte '{col}' fehlt in der CSV-Datei.")

    required_vul_cols_base = [
        "Country Code",
        "Country Name",
        "Corresponding NUTS2 level",
        "NUTS 2 Code",
    ]
    for col in required_vul_cols_base:
        if col not in df_vul.columns:
            raise ValueError(f"Spalte '{col}' fehlt in der Excel-Datei.")

    # zusätzlich für die beiden Indikatoren
    if "Index Adaptive Capacity" not in df_vul.columns:
        raise ValueError("Spalte 'Index Adaptive Capacity' fehlt in der Excel-Datei.")
    if "Ranking Cost of Capital Average Rank 2019" not in df_vul.columns:
        raise ValueError("Spalte 'Ranking Cost of Capital Average Rank 2019' fehlt in der Excel-Datei.")
    
        # ---------------------------------------------------------
    # Nationaler Adaptive-Capacity-Index pro Land (nur NUTS0-Zeilen)
    # ---------------------------------------------------------
    # Sicherstellen, dass NUTS-2-Codes als String vorliegen
    df_vul["NUTS 2 Code"] = df_vul["NUTS 2 Code"].astype(str).str.strip()

    # Nur Länder-Codes (z.B. AT, NL, BE): genau 2 Großbuchstaben, keine NUTS2-Codes wie NL11
    df_vul_nat = df_vul[
        df_vul["NUTS 2 Code"].str.match(r"^[A-Z]{2}$", na=False)
    ].copy()

    country_ac_map = (
        df_vul_nat
        .groupby("NUTS 2 Code")["Index Adaptive Capacity"]
        .mean()
        .to_dict()
    )


    # ---------------------------------------------------------
    # 2) Own Amount pro Projekt
    # ---------------------------------------------------------
    df_fin["own_amount"] = (1 - df_fin["cofinancing_rate"]) * df_fin["eu_amount"]

    # ---------------------------------------------------------
    # 3) Aggregation auf Regionen-Ebene (einmalig)
    # ---------------------------------------------------------
    group_cols = ["NUTS 2 Code", "NUTS 1 Code", "ms"]

    df_reg = (
        df_fin
        .groupby(group_cols, dropna=False)
        .agg(
            sum_eu_amount=("eu_amount", "sum"),
            sum_total_amount=("total_amount", "sum"),
            sum_own_amount=("own_amount", "sum"),
            n_projects=("cofinancing_rate", "size"),
            avg_own_amount=("own_amount", "mean"),
            jtf_share=("Proportion of JTF", "sum"),
        )
        .reset_index()
    )

    df_reg["avg_cofinancing_rate"] = df_reg["sum_eu_amount"] / df_reg["sum_total_amount"]


    # Region-Label (wird für beide Analysen benutzt)
    def choose_label(row):
        if pd.notna(row["NUTS 2 Code"]):
            return row["NUTS 2 Code"]
        elif pd.notna(row["NUTS 1 Code"]):
            return row["NUTS 1 Code"]
        else:
            return row["ms"]

    df_reg["region_label"] = df_reg.apply(choose_label, axis=1)
    # ---------------------------------------------------------
    # 4b) Hierarchischen Durchschnitt der cofinancing_rate berechnen
    #      NUTS2 -> NUTS1 -> Country (ms)
    # ---------------------------------------------------------
    # Sicherstellen, dass cofinancing_rate numerisch ist
    df_fin["cofinancing_rate"] = pd.to_numeric(df_fin["cofinancing_rate"], errors="coerce")
    df_fin_nonnull = df_fin[df_fin["cofinancing_rate"].notna()].copy()

    # Durchschnitt auf den drei Ebenen
    avg_n2 = df_fin_nonnull.groupby("NUTS 2 Code")["cofinancing_rate"].mean()
    avg_n1 = df_fin_nonnull.groupby("NUTS 1 Code")["cofinancing_rate"].mean()
    avg_n0 = df_fin_nonnull.groupby("ms")["cofinancing_rate"].mean()

    def calc_avg_cofin(label):
        """Gibt den hierarchischen Durchschnitt der cofinancing_rate für ein region_label zurück."""
        if pd.isna(label):
            return np.nan
        # 1) NUTS 2
        if label in avg_n2.index:
            return avg_n2.loc[label]
        # 2) NUTS 1
        if label in avg_n1.index:
            return avg_n1.loc[label]
        # 3) Country (NUTS 0)
        if label in avg_n0.index:
            return avg_n0.loc[label]
        return np.nan

    # Nationalen Adaptive-Capacity-Index je Land in df_reg mappen
    df_reg["country_ac_index"] = df_reg["ms"].map(country_ac_map)

    # ---------------------------------------------------------
    # 4c) Gestapeltes Balkendiagramm auf Länderebene (alphabetisch)
    # ---------------------------------------------------------
    country_bar_path = output_dir_stacked / "stacked_bar_eu_vs_own_share_by_country.pdf"
    plot_stacked_bar_cofinancing_by_country(
        df=df_reg,
        title="EU-Anteil vs. Eigenanteil an der Projektfinanzierung (Länderdurchschnitt)",
        save_path=country_bar_path
    )

    # ---------------------------------------------------------
    # 4d) Gestapeltes Balkendiagramm auf Länderebene,
    #     sortiert nach nationaler Adaptive Capacity
    # ---------------------------------------------------------
    country_ac_path = output_dir_stacked / "stacked_bar_country_sorted_by_national_adaptive_capacity.pdf"
    plot_stacked_bar_cofinancing_by_country_ac(
        df=df_reg,
        title="EU- und Eigenanteil auf Länderebene (sortiert nach nationaler Adaptive Capacity)",
        save_path=country_ac_path
    )



    # ---------------------------------------------------------
    # 5) ANALYSE 1: Adaptive Capacity
    # ---------------------------------------------------------
    ac_getter = make_index_with_fallback(df_vul, "Index Adaptive Capacity")
    df_reg["adaptive_capacity_index"] = df_reg.apply(ac_getter, axis=1)
    df_reg_ac = df_reg.dropna(subset=["adaptive_capacity_index"]).copy()


    # ---------------------------------------------------------
    # 5a) Gestapeltes Balkendiagramm: EU-Anteil vs. Eigenanteil,
    #     nach Adaptive Capacity sortiert
    # ---------------------------------------------------------
    stacked_path = output_dir_stacked / "stacked_bar_eu_vs_own_share_by_region_sorted_by_adaptive_capacity.pdf"
    plot_stacked_bar_cofinancing(
        df=df_reg_ac,
        title="EU-Anteil vs. Eigenanteil (nach Adaptive Capacity sortiert)",
        save_path=stacked_path
    )

    # Plots: Adaptive Capacity
    scatter_with_lowess(
        df=df_reg_ac,
        x_col="adaptive_capacity_index",
        y_col="avg_cofinancing_rate",
        title="Adaptive Capacity vs. durchschnittliche Cofinancing Rate (Regionen)",
        x_label="Adaptive Capacity Index",
        y_label="Durchschnittliche Cofinancing Rate",
        save_path=output_dir_cofin / "scatter_lowess_adaptive_capacity_vs_avg_cofinancing_rate.pdf",
        frac=0.3,
        label_col="region_label"
    )

           # Variante 1: automatische Entzerrung mit adjustText
    scatter_bubble_ownamount(
        df=df_reg_ac,
        x_col="adaptive_capacity_index",
        y_col="avg_cofinancing_rate",
        size_col="sum_own_amount",
        title=None,
        x_label="Adaptive Capacity Index",
        y_label="Average co-financing rate",
        save_path=output_dir_cofin_ownamount
        / "bubble_adaptive_capacity_vs_avg_cofinancing_rate_ownamount_adjusttext.pdf",
        label_col="region_label",
        label_mode="adjust",        # NEU
    )

    # Variante 2: einfache Offsets
    scatter_bubble_ownamount(
        df=df_reg_ac,
        x_col="adaptive_capacity_index",
        y_col="avg_cofinancing_rate",
        size_col="sum_own_amount",
        title=None,
        x_label="Adaptive Capacity Index",
        y_label="Average co-financing rate",
        save_path=output_dir_cofin_ownamount
        / "bubble_adaptive_capacity_vs_avg_cofinancing_rate_ownamount_offsetlabels.pdf",
        label_col="region_label",
        label_mode="offset",        # NEU
    )

    # Variante 3: nur Top-15 Regionen beschriften
    scatter_bubble_ownamount(
        df=df_reg_ac,
        x_col="adaptive_capacity_index",
        y_col="avg_cofinancing_rate",
        size_col="sum_own_amount",
        title=None,
        x_label="Adaptive Capacity Index",
        y_label="EU Co-Financing Rate (average)",
        save_path=output_dir_cofin_ownamount
        / "bubble_adaptive_capacity_vs_avg_cofinancing_rate_ownamount_top15labels.pdf",
        label_col="region_label",
        label_mode="topN",          # NEU
        label_top_n=15,             # NEU
        nuts_mapping=nuts_mapping  # Füge das Mapping als Parameter hinzu
    )
    # Neue Y-Spalte erzeugen (invertiert)
    df_reg_ac["avg_cofinancing_rate_inverted"] = 1 - df_reg_ac["avg_cofinancing_rate"]

    scatter_bubble_jtfshare(
        df=df_reg_ac,
        x_col="adaptive_capacity_index",
        y_col="avg_cofinancing_rate_inverted",   # <-- invertiert
        size_col="sum_own_amount",
        color_col="jtf_share",
        title=None,
        x_label="Adaptive Capacity Index",
        y_label="Co-Financing Rate (average)",      # <-- neuer Achsenname
        save_path=output_dir_cofin_ownamount
            / "bubble_adaptive_capacity_vs_avg_cofinancing_rate_jtfshare_top15labels.pdf",
        label_col="region_label",
        label_top_n=15,
        nuts_mapping=nuts_mapping,
    )


    # ------------------------------------------
    # Excel-Export: alle Daten, die in den Plot gehen
    # ------------------------------------------
    plot_cols = [
        "region_label",
        "adaptive_capacity_index",
        "avg_cofinancing_rate",
        "avg_cofinancing_rate_inverted",
        "sum_own_amount",
        "jtf_share",
        "ms",
        "NUTS 1 Code",
        "NUTS 2 Code",
        "sum_eu_amount",
        "sum_total_amount",
        "n_projects",
        "avg_own_amount",
    ]

    df_export = df_reg_ac[plot_cols].copy()

    # exakt wie scatter_bubble_jtfshare: dropna über die relevanten Spalten
    df_export = df_export.dropna(subset=[
        "adaptive_capacity_index",
        "avg_cofinancing_rate_inverted",
        "sum_own_amount",
        "jtf_share",
        "region_label",
    ])

    # exakt wie scatter_bubble_jtfshare: nur NUTS1 & NUTS2 (keine Länder-Codes)
    df_export = df_export[df_export["region_label"].astype(str).str.len() > 2].copy()

    # Region Name ergänzen (Mapping aus deiner Excel)
    df_export["region_name"] = df_export["region_label"].map(nuts_mapping)

    # optional: Sortierung für Lesbarkeit
    df_export = df_export.sort_values(["adaptive_capacity_index", "region_label"])

    # ======================================================
    # Excel-Export: Datensatz für Trendline-Bubbleplot (AC vs. Cofinancing, Farbe=JTF share)
    # ======================================================

    # Basename wie beim Plot-Export (nur ohne Endung)
    export_base = output_dir_cofin_ownamount / "bubble_adaptive_capacity_vs_avg_cofinancing_rate_jtfshare_top15labels_trendline"
    excel_export_path = Path(str(export_base) + ".xlsx")

    plot_cols = [
        "region_label",
        "adaptive_capacity_index",
        "avg_cofinancing_rate",
        "avg_cofinancing_rate_inverted",
        "sum_own_amount",
        "jtf_share",
        "ms",
        "NUTS 1 Code",
        "NUTS 2 Code",
        "sum_eu_amount",
        "sum_total_amount",
        "n_projects",
        "avg_own_amount",
    ]

    df_export = df_reg_ac[plot_cols].copy()

    # exakt wie scatter_bubble_jtfshare: dropna über relevante Spalten
    df_export = df_export.dropna(subset=[
        "region_label",
        "adaptive_capacity_index",
        "avg_cofinancing_rate_inverted",
        "sum_own_amount",
        "jtf_share",
    ])

    # exakt wie scatter_bubble_jtfshare: nur NUTS1 & NUTS2 (keine Länder-Codes)
    df_export = df_export[df_export["region_label"].astype(str).str.len() > 2].copy()

    # Region Name ergänzen
    df_export["region_name"] = df_export["region_label"].map(nuts_mapping)

    # OLS Trendline (wie in scatter_bubble_jtfshare bei add_trendline=True)
    X = sm.add_constant(df_export["adaptive_capacity_index"].values)
    y = df_export["avg_cofinancing_rate_inverted"].values
    ols_model = sm.OLS(y, X).fit()

    df_ols = pd.DataFrame([{
        "dependent": "avg_cofinancing_rate_inverted",
        "independent": "adaptive_capacity_index",
        "n": int(ols_model.nobs),
        "intercept": ols_model.params[0],
        "beta_x": ols_model.params[1],
        "p_intercept": ols_model.pvalues[0],
        "p_beta_x": ols_model.pvalues[1],
        "r_squared": ols_model.rsquared,
        "adj_r_squared": ols_model.rsquared_adj,
    }])

    with pd.ExcelWriter(excel_export_path, engine="openpyxl") as writer:
        df_export.to_excel(writer, sheet_name="plot_data", index=False)
        df_ols.to_excel(writer, sheet_name="ols_trendline", index=False)

    print(f"Excel-Export gespeichert unter: {excel_export_path}")

    scatter_bubble_jtfshare(
        df=df_reg_ac,
        x_col="adaptive_capacity_index",
        y_col="avg_cofinancing_rate_inverted",
        size_col="sum_own_amount",
        color_col="jtf_share",
        title=None,
        x_label="Adaptive Capacity Index",
        y_label="Co-Financing Rate (average)",
        save_path=output_dir_cofin_ownamount
            / "bubble_adaptive_capacity_vs_avg_cofinancing_rate_jtfshare_top15labels_trendline.pdf",
        label_col="region_label",
        label_top_n=15,
        nuts_mapping=nuts_mapping,
        add_trendline=True,      # <<< HIER aktivierst du die Linie
    )


    scatter_with_lowess(
        df=df_reg_ac,
        x_col="adaptive_capacity_index",
        y_col="sum_own_amount",
        title=None,
        x_label="Adaptive Capacity Index",
        y_label="Summe Eigenanteil (own amount)",
        save_path=output_dir_ownamount / "scatter_lowess_adaptive_capacity_vs_sum_own_amount.pdf",
        frac=0.3,
        label_col="region_label"
    )

    scatter_with_lowess(
        df=df_reg_ac,
        x_col="adaptive_capacity_index",
        y_col="avg_own_amount",
        title="None",
        x_label="Adaptive Capacity Index",
        y_label="Durchschnittlicher Eigenanteil pro Investment (own amount)",
        save_path=output_dir_avgownamount / "scatter_lowess_adaptive_capacity_vs_avg_own_amount.pdf",
        frac=0.3,
        label_col="region_label"
    )

    df_reg_ac.to_csv(
        output_dir_avgownamount / "regional_aggregates_with_adaptive_capacity_and_labels.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 6) ANALYSE 2: Cost of Capital Rank
    # ---------------------------------------------------------
    coc_col = "Ranking Cost of Capital Average Rank 2019"
    coc_getter = make_index_with_fallback(df_vul, coc_col)
    df_reg["cost_of_capital_rank"] = df_reg.apply(coc_getter, axis=1)
    df_reg_coc = df_reg.dropna(subset=["cost_of_capital_rank"]).copy()

        # Gestapeltes Balkendiagramm: EU-Anteil vs. Eigenanteil,
    # sortiert nach Cost of Capital Rank
    stacked_coc_path = output_dir_coc_bar / "stacked_bar_eu_vs_own_share_by_region_sorted_by_coc.pdf"
    plot_stacked_bar_cofinancing_coc(
        df=df_reg_coc,
        title="EU-Anteil vs. Eigenanteil (nach Cost of Capital Rank sortiert)",
        save_path=stacked_coc_path
    )


    scatter_with_lowess(
        df=df_reg_coc,
        x_col="cost_of_capital_rank",
        y_col="avg_cofinancing_rate",
        title="Cost of Capital Rank vs. durchschnittliche Cofinancing Rate (Regionen)",
        x_label="Cost of Capital Average Rank (2019)",
        y_label="Durchschnittliche Cofinancing Rate",
        save_path=output_dir_coc / "scatter_lowess_coc_vs_avg_cofinancing_rate.pdf",
        frac=0.3,
        label_col="region_label"
    )

    scatter_with_lowess(
        df=df_reg_coc,
        x_col="cost_of_capital_rank",
        y_col="sum_own_amount",
        title="Cost of Capital Rank vs. Summe Eigenanteil (own amount) pro Region",
        x_label="Cost of Capital Average Rank (2019)",
        y_label="Summe Eigenanteil (own amount)",
        save_path=output_dir_coc / "scatter_lowess_coc_vs_sum_own_amount.pdf",
        frac=0.3,
        label_col="region_label"
    )

    scatter_with_lowess(
        df=df_reg_coc,
        x_col="cost_of_capital_rank",
        y_col="avg_own_amount",
        title="Cost of Capital Rank vs. durchschnittlicher Eigenanteil pro Investment",
        x_label="Cost of Capital Average Rank (2019)",
        y_label="Durchschnittlicher Eigenanteil pro Investment (own amount)",
        save_path=output_dir_coc / "scatter_lowess_coc_vs_avg_own_amount.pdf",
        frac=0.3,
        label_col="region_label"
    )

    df_reg_coc.to_csv(
        output_dir_coc / "regional_aggregates_with_cost_of_capital_and_labels.csv",
        index=False
    )

        # ---------------------------------------------------------
    # Bubbleplots: Cost of Capital Rank vs. Co-Financing Rate
    # (wie bei Adaptive Capacity, nur andere x-Achse und anderes Output-Verzeichnis)
    # ---------------------------------------------------------

    # Y invertieren, analog zu df_reg_ac
    df_reg_coc["avg_cofinancing_rate_inverted"] = 1 - df_reg_coc["avg_cofinancing_rate"]

    # 1) Bubbleplot ohne Regressionslinie
    scatter_bubble_jtfshare(
        df=df_reg_coc,
        x_col="cost_of_capital_rank",                 # X-Achse: Cost of Capital Rank
        y_col="avg_cofinancing_rate_inverted",        # invertierte Cofinancing-Rate
        size_col="sum_own_amount",                    # Bubblegröße: Eigenanteile
        color_col="jtf_share",                        # Farbe: JTF-Share
        title=(
            "Cost of Capital Rank and Co-Financing Rate"
            "\n "
        ),
        x_label="Cost of Capital Rank (higher = better)",
        y_label="Co-Financing Rate (average)",
        save_path=output_dir_coc_bubbles
            / "bubble_coc_rank_vs_avg_cofinancing_rate_jtfshare_top15labels.pdf",
        label_col="region_label",
        label_top_n=15,
        nuts_mapping=nuts_mapping,
        add_trendline=False,                          # hier OHNE OLS-Linie
    )

    # 2) Bubbleplot MIT Regressionslinie (OLS)
    scatter_bubble_jtfshare(
        df=df_reg_coc,
        x_col="cost_of_capital_rank",
        y_col="avg_cofinancing_rate_inverted",
        size_col="sum_own_amount",
        color_col="jtf_share",
        title=(
            "Cost of Capital Rank and Co-Financing Rate"
            "\nWith Linear Regression Line (OLS)"
        ),
        x_label="Cost of Capital (higher = better)",
        y_label="Co-Financing Rate (average)",
        save_path=output_dir_coc_bubbles
            / "bubble_coc_rank_vs_avg_cofinancing_rate_jtfshare_top15labels_trendlinedf.pdf",
        label_col="region_label",
        label_top_n=15,
        nuts_mapping=nuts_mapping,
        add_trendline=True,                           # hier MIT OLS-Linie
    )

    # ---------------------------------------------------------
    # Spearman Correlation – Kombination in einer CSV
    # ---------------------------------------------------------
    from scipy.stats import spearmanr

    spearman_output_dir = Path(
        "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/06_Adaptive_Capacity/Spearman_Correlation"
    )
    spearman_output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1) Spearman: Cost of Capital Rank vs. avg_cofinancing_rate
    df_coc_corr = df_reg_coc.dropna(subset=["cost_of_capital_rank", "avg_cofinancing_rate"])
    rho_coc, pval_coc = spearmanr(df_coc_corr["cost_of_capital_rank"], df_coc_corr["avg_cofinancing_rate"])
    results.append({
        "Relationship": "Cost of Capital Rank vs. Average Cofinancing Rate",
        "Variable X": "cost_of_capital_rank",
        "Variable Y": "avg_cofinancing_rate",
        "Spearman_rho": rho_coc,
        "p_value": pval_coc,
        "N": len(df_coc_corr)
    })

    # 2) Spearman: Adaptive Capacity vs. avg_cofinancing_rate
    df_ac_corr = df_reg_ac.dropna(subset=["adaptive_capacity_index", "avg_cofinancing_rate"])
    rho_ac, pval_ac = spearmanr(df_ac_corr["adaptive_capacity_index"], df_ac_corr["avg_cofinancing_rate"])
    results.append({
        "Relationship": "Adaptive Capacity vs. Average Cofinancing Rate",
        "Variable X": "adaptive_capacity_index",
        "Variable Y": "avg_cofinancing_rate",
        "Spearman_rho": rho_ac,
        "p_value": pval_ac,
        "N": len(df_ac_corr)
    })

    df_spearman = pd.DataFrame(results)
    spearman_path = spearman_output_dir / "spearman_correlations_combined.csv"
    df_spearman.to_csv(spearman_path, index=False)
    print(f"Spearman-Korrelationen gespeichert unter: {spearman_path}")

    # ---------------------------------------------------------
    # Quadratische Regression für beide Beziehungen
    # ---------------------------------------------------------
    quad_results = []

    def quadratic_regression(df, x_col, y_col):
        df_clean = df.dropna(subset=[x_col, y_col]).copy()
        X = df_clean[x_col]
        X_quad = np.column_stack([X, X**2])
        X_quad = sm.add_constant(X_quad)
        model = sm.OLS(df_clean[y_col], X_quad).fit()
        return model, len(df_clean)

    # 1) Adaptive Capacity vs. avg cofinancing rate
    model_ac, n_ac = quadratic_regression(df_reg_ac, "adaptive_capacity_index", "avg_cofinancing_rate")
    quad_results.append({
        "Relationship": "Adaptive Capacity vs. Average Cofinancing Rate",
        "beta_0 (Intercept)": model_ac.params[0],
        "beta_1 (X)": model_ac.params[1],
        "beta_2 (X^2)": model_ac.params[2],
        "p_beta1": model_ac.pvalues[1],
        "p_beta2": model_ac.pvalues[2],
        "R_squared": model_ac.rsquared,
        "N": n_ac
    })

    # 2) Cost of Capital Rank vs. avg cofinancing rate
    model_coc, n_coc = quadratic_regression(df_reg_coc, "cost_of_capital_rank", "avg_cofinancing_rate")
    quad_results.append({
        "Relationship": "Cost of Capital Rank vs. Average Cofinancing Rate",
        "beta_0 (Intercept)": model_coc.params[0],
        "beta_1 (X)": model_coc.params[1],
        "beta_2 (X^2)": model_coc.params[2],
        "p_beta1": model_coc.pvalues[1],
        "p_beta2": model_coc.pvalues[2],
        "R_squared": model_coc.rsquared,
        "N": n_coc
    })

    quad_output_dir = Path(
        "/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/06_Adaptive_Capacity/Quadratic_Regression"
    )
    quad_output_dir.mkdir(parents=True, exist_ok=True)
    quad_output_path = quad_output_dir / "quadratic_regression_results.csv"
    pd.DataFrame(quad_results).to_csv(quad_output_path, index=False)
    print(f"Quadratische Regression gespeichert unter: {quad_output_path}")

    print("Fertig: Adaptive Capacity + Cost of Capital analysiert und geplottet.")



if __name__ == "__main__":
    main()
