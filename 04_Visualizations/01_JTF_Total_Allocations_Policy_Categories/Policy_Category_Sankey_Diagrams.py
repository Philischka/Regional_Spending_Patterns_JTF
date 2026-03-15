import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np



# ================================================================
# Pfade
# ================================================================
input_path = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/04_JTF_Finances_Details/Data_used/2021-2027_JTF_Finances_Details_with_NUTS_20251115.csv"

output_dir = r"/Users/philina/Desktop/TUM/4. Semester/Masterarbeit/01_MA Vulnerability Framework/07_Visualizations/00_Publication/02_BarChart_PolicyCategories"
os.makedirs(output_dir, exist_ok=True)

output_intervention_html = os.path.join(output_dir, "Sankey_JTF_Finances_Intervention_Fields.html")
output_policy_html = os.path.join(output_dir, "Sankey_JTF_Finances_Policy_Category.html")

output_intervention_pdf = os.path.join(output_dir, "Sankey_JTF_Finances_Intervention_Fields.pdf")
output_policy_pdf = os.path.join(output_dir, "Sankey_JTF_Finances_Policy_Category.pdf")


# ================================================================
# Farb-Definitionen (pastellig)
# ================================================================
policy_color_map = {
    "Decarbonization": "#002060",
    "Innovation and Economic Transformation": "#002060",
    "Social Infrastructure": "#002060",
    "Technical Assistance": "#002060",
    "Labour and Reskilling": "#002060",
    "Mobility and Infrastructure": "#002060",
    "Local communities/tourism/culture and natural heritage":  "#002060",
}

default_color = "#002060"


# ================================================================
# CSV einlesen
# ================================================================
df = pd.read_csv(input_path, encoding="utf-8-sig")
# Nur Intervention-Field-Einträge verwenden
df = df[df["dimension_type"] == "Intervention Field"]


col_nuts2 = "NUTS 2 Code"
col_nuts1 = "NUTS 1 Code"
col_ms = "ms"
col_amount = "eu_amount"
col_intervention = "Intervention Field"
col_policy = "Policy Category Title"


# ================================================================
# Kürzere Namen für Policy Categories (Mapping)
# ================================================================
policy_rename_map = {
    "Innovation and Economic Transformation": "Innovation & Econ. Transformation",
    "Local communities/tourism/culture and natural heritage": "Community & Heritage",
    "Labour and Reskilling": "Labour & Reskilling",
    "Social Infrastructure": "Social Infrastructure",
    "Technical Assistance": "Technical Assistance",
    "Mobility and Infrastructure": "Mobility & Infrastructure",
    "Decarbonization": "Decarbonization",
}

# Policy Category umbenennen (nur wenn gematcht, sonst Original behalten)
df[col_policy] = df[col_policy].replace(policy_rename_map)


# ================================================================
# Region-Fallback erstellen
# ================================================================
df[[col_nuts2, col_nuts1, col_ms]] = df[[col_nuts2, col_nuts1, col_ms]].replace("", pd.NA)

df["region"] = df[col_nuts2]
df["region"] = df["region"].fillna(df[col_nuts1])
df["region"] = df["region"].fillna(df[col_ms])

df = df.dropna(subset=["region"])

df[col_amount] = pd.to_numeric(df[col_amount], errors="coerce").fillna(0)
df = df[df[col_amount] > 0]

# ================================================================
# EXPORT: all data used for figures (Excel)
# ================================================================
export_xlsx = os.path.join(output_dir, "figure_data_export.xlsx")

# Basisdaten, die in alle Aggregationen eingehen
df_base = df.copy()

with pd.ExcelWriter(export_xlsx, engine="openpyxl") as writer:
    df_base.to_excel(writer, sheet_name="base_after_cleaning", index=False)

print("Saved figure data Excel (base) to:", export_xlsx)


# ================================================================
# Sankey-Funktion (NEU mit Zielknoten-Farbe)
# ================================================================
def create_sankey(df_group, source_col, target_col, value_col,
                  title, output_html, output_pdf,
                  link_colors=None, target_color_map=None):

    df_group = df_group.dropna(subset=[source_col, target_col]).copy()

    # --- 1) SOURCES (links) sortieren: Summe eu_amount absteigend
    source_totals = (
        df_group.groupby(source_col, as_index=True)[value_col]
                .sum()
                .sort_values(ascending=False)
    )
    sources = source_totals.index.astype(str).tolist()

    # --- 2) TARGETS (rechts) sortieren: Summe eu_amount absteigend
    targets = df_group[target_col].astype(str).unique().tolist()


    # --- 3) Labels + Mapping (einmal, sauber)
    labels = sources + targets
    label_to_index = {label: i for i, label in enumerate(labels)}

    # Links: Indizes
    source_indices = df_group[source_col].astype(str).map(label_to_index)
    target_indices = df_group[target_col].astype(str).map(label_to_index)
    values = df_group[value_col].astype(float)

    # --- 4) Node-Farben
    node_colors = []
    for label in labels:
        if target_color_map and label in target_color_map:
            node_colors.append(target_color_map[label])
        else:
            node_colors.append("#dddddd")

    # --- 5) Link-Farben (WICHTIG: Länge muss exakt = Anzahl Links sein)
    link_dict = dict(
        source=source_indices,
        target=target_indices,
        value=values,
        color=list(link_colors) if link_colors is not None else None
    )

    # --- 6) FIXED-Layout: x/y für alle Nodes so, dass nichts überlappt
    n_sources = len(sources)
    n_targets = len(targets)

    node_x = [0.01] * n_sources + [0.99] * n_targets  # minimal eingerückt

    def spaced_y(n, top=0.05, bottom=0.95):
        if n <= 1:
            return [0.5]
        step = (bottom - top) / (n - 1)
        return [top + i * step for i in range(n)]


    # y=0 oben, y=1 unten -> größte zuerst = oben
    node_y = (
        spaced_y(n_sources, top=0.10, bottom=0.88)   # LINKS (wie zuletzt)
        + spaced_y(n_targets, top=0.25, bottom=0.75) # RECHTS: klar mittig
    )





    # --- 7) Dynamische Figure-Höhe gegen Overlap (insb. viele Intervention Fields)
    # Faustregel: pro Knoten etwas vertikale Fläche reservieren
    n_max = max(n_sources, n_targets)
    fig_height = max(800, min(4000, 18 * n_max + 200))  # capped

    # Weniger dick/pad reduziert Überlappungen drastisch
    node_thickness = 12
    node_pad = 8

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=node_pad,
                    thickness=node_thickness,
                    line=dict(color="black", width=0.4),
                    label=labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y
                ),
                link=link_dict
            )
        ]
    )

    fig.update_layout(
        title_text=title,
        font_size=14,
        width=1400,
        height=fig_height,
        margin=dict(l=140, r=80, t=70, b=120)  # <- mehr Platz links & unten
    )


    fig.write_html(output_html)
    print("HTML gespeichert:", output_html)

    try:
        fig.write_image(output_pdf, format="pdf")
        print("PDF gespeichert:", output_pdf)
    except Exception as e:
        print("PDF Fehler:", e)



# ================================================================
# 1) Sankey Region -> Intervention Field (Links nach Policy färben)
# ================================================================
df_intervention = (
    df.dropna(subset=[col_intervention, col_policy])
      .groupby(["region", col_intervention, col_policy], as_index=False)[col_amount]
      .sum()
)

df_intervention["policy_color"] = (
    df_intervention[col_policy].map(policy_color_map).fillna(default_color)
)

target_color_map_intervention = {p: c for p, c in policy_color_map.items()}

create_sankey(
    df_group=df_intervention,
    source_col="region",
    target_col=col_intervention,
    value_col=col_amount,
    title="JTF-Finances: Region → Intervention Field",
    output_html=output_intervention_html,
    output_pdf=output_intervention_pdf,
    link_colors=df_intervention["policy_color"],
    target_color_map=target_color_map_intervention
)


# ================================================================
# 2) Sankey Region -> Policy Category Title (Links + Ziel färben)
# ================================================================
df_policy = (
    df.dropna(subset=[col_policy])
      .groupby(["region", col_policy], as_index=False)[col_amount]
      .sum()
)

df_policy["policy_color"] = (
    df_policy[col_policy].map(policy_color_map).fillna(default_color)
)

target_color_map_policy = {p: c for p, c in policy_color_map.items()}

create_sankey(
    df_group=df_policy,
    source_col="region",
    target_col=col_policy,
    value_col=col_amount,
    title="JTF-Finances: Region → Policy Category Title",
    output_html=output_policy_html,
    output_pdf=output_policy_pdf,
    link_colors=df_policy["policy_color"],
    target_color_map=target_color_map_policy
)

# # ================================================================
# 3) Financial Flows: Gesamtmittel pro Policy Category (kein Sankey)
# ================================================================

# Neue Output-Pfade
output_policy_flow_pdf = os.path.join(output_dir, "Flow_JTF_Finances_Policy_Category.pdf")
output_policy_flow_html = os.path.join(output_dir, "Flow_JTF_Finances_Policy_Category.html")
output_policy_flow_pdf  = os.path.join(output_dir, "Flow_JTF_Finances_Policy_Category.pdf")

# ------------------------------------------------
# A) Aggregation: Summe eu_amount je Policy Category
# ------------------------------------------------
df_policy_total = (
    df.dropna(subset=[col_policy])
      .groupby(col_policy, as_index=False)[col_amount]
      .sum()
)

# ------------------------------------------------
# B) Distinct Count: wie viele verschiedene Intervention-Field-CODES
#    pro Policy Category einfließen (NICHT summieren!)
# ------------------------------------------------
df_counts = (
    df.dropna(subset=[col_policy, col_intervention])
      .assign(**{col_intervention: pd.to_numeric(df[col_intervention], errors="coerce")})
      .dropna(subset=[col_intervention])
      .groupby(col_policy)[col_intervention]
      .nunique()
      .reset_index(name="n_intervention_codes")
)

# Counts an die Policy-Totals hängen
df_policy_total = df_policy_total.merge(df_counts, on=col_policy, how="left")
df_policy_total["n_intervention_codes"] = df_policy_total["n_intervention_codes"].fillna(0).astype(int)

# ------------------------------------------------
# C) Gesamtsumme + Shares
# ------------------------------------------------
total_amount = df_policy_total[col_amount].sum()
total_amount_mil = total_amount / 1e6
print(f"Gesamtsumme JTF-Mittel: {total_amount_mil:.2f} M€")

df_policy_total["share"] = df_policy_total[col_amount] / total_amount * 100
df_policy_total["amount_million"] = df_policy_total[col_amount] / 1e6

# Sortierung nach Betrag (aufsteigend für saubere horizontale Darstellung)
df_policy_total = df_policy_total.sort_values(by=col_amount, ascending=True)

# Farben wie oben definiert
df_policy_total["color"] = df_policy_total[col_policy].map(policy_color_map).fillna(default_color)

# ------------------------------------------------
# D) Labels (Policy wrapped, Amount im Balken, Count rechts daneben)
# ------------------------------------------------
def wrap_label(label, width=15):
    import textwrap
    return "<br>".join(textwrap.fill(str(label), width=width).split("\n"))

df_policy_total["label_wrapped"] = df_policy_total[col_policy].apply(lambda x: wrap_label(x, width=18))

# Amount-Text: im Balken
df_policy_total["amount_text"] = df_policy_total["amount_million"].round(0).astype(int).astype(str) + " M€"


# Count-Text: rechts hinter dem Balken
df_policy_total["count_text"] = df_policy_total["n_intervention_codes"].apply(lambda n: f"({int(n)})")


# ------------------------------------------------
# E) Plot als horizontales Balkendiagramm
# ------------------------------------------------
fig_flow = go.Figure()

max_x = df_policy_total["amount_million"].max()

# 1) Regel, wann Amount in den Balken darf
#    Stellschraube: größer -> weniger "inside", mehr "outside"
min_inside_factor = 0.13
min_inside = min_inside_factor * max_x

is_inside = df_policy_total["amount_million"] >= min_inside

# Amount nur für "inside" im Bar-Trace, sonst leer lassen
df_policy_total["amount_text_inside"] = np.where(is_inside, df_policy_total["amount_text"], "")

# 2) Rechter Textblock
#    - wenn Amount inside: nur IF-Codes rechts
#    - wenn Amount outside: Amount + IF-Codes untereinander rechts
df_policy_total["right_text"] = np.where(
    is_inside,
    "(" + df_policy_total["n_intervention_codes"].astype(int).astype(str) + ")",
    df_policy_total["amount_text"] + "<br>(" + df_policy_total["n_intervention_codes"].astype(int).astype(str) + ")"
)


# 3) Balken (Amount inside, wenn erlaubt)
fig_flow.add_trace(
    go.Bar(
        x=df_policy_total["amount_million"],
        y=df_policy_total["label_wrapped"],
        orientation="h",
        marker=dict(color=df_policy_total["color"]),
        text=df_policy_total["amount_text_inside"],
        textposition="inside",
        insidetextanchor="middle",
        cliponaxis=False,     # verhindert Abschneiden bei Text
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>" +
                      "Amount: %{x:.0f} Million €<br>" +
                      "Share: %{customdata[0]:.1f}%<br>" +
                      "Distinct IF-Codes: %{customdata[1]}<extra></extra>",
        customdata=list(zip(df_policy_total["share"], df_policy_total["n_intervention_codes"]))
    )
)


# 4) Rechter Text (als Scatter), damit nichts rotiert oder überlappt
offset_factor = 0.01     # größer -> weiter nach rechts
offset = offset_factor * max_x

annotations = [
    dict(
        x=row["amount_million"] + offset,
        y=row["label_wrapped"],
        xref="x",
        yref="y",
        text=row["right_text"],
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        align="left"  # das ist der entscheidende Punkt für linksbündig
    )
    for _, row in df_policy_total.iterrows()
]

fig_flow.update_layout(annotations=annotations)


fig_flow.update_layout(
    title=None,
    xaxis_title=None,
    font=dict(
        size=20,
        color="black"
    ),
    margin=dict(l=200, r=300, t=60, b=60),
    width=1500,
    height=900,
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white"
)

# Achsen: nur graues Raster
fig_flow.update_xaxes(
    showgrid=True,
    gridcolor="lightgrey",
    zeroline=False,
    tickformat=",.0f",
    title_text="Million €"
)
fig_flow.update_yaxes(
    showline=True,
    linecolor="lightgrey",
    linewidth=1,
    showgrid=False,
    zeroline=False
)

fig_flow.update_xaxes(showline=True, linecolor="lightgrey", linewidth=1)

# x-Achse erweitern, damit der rechte Text Platz hat
fig_flow.update_xaxes(range=[0, max_x * (1 + offset_factor + 0.25)])

fig_flow.update_layout(margin=dict(l=260, r=300, t=20, b=20))

# ------------------------------------------------
# F) Export
# ------------------------------------------------
output_policy_flow_html = os.path.join(output_dir, "Flow_JTF_Finances_Policy_Category.html")
output_policy_flow_pdf  = os.path.join(output_dir, "Flow_JTF_Finances_Policy_Category.pdf")

# HTML (zum Prüfen)
fig_flow.write_html(output_policy_flow_html)
print("HTML gespeichert:", output_policy_flow_html)

# Für PDF eine fixe Größe setzen
fig_flow.update_layout(width=1200, height=800)

try:
    fig_flow.write_image(output_policy_flow_pdf, format="pdf")
    print("PDF gespeichert:", output_policy_flow_pdf)
except Exception as e:
    print("PDF Fehler (Flow Policy Category):", e)
