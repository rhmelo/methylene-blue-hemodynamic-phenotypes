"""Source: original notebook cell 73 with the revised primary GEE result wired into the footnote. Run through run.py."""

from common import artifact, display, phase_dir, run_stamp

# ============================================================
# TABLE 2 FINAL — VERSÃO CORRIGIDA
#
# População:
#   pacientes vivos e completamente observados até 6 h
#
# Variáveis:
#   - Baseline NEE
#   - NEE at 6 h
#   - Baseline-adjusted 6-h NEE AUC
#
# Estatística:
#   median [IQR]
#   Kruskal-Wallis
#
# Nota:
#   Global phenotype-by-time interaction from the revised primary GEE
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import kruskal

# ------------------------------------------------------------
# 2. Caminhos
# ------------------------------------------------------------

PROJECT_DIR = phase_dir("mimic_iv")

TABLE_DIR = PROJECT_DIR / "tables"
DOCUMENT_DIR = PROJECT_DIR / "documents"
QC_DIR = PROJECT_DIR / "qc"

for folder in [TABLE_DIR, DOCUMENT_DIR, QC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def latest_file(pattern):
    return artifact(PROJECT_DIR, pattern)


def read_csv_auto(path):

    return pd.read_csv(
        path,
        compression=("gzip" if str(path).lower().endswith(".gz") else "infer"),
        low_memory=False,
    )


# ------------------------------------------------------------
# 3. Localizar arquivos
# ------------------------------------------------------------

INDIVIDUAL_PATH = latest_file("methylene_blue_6h_NEE_AUC_individual_*.csv.gz")

HOURLY_PATH = latest_file("methylene_blue_hourly_NEE_*.csv.gz")

GEE_PATH = (
    PROJECT_DIR
    / "revision_analyses"
    / "GEE_global_interaction_original_vs_no_post_death.csv"
)

print("=" * 100)
print("TABLE 2 FINAL — VERSÃO CORRIGIDA")
print("=" * 100)

print("\nBase individual:")
print(INDIVIDUAL_PATH)

print("\nBase horária:")
print(HOURLY_PATH)

print("\nResumo GEE:")
print(GEE_PATH)

# ------------------------------------------------------------
# 4. Carregar dados
# ------------------------------------------------------------

individual = read_csv_auto(INDIVIDUAL_PATH)

hourly = read_csv_auto(HOURLY_PATH)

gee_summary = read_csv_auto(GEE_PATH)

# ------------------------------------------------------------
# 5. Padronizar identificadores
# ------------------------------------------------------------

ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]

for dataframe in [individual, hourly]:

    for column in ID_COLUMNS:

        if column in dataframe.columns:

            dataframe[column] = pd.to_numeric(
                dataframe[column], errors="coerce"
            ).astype("Int64")

# ------------------------------------------------------------
# 6. Padronizar variáveis
# ------------------------------------------------------------

for column in ["nee_baseline", "auc_adjusted_for_baseline"]:

    individual[column] = pd.to_numeric(individual[column], errors="coerce")

hourly["hour_index"] = pd.to_numeric(hourly["hour_index"], errors="coerce")

hourly["nee_median"] = pd.to_numeric(hourly["nee_median"], errors="coerce")

# ------------------------------------------------------------
# 7. Elegibilidade
# ------------------------------------------------------------


def robust_boolean(series):

    numeric = pd.to_numeric(series, errors="coerce")

    text = series.astype(str).str.strip().str.lower()

    return numeric.eq(1) | text.isin(["true", "t", "yes", "y", "sim", "1", "1.0"])


individual["eligible_for_table2"] = robust_boolean(individual["eligible_auc_6h"])

analysis = individual[individual["eligible_for_table2"]].copy()

# ------------------------------------------------------------
# 8. Extrair NEE em 6 horas da base horária
#
# hour_index = 5 corresponde ao intervalo 5–6 h
# ------------------------------------------------------------

nee_6h = hourly[hourly["hour_index"].eq(5)].copy()

nee_6h = (
    nee_6h[ID_COLUMNS + ["nee_median"]]
    .rename(columns={"nee_median": "nee_6h"})
    .drop_duplicates(subset=ID_COLUMNS, keep="first")
)

# ------------------------------------------------------------
# 9. Merge
# ------------------------------------------------------------

analysis = analysis.merge(nee_6h, on=ID_COLUMNS, how="left", validate="one_to_one")

analysis["nee_6h"] = pd.to_numeric(analysis["nee_6h"], errors="coerce")

# ------------------------------------------------------------
# 10. Fenótipos
# ------------------------------------------------------------

PHENOTYPE_ORDER = [
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
    "Preserved diastolic pressure",
]

analysis = analysis[analysis["hemodynamic_phenotype"].isin(PHENOTYPE_ORDER)].copy()

analysis["hemodynamic_phenotype"] = pd.Categorical(
    analysis["hemodynamic_phenotype"], categories=PHENOTYPE_ORDER, ordered=True
)

# ------------------------------------------------------------
# 11. Auditoria de disponibilidade
# ------------------------------------------------------------

availability = pd.DataFrame(
    {
        "variable": [
            "Eligible patients",
            "Baseline NEE available",
            "NEE at 6 h available",
            "Adjusted AUC available",
        ],
        "n": [
            len(analysis),
            analysis["nee_baseline"].notna().sum(),
            analysis["nee_6h"].notna().sum(),
            analysis["auc_adjusted_for_baseline"].notna().sum(),
        ],
    }
)

print("\n" + "=" * 100)
print("DISPONIBILIDADE")
print("=" * 100)

display(availability)

if analysis["nee_6h"].isna().any():

    print("\nAVISO: existem pacientes elegíveis sem NEE em 6 h.")

    display(
        analysis.loc[analysis["nee_6h"].isna(), ID_COLUMNS + ["hemodynamic_phenotype"]]
    )

# ------------------------------------------------------------
# 12. Funções estatísticas
# ------------------------------------------------------------


def summarize_variable(dataframe, variable):

    rows = []

    for phenotype in PHENOTYPE_ORDER:

        values = (
            dataframe.loc[dataframe["hemodynamic_phenotype"].eq(phenotype), variable]
            .dropna()
            .astype(float)
        )

        rows.append(
            {
                "phenotype": phenotype,
                "n": len(values),
                "median": values.median(),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
            }
        )

    return pd.DataFrame(rows)


def kruskal_by_phenotype(dataframe, variable):

    groups = [
        dataframe.loc[dataframe["hemodynamic_phenotype"].eq(phenotype), variable]
        .dropna()
        .to_numpy(dtype=float)
        for phenotype in PHENOTYPE_ORDER
    ]

    if any(len(group) == 0 for group in groups):

        return np.nan, np.nan

    result = kruskal(*groups)

    return (float(result.statistic), float(result.pvalue))


def format_median_iqr(median, p25, p75):

    return f"{median:.2f} " f"[{p25:.2f}–{p75:.2f}]"


def format_p(value):

    if pd.isna(value):
        return "NA"

    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


# ------------------------------------------------------------
# 13. Calcular resumos e testes
# ------------------------------------------------------------

VARIABLES = {
    "nee_baseline": "Baseline norepinephrine-equivalent dose, µg/kg/min",
    "nee_6h": "Norepinephrine-equivalent dose at 6 h, µg/kg/min",
    "auc_adjusted_for_baseline": "Baseline-adjusted 6-h NEE AUC, µg/kg/min × h",
}

summaries = {}

tests = []

for variable in VARIABLES:

    summaries[variable] = summarize_variable(analysis, variable)

    statistic, p_value = kruskal_by_phenotype(analysis, variable)

    tests.append(
        {
            "variable": variable,
            "test": "Kruskal-Wallis",
            "statistic": statistic,
            "p_value": p_value,
        }
    )

test_results = pd.DataFrame(tests)

# ------------------------------------------------------------
# 14. GEE
# ------------------------------------------------------------

gee_row = gee_summary.loc[
    gee_summary["model"].eq("Without intervals entirely after death"), "p_value"
]
if len(gee_row) != 1:
    raise ValueError("Expected one primary post-death-excluded GEE result.")
gee_p = float(gee_row.iloc[0])

# ------------------------------------------------------------
# 15. Montar Table 2
# ------------------------------------------------------------

group_n = analysis.groupby("hemodynamic_phenotype", observed=True).size().to_dict()

COLUMN_NAMES = {
    "Vasoplegia-compatible": (
        "Vasoplegia-compatible " f"(n={group_n['Vasoplegia-compatible']})"
    ),
    "Mixed/low-pulsatility": (
        "Mixed/low-pulsatility " f"(n={group_n['Mixed/low-pulsatility']})"
    ),
    "Preserved diastolic pressure": (
        "Preserved diastolic pressure " f"(n={group_n['Preserved diastolic pressure']})"
    ),
}

table_rows = []

for variable, label in VARIABLES.items():

    variable_summary = summaries[variable].set_index("phenotype")

    p_value = test_results.loc[test_results["variable"].eq(variable), "p_value"].iloc[0]

    row = {"Variable": label, "P value": format_p(p_value)}

    for phenotype in PHENOTYPE_ORDER:

        values = variable_summary.loc[phenotype]

        row[COLUMN_NAMES[phenotype]] = format_median_iqr(
            values["median"], values["p25"], values["p75"]
        )

    table_rows.append(row)

table_columns = [
    "Variable",
    COLUMN_NAMES["Vasoplegia-compatible"],
    COLUMN_NAMES["Mixed/low-pulsatility"],
    COLUMN_NAMES["Preserved diastolic pressure"],
    "P value",
]

table2 = pd.DataFrame(table_rows, columns=table_columns)

# ------------------------------------------------------------
# 16. Mostrar
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("TABLE 2")
print("=" * 100)

display(table2)

print("\n" + "=" * 100)
print("TESTES")
print("=" * 100)

display(test_results)

print("\nGlobal phenotype-by-time interaction, GEE: " f"P = {gee_p:.3f}")

# ------------------------------------------------------------
# 17. Criar DOCX
# ------------------------------------------------------------

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

document = Document()

section = document.sections[0]

section.top_margin = Inches(0.7)
section.bottom_margin = Inches(0.7)
section.left_margin = Inches(0.55)
section.right_margin = Inches(0.55)

title = document.add_paragraph()

title_run = title.add_run(
    "Table 2. Early physiological response according " "to macrohemodynamic phenotype"
)

title_run.bold = True
title_run.font.name = "Arial"
title_run.font.size = Pt(10)

word_table = document.add_table(rows=1, cols=len(table_columns))

word_table.alignment = WD_TABLE_ALIGNMENT.CENTER
word_table.style = "Table Grid"

for index, column in enumerate(table_columns):

    cell = word_table.rows[0].cells[index]

    cell.text = column

    for paragraph in cell.paragraphs:

        paragraph.alignment = (
            WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
        )

        for run in paragraph.runs:

            run.bold = True
            run.font.name = "Arial"
            run.font.size = Pt(8)

for _, data_row in table2.iterrows():

    cells = word_table.add_row().cells

    for index, column in enumerate(table_columns):

        cells[index].text = str(data_row[column])

        for paragraph in cells[index].paragraphs:

            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )

            for run in paragraph.runs:

                run.font.name = "Arial"
                run.font.size = Pt(8)

note = document.add_paragraph()

note.paragraph_format.space_before = Pt(5)

note_run = note.add_run(
    "Values are median [interquartile range]. "
    "Continuous variables were compared using the "
    "Kruskal-Wallis test. The analysis included patients "
    "alive and completely observed through 6 hours. "
    f"The global phenotype-by-time interaction in the primary "
    f"generalized estimating equation model was "
    f"P = {gee_p:.3f}. "
    "AUC, area under the curve; GEE, generalized estimating "
    "equations; NEE, norepinephrine-equivalent dose."
)

note_run.font.name = "Arial"
note_run.font.size = Pt(8)

# ------------------------------------------------------------
# 18. Salvar
# ------------------------------------------------------------

timestamp = run_stamp()

csv_path = TABLE_DIR / f"Table_2_FINAL_physiological_response_{timestamp}.csv"

docx_path = DOCUMENT_DIR / f"Table_2_FINAL_physiological_response_{timestamp}.docx"

qc_path = QC_DIR / f"Table_2_FINAL_tests_{timestamp}.csv"

availability_path = QC_DIR / f"Table_2_FINAL_availability_{timestamp}.csv"

table2.to_csv(csv_path, index=False)

test_results.to_csv(qc_path, index=False)

availability.to_csv(availability_path, index=False)

document.save(docx_path)

print("\n" + "=" * 100)
print("TABLE 2 CONCLUÍDA")
print("=" * 100)

print("\nCSV:")
print(csv_path)

print("\nDOCX:")
print(docx_path)

print("\nQC:")
print(qc_path)

print("\nDisponibilidade:")
print(availability_path)
