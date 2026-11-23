"""Source: original notebook cell 52. Run through run.py."""

from common import artifact, display, phase_dir, run_stamp

# ============================================================
# CÓDIGO 12B — TABLE 1 FINAL CORRIGIDA
#
# Correções incorporadas:
#   - Azul de metileno em mg/kg
#   - Vasopressina removida da tabela
#   - Sem p para ventilação mecânica
#   - Sem p para PAD, PAM e PP
#   - Kruskal–Wallis para contínuas
#   - Qui-quadrado quando adequado
#   - Teste exato de Monte Carlo para tabelas esparsas
#
# Outputs:
#   - Table 1 em CSV
#   - Table 1 em DOCX
#   - Log estatístico
#   - Methods draft em DOCX
#   - Results draft em DOCX e TXT
#   - Base analítica congelada
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# ------------------------------------------------------------
# 1. Caminhos
# ------------------------------------------------------------

PROJECT_DIR = phase_dir("mimic_iv")

COHORT_DIR = PROJECT_DIR / "cohorts"
TABLE_DIR = PROJECT_DIR / "tables"
QC_DIR = PROJECT_DIR / "qc"
DOCUMENT_DIR = PROJECT_DIR / "documents"

for directory in [COHORT_DIR, TABLE_DIR, QC_DIR, DOCUMENT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Localizar a base com dose em mg/kg
# ------------------------------------------------------------

BASE_PATH = artifact(PROJECT_DIR, "TABLE1_BASE_FINAL_COMPLETE_WITH_MB_MGKG_*.csv.gz")

print("=" * 100)
print("CÓDIGO 12B — TABLE 1 FINAL CORRIGIDA")
print("=" * 100)

print("\nBase selecionada:")
print(BASE_PATH)

# ------------------------------------------------------------
# 3. Carregar a base
# ------------------------------------------------------------

df = pd.read_csv(BASE_PATH, compression="gzip", low_memory=False)

df = df.drop_duplicates(subset=["subject_id"], keep="first").copy()

PHENOTYPE_ORDER = [
    "Vasoplegia-compatible",
    "Mixed/low-pulsatility",
    "Preserved diastolic pressure",
]

df = df[df["hemodynamic_phenotype"].isin(PHENOTYPE_ORDER)].copy()

df["hemodynamic_phenotype"] = pd.Categorical(
    df["hemodynamic_phenotype"], categories=PHENOTYPE_ORDER, ordered=True
)

print("\nPacientes:", len(df))

print("\nFenótipos:")
display(
    df["hemodynamic_phenotype"]
    .value_counts(sort=False)
    .rename_axis("phenotype")
    .reset_index(name="n")
)

# ------------------------------------------------------------
# 4. Mapeamento das colunas
# ------------------------------------------------------------

COL = {
    "age": "age_at_admission",
    "sex": "sex",
    "hours_icu_to_mb": "hours_icu_to_mb",
    "mb_dose_mgkg": "mb_dose_mg_per_kg",
    "lactate": "lactate_pre_mb",
    "creatinine": "creatinine_pre_mb",
    "bilirubin": "bilirubin_total_pre_mb",
    "platelets": "platelets_pre_mb",
    "mechanical_ventilation": "invasive_mechanical_ventilation_at_mb",
    "rrt": "rrt_at_mb_baseline",
    "nee": "nee_baseline_median",
    "epinephrine": "epinephrine_active_baseline",
    "dbp": "dbp_pre_mb_median",
    "map": "map_pre_mb_median",
    "pp": "pp_pre_mb_median",
    "heart_rate": "hr_pre_mb_median",
}

missing_columns = [column for column in COL.values() if column not in df.columns]

if missing_columns:
    raise KeyError(f"Colunas ausentes na base: {missing_columns}")

print("\nMapeamento confirmado:")

for concept, column in COL.items():
    print(f"{concept:<30} -> {column}")

# ------------------------------------------------------------
# 5. Padronizar tipos
# ------------------------------------------------------------

continuous_columns = [
    COL["age"],
    COL["hours_icu_to_mb"],
    COL["mb_dose_mgkg"],
    COL["lactate"],
    COL["creatinine"],
    COL["bilirubin"],
    COL["platelets"],
    COL["nee"],
    COL["dbp"],
    COL["map"],
    COL["pp"],
    COL["heart_rate"],
]

for column in continuous_columns:

    df[column] = pd.to_numeric(df[column], errors="coerce")


def to_boolean(series):

    normalized = series.astype(str).str.strip().str.lower()

    output = pd.Series(pd.NA, index=series.index, dtype="boolean")

    output.loc[normalized.isin(["true", "1", "yes", "y"])] = True

    output.loc[normalized.isin(["false", "0", "no", "n"])] = False

    return output


for column in [COL["mechanical_ventilation"], COL["rrt"], COL["epinephrine"]]:

    df[column] = to_boolean(df[column])

sex_normalized = df[COL["sex"]].astype(str).str.strip().str.upper()

df["female"] = pd.Series(
    np.where(
        sex_normalized.isin(["F", "FEMALE"]),
        True,
        np.where(sex_normalized.isin(["M", "MALE"]), False, pd.NA),
    ),
    index=df.index,
    dtype="boolean",
)

# ------------------------------------------------------------
# 6. Funções de apresentação
# ------------------------------------------------------------


def format_continuous(series, decimals=1):

    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return "NA"

    median = values.median()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    return f"{median:.{decimals}f} " f"[{q1:.{decimals}f}–{q3:.{decimals}f}]"


def format_binary(series):

    values = series.dropna()

    denominator = len(values)

    if denominator == 0:
        return "NA"

    numerator = int((values == True).sum())

    percentage = 100 * numerator / denominator

    return f"{numerator}/{denominator} " f"({percentage:.1f})"


def format_p(p_value):

    if pd.isna(p_value):
        return "—"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.3f}"


# ------------------------------------------------------------
# 7. Kruskal–Wallis
# ------------------------------------------------------------


def kruskal_p(dataframe, column):

    groups = []

    for phenotype in PHENOTYPE_ORDER:

        values = pd.to_numeric(
            dataframe.loc[dataframe["hemodynamic_phenotype"].eq(phenotype), column],
            errors="coerce",
        ).dropna()

        groups.append(values)

    if any(len(group) == 0 for group in groups):
        return np.nan

    try:

        return stats.kruskal(*groups).pvalue

    except ValueError:

        return np.nan


# ------------------------------------------------------------
# 8. Testes categóricos
# ------------------------------------------------------------

MONTE_CARLO_REPLICATES = 20000
RANDOM_SEED = 20260712


def pearson_statistic(table):

    table = np.asarray(table, dtype=float)

    row_totals = table.sum(axis=1, keepdims=True)

    col_totals = table.sum(axis=0, keepdims=True)

    total = table.sum()

    if total == 0:
        return np.nan

    expected = row_totals @ col_totals / total

    valid = expected > 0

    return np.sum((table[valid] - expected[valid]) ** 2 / expected[valid])


def categorical_test(dataframe, column):

    analysis = dataframe[["hemodynamic_phenotype", column]].dropna().copy()

    if analysis.empty:
        return np.nan, "Not available"

    analysis[column] = analysis[column].astype(bool).astype(int)

    observed_table = pd.crosstab(
        analysis["hemodynamic_phenotype"], analysis[column]
    ).reindex(index=PHENOTYPE_ORDER, columns=[0, 1], fill_value=0)

    if observed_table.shape[0] < 2 or observed_table.shape[1] < 2:
        return np.nan, "Not applicable"

    _, chi_square_p, _, expected = stats.chi2_contingency(
        observed_table.values, correction=False
    )

    # Qui-quadrado se frequências esperadas forem adequadas
    if np.all(expected >= 5):

        return (chi_square_p, "Pearson chi-square")

    # Teste exato/Monte Carlo com margens fixas
    try:

        random_tables = stats.random_table.rvs(
            row=observed_table.sum(axis=1).to_numpy(),
            col=observed_table.sum(axis=0).to_numpy(),
            size=MONTE_CARLO_REPLICATES,
            random_state=RANDOM_SEED,
        )

        observed_statistic = pearson_statistic(observed_table.values)

        simulated_statistics = np.array(
            [pearson_statistic(table) for table in random_tables]
        )

        monte_carlo_p = (
            1 + np.sum(simulated_statistics >= observed_statistic - 1e-12)
        ) / (MONTE_CARLO_REPLICATES + 1)

        return (monte_carlo_p, ("Fisher–Freeman–Halton " "Monte Carlo exact test"))

    except Exception:

        # Fallback caso a versão do scipy não tenha random_table
        return (chi_square_p, "Pearson chi-square (fallback)")


# ------------------------------------------------------------
# 9. Variáveis da Tabela 1
# ------------------------------------------------------------

VARIABLES = [
    {
        "section": "Demographic and treatment characteristics",
        "label": "Age, years",
        "type": "continuous",
        "column": COL["age"],
        "decimals": 0,
    },
    {
        "section": "Demographic and treatment characteristics",
        "label": "Female sex",
        "type": "binary",
        "column": "female",
    },
    {
        "section": "Demographic and treatment characteristics",
        "label": "Time from ICU admission to methylene blue, h",
        "type": "continuous",
        "column": COL["hours_icu_to_mb"],
        "decimals": 1,
    },
    {
        "section": "Demographic and treatment characteristics",
        "label": "Methylene blue dose, mg/kg",
        "type": "continuous",
        "column": COL["mb_dose_mgkg"],
        "decimals": 2,
    },
    {
        "section": "Markers of illness severity and organ dysfunction",
        "label": "Lactate, mmol/L",
        "type": "continuous",
        "column": COL["lactate"],
        "decimals": 1,
    },
    {
        "section": "Markers of illness severity and organ dysfunction",
        "label": "Serum creatinine, mg/dL",
        "type": "continuous",
        "column": COL["creatinine"],
        "decimals": 1,
    },
    {
        "section": "Markers of illness severity and organ dysfunction",
        "label": "Total bilirubin, mg/dL",
        "type": "continuous",
        "column": COL["bilirubin"],
        "decimals": 1,
    },
    {
        "section": "Markers of illness severity and organ dysfunction",
        "label": "Platelet count, ×10³/µL",
        "type": "continuous",
        "column": COL["platelets"],
        "decimals": 0,
    },
    {
        "section": "Markers of illness severity and organ dysfunction",
        "label": "Invasive mechanical ventilation",
        "type": "binary",
        "column": COL["mechanical_ventilation"],
        "no_test": True,
    },
    {
        "section": "Markers of illness severity and organ dysfunction",
        "label": "Renal replacement therapy",
        "type": "binary",
        "column": COL["rrt"],
    },
    {
        "section": "Vasopressor support",
        "label": "Norepinephrine-equivalent dose, µg/kg/min",
        "type": "continuous",
        "column": COL["nee"],
        "decimals": 2,
    },
    {
        "section": "Vasopressor support",
        "label": "Epinephrine use",
        "type": "binary",
        "column": COL["epinephrine"],
    },
    {
        "section": "Baseline hemodynamics",
        "label": "Diastolic arterial pressure, mmHg",
        "type": "continuous",
        "column": COL["dbp"],
        "decimals": 1,
        "no_test": True,
    },
    {
        "section": "Baseline hemodynamics",
        "label": "Mean arterial pressure, mmHg",
        "type": "continuous",
        "column": COL["map"],
        "decimals": 1,
        "no_test": True,
    },
    {
        "section": "Baseline hemodynamics",
        "label": "Pulse pressure, mmHg",
        "type": "continuous",
        "column": COL["pp"],
        "decimals": 1,
        "no_test": True,
    },
    {
        "section": "Baseline hemodynamics",
        "label": "Heart rate, beats/min",
        "type": "continuous",
        "column": COL["heart_rate"],
        "decimals": 1,
    },
]

# ------------------------------------------------------------
# 10. Construir a tabela
# ------------------------------------------------------------

group_sizes = {
    phenotype: int(df["hemodynamic_phenotype"].eq(phenotype).sum())
    for phenotype in PHENOTYPE_ORDER
}

COLUMN_NAMES = [
    "Characteristic",
    f"Overall\n(n={len(df)})",
    ("Vasoplegia-compatible\n" f"(n={group_sizes[PHENOTYPE_ORDER[0]]})"),
    ("Mixed/low-pulsatility\n" f"(n={group_sizes[PHENOTYPE_ORDER[1]]})"),
    ("Preserved diastolic pressure\n" f"(n={group_sizes[PHENOTYPE_ORDER[2]]})"),
    "P value",
]

table_rows = []
test_rows = []

current_section = None

for variable in VARIABLES:

    if variable["section"] != current_section:

        table_rows.append(
            {
                COLUMN_NAMES[0]: variable["section"],
                COLUMN_NAMES[1]: "",
                COLUMN_NAMES[2]: "",
                COLUMN_NAMES[3]: "",
                COLUMN_NAMES[4]: "",
                COLUMN_NAMES[5]: "",
                "_row_type": "section",
            }
        )

        current_section = variable["section"]

    column = variable["column"]
    no_test = variable.get("no_test", False)

    if variable["type"] == "continuous":

        overall_value = format_continuous(df[column], variable["decimals"])

        group_values = [
            format_continuous(
                df.loc[df["hemodynamic_phenotype"].eq(phenotype), column],
                variable["decimals"],
            )
            for phenotype in PHENOTYPE_ORDER
        ]

        if no_test:

            p_value = np.nan
            test_name = "Not tested"

        else:

            p_value = kruskal_p(df, column)

            test_name = "Kruskal–Wallis"

    else:

        overall_value = format_binary(df[column])

        group_values = [
            format_binary(df.loc[df["hemodynamic_phenotype"].eq(phenotype), column])
            for phenotype in PHENOTYPE_ORDER
        ]

        if no_test:

            p_value = np.nan
            test_name = "Not tested"

        else:

            p_value, test_name = categorical_test(df, column)

    table_rows.append(
        {
            COLUMN_NAMES[0]: variable["label"],
            COLUMN_NAMES[1]: overall_value,
            COLUMN_NAMES[2]: group_values[0],
            COLUMN_NAMES[3]: group_values[1],
            COLUMN_NAMES[4]: group_values[2],
            COLUMN_NAMES[5]: format_p(p_value),
            "_row_type": "variable",
        }
    )

    test_rows.append(
        {
            "variable": variable["label"],
            "test": test_name,
            "p_value": p_value,
            "n_available": int(df[column].notna().sum()),
            "n_missing": int(df[column].isna().sum()),
        }
    )

table_internal = pd.DataFrame(table_rows)

table_display = table_internal.drop(columns="_row_type")

test_log = pd.DataFrame(test_rows)

# ------------------------------------------------------------
# 11. Mostrar a tabela
# ------------------------------------------------------------

pd.set_option("display.max_rows", 100)

pd.set_option("display.max_columns", 20)

pd.set_option("display.max_colwidth", 120)

pd.set_option("display.width", 320)

print("\n" + "=" * 100)
print("TABLE 1 — FINAL CORRIGIDA")
print("=" * 100)

display(table_display)

print("\n" + "=" * 100)
print("TESTES E DADOS AUSENTES")
print("=" * 100)

display(test_log)

# ------------------------------------------------------------
# 12. Funções para o DOCX
# ------------------------------------------------------------


def shade_cell(cell, fill):

    properties = cell._tc.get_or_add_tcPr()

    shading = OxmlElement("w:shd")

    shading.set(qn("w:fill"), fill)

    properties.append(shading)


def write_cell(cell, text, bold=False, alignment=WD_ALIGN_PARAGRAPH.LEFT, font_size=8):

    cell.text = ""

    paragraph = cell.paragraphs[0]
    paragraph.alignment = alignment

    run = paragraph.add_run(str(text))

    run.bold = bold
    run.font.name = "Arial"
    run.font.size = Pt(font_size)

    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


# ------------------------------------------------------------
# 13. DOCX da Tabela 1
# ------------------------------------------------------------

table_document = Document()

section = table_document.sections[0]

section.top_margin = Inches(0.5)
section.bottom_margin = Inches(0.5)
section.left_margin = Inches(0.4)
section.right_margin = Inches(0.4)

title = table_document.add_paragraph()

title.alignment = WD_ALIGN_PARAGRAPH.CENTER

title_run = title.add_run(
    "Table 1. Baseline characteristics according to " "hemodynamic phenotype"
)

title_run.bold = True
title_run.font.name = "Arial"
title_run.font.size = Pt(11)

table = table_document.add_table(rows=1, cols=len(table_display.columns))

table.alignment = WD_TABLE_ALIGNMENT.CENTER

table.style = "Table Grid"

for index, column_name in enumerate(table_display.columns):

    write_cell(
        table.rows[0].cells[index],
        column_name,
        bold=True,
        alignment=(
            WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
        ),
    )

    shade_cell(table.rows[0].cells[index], "D9EAF7")

for _, row in table_internal.iterrows():

    new_row = table.add_row()

    section_row = row["_row_type"] == "section"

    for index, column_name in enumerate(table_display.columns):

        write_cell(
            new_row.cells[index],
            row[column_name],
            bold=section_row,
            alignment=(
                WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            ),
        )

        if section_row:

            shade_cell(new_row.cells[index], "EDEDED")

table_document.add_paragraph()

footnote = table_document.add_paragraph()

footnote_run = footnote.add_run(
    "Data are presented as median [interquartile range] or "
    "n/N (%). P values represent global comparisons across "
    "the three phenotypes. Continuous variables were compared "
    "using the Kruskal–Wallis test. Categorical variables were "
    "compared using Pearson's chi-square test or an exact "
    "Monte Carlo test when expected cell frequencies were "
    "small. P values were not calculated for invasive "
    "mechanical ventilation because of near-universal exposure, "
    "or for diastolic arterial pressure, mean arterial pressure, "
    "and pulse pressure because these variables were directly "
    "or mathematically related to phenotype assignment. "
    "ICU, intensive care unit; NEE, norepinephrine-equivalent "
    "dose."
)

footnote_run.font.name = "Arial"
footnote_run.font.size = Pt(8)

# ------------------------------------------------------------
# 14. Methods draft
# ------------------------------------------------------------

methods_document = Document()

methods_document.add_heading("Methods draft", level=1)

methods_text = (
    "Baseline characteristics were compared across the three "
    "hemodynamic phenotypes. Continuous variables were summarized "
    "as medians and interquartile ranges and compared using the "
    "Kruskal–Wallis test. Categorical variables were summarized "
    "as counts and percentages. Pearson's chi-square test was "
    "used when expected cell frequencies were adequate; "
    "otherwise, an exact Monte Carlo test with fixed marginal "
    "totals was applied. Analyses were performed using available "
    "data without imputation. Inferential comparisons were not "
    "performed for invasive mechanical ventilation because of "
    "near-universal exposure, or for diastolic arterial pressure, "
    "mean arterial pressure, and pulse pressure because these "
    "variables were directly or mathematically related to "
    "phenotype assignment. All statistical tests were two-sided."
)

paragraph = methods_document.add_paragraph(methods_text)

for run in paragraph.runs:

    run.font.name = "Arial"
    run.font.size = Pt(10)

# ------------------------------------------------------------
# 15. Results draft
# ------------------------------------------------------------


def continuous_text(column, decimals=1):

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    return (
        f"{values.median():.{decimals}f} "
        f"[{values.quantile(0.25):.{decimals}f}–"
        f"{values.quantile(0.75):.{decimals}f}]"
    )


female_n = int((df["female"] == True).sum())

female_total = int(df["female"].notna().sum())

vm_n = int((df[COL["mechanical_ventilation"]] == True).sum())

vm_total = int(df[COL["mechanical_ventilation"]].notna().sum())

rrt_n = int((df[COL["rrt"]] == True).sum())

rrt_total = int(df[COL["rrt"]].notna().sum())

results_text = (
    f"The final classifiable cohort included {len(df)} patients, "
    f"including {group_sizes['Vasoplegia-compatible']} with a "
    f"vasoplegia-compatible phenotype, "
    f"{group_sizes['Mixed/low-pulsatility']} with a "
    f"mixed/low-pulsatility phenotype, and "
    f"{group_sizes['Preserved diastolic pressure']} with "
    f"preserved diastolic pressure. Median age was "
    f"{continuous_text(COL['age'], 0)} years, and "
    f"{female_n}/{female_total} "
    f"({100 * female_n / female_total:.1f}%) were female. "
    f"The median methylene blue dose was "
    f"{continuous_text(COL['mb_dose_mgkg'], 2)} mg/kg. "
    f"Median lactate was "
    f"{continuous_text(COL['lactate'], 1)} mmol/L, and the "
    f"median baseline norepinephrine-equivalent dose was "
    f"{continuous_text(COL['nee'], 2)} µg/kg/min. "
    f"Invasive mechanical ventilation was present in "
    f"{vm_n}/{vm_total} "
    f"({100 * vm_n / vm_total:.1f}%) patients, and renal "
    f"replacement therapy was active in "
    f"{rrt_n}/{rrt_total} "
    f"({100 * rrt_n / rrt_total:.1f}%). "
    f"No statistically significant differences were observed "
    f"across phenotypes in demographic characteristics, "
    f"laboratory markers of organ dysfunction, methylene blue "
    f"dose, baseline vasopressor burden, or heart rate. "
    f"Baseline characteristics are presented in Table 1."
)

results_document = Document()

results_document.add_heading("Results draft", level=1)

paragraph = results_document.add_paragraph(results_text)

for run in paragraph.runs:

    run.font.name = "Arial"
    run.font.size = Pt(10)

# ------------------------------------------------------------
# 16. Salvar outputs
# ------------------------------------------------------------

timestamp = run_stamp()

frozen_base_path = (
    COHORT_DIR / f"METHYLENE_BLUE_ANALYTIC_COHORT_FROZEN_FINAL_{timestamp}.csv.gz"
)

table_csv_path = TABLE_DIR / f"Table_1_FINAL_corrected_{timestamp}.csv"

table_docx_path = DOCUMENT_DIR / f"Table_1_FINAL_corrected_{timestamp}.docx"

methods_docx_path = DOCUMENT_DIR / f"Methods_draft_Table_1_FINAL_{timestamp}.docx"

results_docx_path = DOCUMENT_DIR / f"Results_draft_baseline_FINAL_{timestamp}.docx"

results_txt_path = DOCUMENT_DIR / f"Results_draft_baseline_FINAL_{timestamp}.txt"

test_log_path = QC_DIR / f"Table_1_FINAL_tests_and_missingness_{timestamp}.csv"

df.to_csv(frozen_base_path, index=False, compression="gzip")

table_display.to_csv(table_csv_path, index=False)

test_log.to_csv(test_log_path, index=False)

table_document.save(table_docx_path)

methods_document.save(methods_docx_path)

results_document.save(results_docx_path)

with open(results_txt_path, "w", encoding="utf-8") as file:

    file.write(results_text)

print("\n" + "=" * 100)
print("ARQUIVOS GERADOS")
print("=" * 100)

print("\nBase analítica congelada:")
print(frozen_base_path)

print("\nTabela 1 — CSV:")
print(table_csv_path)

print("\nTabela 1 — DOCX:")
print(table_docx_path)

print("\nMethods draft:")
print(methods_docx_path)

print("\nResults draft:")
print(results_docx_path)

print("\nLog estatístico:")
print(test_log_path)

print("\nCÓDIGO 12B CONCLUÍDO.")
