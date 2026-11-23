"""Source: original notebook cell 69. Run through run.py."""

from common import phase_dir

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ------------------------------------------------------------
# Caminho
# ------------------------------------------------------------

FIGURE_DIR = phase_dir("mimic_iv") / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Figura
# ------------------------------------------------------------

plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(7.2, 11))
ax = plt.axes([0, 0, 1, 1])

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ------------------------------------------------------------
# Funções
# ------------------------------------------------------------


def box(
    x,
    y,
    w,
    h,
    text,
    facecolor="white",
    edgecolor="#4d4d4d",
    textcolor="black",
    fontsize=11,
    lw=1.3,
):

    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )

    ax.add_patch(rect)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        wrap=True,
    )


def arrow(y1, y2):

    ax.add_patch(
        FancyArrowPatch(
            (0.5, y1),
            (0.5, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color="#707070",
        )
    )


# ------------------------------------------------------------
# Coordenadas
# ------------------------------------------------------------

w = 0.62
h = 0.065
x = 0.19

ys = [0.90, 0.79, 0.67, 0.54, 0.42, 0.29, 0.15]

# ------------------------------------------------------------
# Caixas
# ------------------------------------------------------------

blue = "#2F5597"

box(
    x,
    ys[0],
    w,
    h,
    "HiRID Critical Care Database",
    facecolor=blue,
    edgecolor=blue,
    textcolor="white",
    fontsize=12,
)

box(x, ys[1], w, h, "Derivation cohort")

box(
    x,
    ys[2],
    w,
    h + 0.01,
    "Identification of macrohemodynamic phenotypes\n\n(DBP + pulse pressure)",
)

box(
    x,
    ys[3],
    w,
    h,
    "MIMIC-IV",
    facecolor=blue,
    edgecolor=blue,
    textcolor="white",
    fontsize=12,
)

box(
    x,
    ys[4],
    w,
    h + 0.005,
    "External validation cohort\n\nSeptic shock patients treated with methylene blue",
)

box(
    x,
    ys[5],
    w,
    h + 0.005,
    "Classification according to the\nderived physiological phenotypes",
)

box(
    x,
    ys[6],
    w,
    h + 0.015,
    "Early physiological response\n\nNorepinephrine-equivalent trajectory (GEE)",
)

# ------------------------------------------------------------
# Setas
# ------------------------------------------------------------

arrow(0.90, 0.855)
arrow(0.79, 0.735)
arrow(0.67, 0.605)
arrow(0.54, 0.485)
arrow(0.42, 0.355)
arrow(0.29, 0.225)

# ------------------------------------------------------------
# Exportar
# ------------------------------------------------------------

pdf = FIGURE_DIR / "Figure_1_Study_Design.pdf"
svg = FIGURE_DIR / "Figure_1_Study_Design.svg"
png = FIGURE_DIR / "Figure_1_Study_Design.png"

fig.savefig(pdf, bbox_inches="tight")

fig.savefig(svg, bbox_inches="tight")

fig.savefig(png, dpi=600, bbox_inches="tight")

plt.show()

print(pdf)
print(svg)
print(png)
