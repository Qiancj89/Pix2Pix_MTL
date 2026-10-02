"""Reproduce Figure R1's seven calibration curves from frozen predictions.

This is a descriptive visualization of the post hoc patient-level 70:30
development-split sensitivity analysis. No model is trained or re-evaluated.
"""

from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.text import Text
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REVISION = HERE.parents[1]
PREDICTIONS_DIR = Path(os.environ.get("PREDICTIONS_DIR", HERE / "predictions")).expanduser().resolve()
CONDITIONS = [
    ("bmode", "B-mode", "#2E74B5"),
    ("synthetic", "Synthetic CEUS", "#ED7D31"),
    ("bmode_synthetic", "B-mode + synthetic CEUS", "#70AD47"),
    ("weighted_bmode_synthetic", "Weighted fused", "#C00000"),
    ("focal_bmode_synthetic", "Focal fused", "#8064A2"),
    ("real_ceus", "Real CEUS", "#A5643A"),
    ("bmode_real_ceus", "B-mode + real CEUS", "#D86BAA"),
]
FONT_PATH = Path(os.environ.get("TIMES_BOLD_FONT", "C:/Windows/Fonts/timesbd.ttf"))
if FONT_PATH.is_file():
    fontManager.addfont(str(FONT_PATH))
FONT = FontProperties(fname=str(FONT_PATH), weight="bold") if FONT_PATH.is_file() else FontProperties(family="Times New Roman", weight="bold")


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.1, 5.4))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.79, bottom=0.12)
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1.0, zorder=1)
    point_rows = []
    for name, label, color in CONDITIONS:
        frame = pd.read_csv(PREDICTIONS_DIR / f"{name}.csv")
        frame = frame.loc[frame["split"].eq("test_internal_temporal")]
        grouped = frame.groupby("patient_id", sort=True).agg(
            reference=("Malignant_true", "max"),
            risk=("Malignant_prob", "max"),
        )
        if len(grouped) != 62 or int(grouped["reference"].sum()) != 9:
            raise RuntimeError(f"Unexpected test cohort for {name}")
        reference = grouped["reference"].to_numpy()
        risk = grouped["risk"].to_numpy()
        groups = np.array_split(np.argsort(risk), 5)
        predicted = np.array([risk[index].mean() for index in groups])
        observed = np.array([reference[index].mean() for index in groups])
        for bin_number, (index, x, y) in enumerate(zip(groups, predicted, observed), 1):
            point_rows.append(
                {"condition": name, "bin": bin_number, "patients": len(index),
                 "mean_predicted_probability": x, "observed_positive_fraction": y}
            )
        ax.plot(predicted, observed, color=color, linewidth=1.7, marker="o",
                markersize=3.2, label=label, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Predicted probability", fontsize=14)
    ax.set_ylabel("Observed proportion", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, color="#D8D8D8", linewidth=0.65)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2,
               frameon=False, fontsize=12, columnspacing=0.7, handlelength=1.5)
    for item in fig.findobj(match=Text):
        item.set_fontproperties(FONT)
        item.set_color("black")
    plt.rcParams["pdf.fonttype"] = 42
    fig.savefig(HERE / "Figs" / "Figure5.pdf", facecolor="white")
    fig.savefig(HERE / "Figs" / "Figure5.png", dpi=300,
                facecolor="white")
    pd.DataFrame(point_rows).to_csv(HERE / "Figs" / "Figure5_Calibration_Points.csv",
                                    index=False, float_format="%.12f")
    plt.close(fig)


if __name__ == "__main__":
    main()
