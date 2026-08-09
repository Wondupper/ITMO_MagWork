"""Единый стиль фигур и их сохранение (§5, §7).

Одно место, задающее вид всех графиков работы: размеры шрифтов под печать, сетка,
поля. Ноутбуки его только вызывают — так фигуры из разных ноутбуков выглядят
как один документ, а не как четыре разных.

Правило форматов (§7): линейные графики (loss, DET, ROC) — в ВЕКТОР (PDF/SVG),
они масштабируются без потери качества; тепловые карты и спектрограммы — растр
с DPI ≥ 300. `save_figure` по умолчанию пишет PDF; для растровых сюжетов
передайте formats=("png",).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..config import Paths

#: Базовые rcParams. Подобраны под вставку фигуры шириной ~0.8 страницы А4.
RC_PARAMS: dict = {
    "figure.figsize": (6.0, 4.2),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # Шрифты в PDF как настоящий текст, а не кривые: файл легче и текст ищется.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Применить стиль проекта к текущей сессии matplotlib."""
    mpl.rcParams.update(RC_PARAMS)


def save_figure(fig: plt.Figure, name: str, *, paths: Paths | None = None,
                formats: tuple[str, ...] = ("pdf",), dpi: int | None = None) -> list[Path]:
    """Сохранить фигуру в reports/figures/ во всех указанных форматах.

    Возвращает список записанных путей. Папка создаётся автоматически.
    """
    out_dir = (paths or Paths()).figures
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi or mpl.rcParams["savefig.dpi"])
        written.append(p)
    return written