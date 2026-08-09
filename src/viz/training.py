"""Кривые обучения из `experiments/<exp>/metrics.jsonl` (§5, §6.5).

Стадия 2 пишет по строке на эпоху: `epoch`, `train_loss`, `val_eer`,
`val_threshold`, `lr`. Здесь они только читаются и рисуются — ничего не
пересчитывается (§6).

Зачем эта картинка в работе: она показывает, что обучение вообще сходилось и что
выбранная эпоха (`best.pt`) не взята с потолка. Заодно на ней видны две типовые
неприятности — расхождение loss и val-EER (переобучение) и «лучшая» эпоха,
выигравшая случайно на шумной валидации (§6.5).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from ..config import Paths


def load_history(experiment: str, paths: Paths | None = None) -> pd.DataFrame:
    """metrics.jsonl → DataFrame по эпохам (отсортирован, дубли сняты).

    Дедуп по эпохе — та же логика, что при возобновлении Стадии 2 (§7): если файл
    дописывался после рестарта, побеждает последняя запись эпохи.
    """
    p = (paths or Paths()).experiments / experiment / "metrics.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"нет {p} — сначала Стадия 2: python -m src.train --config ..."
        )
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"{p} пуст — обучение не записало ни одной эпохи.")
    df = pd.DataFrame(rows).drop_duplicates(subset="epoch", keep="last")
    return df.sort_values("epoch").reset_index(drop=True)


def best_epoch(history: pd.DataFrame) -> int | None:
    """Эпоха с минимальным val-EER — та, чьи веса лежат в `best.pt` (§6.5)."""
    valid = history.dropna(subset=["val_eer"])
    if valid.empty:
        return None
    return int(valid.loc[valid["val_eer"].idxmin(), "epoch"])


def plot_training(history: pd.DataFrame, title: str = "Кривые обучения") -> plt.Figure:
    """Loss обучения и val-EER на одних осях времени (две шкалы Y).

    Совмещены намеренно: смотреть на них по отдельности бесполезно — интересна
    именно их рассинхронизация. Loss, который продолжает падать при растущем
    val-EER, — это переобучение, и на общей картинке оно видно сразу.
    """
    fig, ax = plt.subplots()
    ax.plot(history["epoch"], history["train_loss"], marker="o", ms=3,
            label="train loss")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Train loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # эпохи целые

    ax2 = ax.twinx()
    ax2.grid(False)
    valid = history.dropna(subset=["val_eer"])
    ax2.plot(valid["epoch"], valid["val_eer"] * 100, marker="s", ms=3,
             color="tab:orange", label="val EER")
    ax2.set_ylabel("Val EER, %")

    be = best_epoch(history)
    if be is not None:
        ax.axvline(be, color="0.5", ls="--", lw=1.0)
        # Подпись уводим внутрь поля, иначе на последней эпохе она уезжает за край.
        right = be > history["epoch"].to_numpy().mean()
        ax.annotate(f"best.pt: эпоха {be}",
                    xy=(be, ax.get_ylim()[1]), fontsize=8, color="0.35",
                    ha="right" if right else "left", va="top",
                    xytext=(-4 if right else 4, -2), textcoords="offset points")

    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper center")
    ax.set_title(title)
    return fig


def plot_val_threshold(history: pd.DataFrame,
                       title: str = "Порог равного риска по эпохам") -> plt.Figure:
    """Как гуляет val-порог от эпохи к эпохе.

    Полезно перед тем, как брать этот порог рабочей точкой на Стадии 3: если он
    скачет от эпохи к эпохе, значит он определён плохо (типично для маленькой
    валидации, §6.4), и матрицу ошибок на нём надо читать осторожно.
    """
    valid = history.dropna(subset=["val_threshold"])
    if valid.empty:
        raise ValueError("в metrics.jsonl нет ни одного val_threshold (val-EER = NaN?)")
    fig, ax = plt.subplots()
    ax.plot(valid["epoch"], valid["val_threshold"], marker="o", ms=3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Порог (логит)")
    ax.set_title(title)
    return fig