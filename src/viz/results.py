"""Графики результатов Стадии 3 (§5, §8). Импортируется ноутбуком 04_results.

Разделение обязанностей (§6): числа считает Стадия 3 и кладёт в
`eval_metrics.json`; здесь они только РИСУЮТСЯ. Поэтому EER, доверительные
интервалы, матрицы ошибок и разбивка по атакам читаются из готового JSON, а не
пересчитываются — иначе в работе могли бы разойтись число в таблице и число на
графике. Единственное исключение — сами кривые DET/ROC: они строятся из
`scores.parquet`, потому что кривая целиком в JSON не хранится. Это отрисовка
той же выгрузки, а не второй расчёт метрики: точка EER на кривую наносится из
JSON.

Все функции возвращают `matplotlib.Figure` — ноутбук решает, показать её или
сохранить через `viz.style.save_figure`.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_curve

from ..config import Paths

#: Деления осей DET в процентах — доменный стандарт (§8).
_DET_TICKS = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 40]) / 100.0


# =============================================================================
# Чтение артефактов
# =============================================================================
def load_scores(experiment: str, paths: Paths | None = None) -> pd.DataFrame:
    """experiments/<exp>/scores.parquet → DataFrame (§6.5)."""
    p = (paths or Paths()).experiments / experiment / "scores.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"нет {p} — сначала Стадия 3: python -m src.evaluate --config ..."
        )
    return pd.read_parquet(p)


def load_metrics(experiment: str, paths: Paths | None = None) -> dict:
    """experiments/<exp>/eval_metrics.json → dict (§6.5)."""
    p = (paths or Paths()).experiments / experiment / "eval_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"нет {p} — сначала Стадия 3.")
    return json.loads(p.read_text(encoding="utf-8"))


def eer_table(metrics: dict) -> pd.DataFrame:
    """Сводная таблица по датасетам: N, баланс классов, EER и его CI (в процентах).

    Готова к вставке в текст работы: одна строка на датасет, кросс-датасетное
    сравнение читается по столбцу EER (§8).
    """
    rows = []
    for ds, m in metrics["per_dataset"].items():
        ci = m.get("eer_ci")
        rows.append({
            "dataset_id": ds,
            "N": m["n"],
            "bonafide": m["n_bonafide"],
            "spoof": m["n_spoof"],
            "EER, %": None if m["eer"] is None else round(m["eer"] * 100, 2),
            "CI нижн., %": None if not ci else round(ci[0] * 100, 2),
            "CI верхн., %": None if not ci else round(ci[1] * 100, 2),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Кривые ошибок
# =============================================================================
def _curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, FNR) по выгрузке одного датасета. Спуф — положительный класс (§6.1)."""
    fpr, tpr, _ = roc_curve(df["label"].to_numpy().astype(int),
                            df["score"].to_numpy(), pos_label=1)
    return fpr, 1.0 - tpr


def plot_det(scores: pd.DataFrame, metrics: dict | None = None,
             title: str = "DET-кривые") -> plt.Figure:
    """DET-кривая по каждому датасету — стандартный график анти-спуфинга (§8).

    Оси — в нормальных отклонениях (probit): для гауссовых распределений скора
    кривая становится прямой, и различия в области малых ошибок, где всё
    интересное и происходит, видны, а не сжаты в угол. EER — точка пересечения
    с диагональю FPR = FNR (она нарисована пунктиром); если передан `metrics`,
    точка ставится по значению из JSON.
    """
    fig, ax = plt.subplots()
    lim = norm.ppf([_DET_TICKS.min(), _DET_TICKS.max()])

    for ds, g in scores.groupby("dataset_id", observed=True):
        if g["label"].nunique() < 2:
            continue
        fpr, fnr = _curve(g)
        m = (fpr > 0) & (fnr > 0)
        line, = ax.plot(norm.ppf(fpr[m]), norm.ppf(fnr[m]), label=str(ds))
        if metrics:
            eer = (metrics["per_dataset"].get(str(ds)) or {}).get("eer")
            if eer:
                ax.plot(norm.ppf(eer), norm.ppf(eer), "o", ms=5,
                        color=line.get_color())

    ax.plot(lim, lim, ls=":", lw=0.9, color="0.5", zorder=0)  # линия FPR = FNR
    for setter, getter in ((ax.set_xticks, ax.set_xticklabels),
                           (ax.set_yticks, ax.set_yticklabels)):
        setter(norm.ppf(_DET_TICKS))
        getter([f"{t * 100:g}" for t in _DET_TICKS])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Ложная тревога FPR, % (подлинную приняли за спуф)")
    ax.set_ylabel("Пропуск FNR, % (спуф приняли за подлинную)")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_roc(scores: pd.DataFrame, title: str = "ROC-кривые") -> plt.Figure:
    """ROC — привычная альтернатива DET; для отчёта основной график всё же DET (§8)."""
    fig, ax = plt.subplots()
    for ds, g in scores.groupby("dataset_id", observed=True):
        if g["label"].nunique() < 2:
            continue
        fpr, fnr = _curve(g)
        ax.plot(fpr, 1.0 - fnr, label=str(ds))
    ax.plot([0, 1], [0, 1], ls=":", lw=0.9, color="0.5", zorder=0)
    ax.set_xlabel("FPR (ложная тревога)")
    ax.set_ylabel("TPR (доля пойманного спуфа)")
    ax.set_title(title)
    ax.legend()
    return fig


# =============================================================================
# Матрица ошибок и разбивка по атакам
# =============================================================================
def plot_confusion(metrics: dict, dataset_id: str, at: str = "eer") -> plt.Figure:
    """Матрица ошибок одного датасета из eval_metrics.json.

    `at="eer"` — порог равного риска, подобранный на САМОМ тесте: показывает, как
    распределены ошибки в точке EER, но это не оценка обобщения (§6.5).
    `at="val"` — порог, выбранный на валидации до теста: честная рабочая точка.
    """
    key = {"eer": "at_eer_threshold", "val": "at_val_threshold"}[at]
    block = metrics["per_dataset"][dataset_id].get(key)
    if block is None:
        raise KeyError(f"в метриках {dataset_id} нет блока '{key}'")
    c = block["confusion"]
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat / mat.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            share = mat[i, j] / max(mat[i].sum(), 1)
            ax.text(j, i, f"{mat[i, j]}\n{share:.1%}", ha="center", va="center",
                    color="white" if share > 0.5 else "black")
    ax.set_xticks([0, 1], ["подлинная", "спуф"])
    ax.set_yticks([0, 1], ["подлинная", "спуф"])
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    suffix = "порог EER (подобран на тесте)" if at == "eer" else "порог с валидации"
    ax.set_title(f"{dataset_id}: {suffix}")
    ax.grid(False)
    return fig


def plot_eer_by_attack(metrics: dict, dataset_id: str) -> plt.Figure:
    """EER по каждой атаке отдельно — то, чего не видно в агрегированном числе (§8).

    Пунктиром — общий EER датасета: столбцы выше него показывают атаки, на которых
    метод проседает. Именно эта картинка отвечает на вопрос комиссии «а где метод
    работает плохо».
    """
    m = metrics["per_dataset"][dataset_id]
    by = {k: v for k, v in m.get("by_attack", {}).items() if v["eer"] is not None}
    if not by:
        raise ValueError(
            f"у {dataset_id} нет разбивки по атакам (attack_type не размечен, §6.1)"
        )
    names = list(by)
    vals = [by[k]["eer"] * 100 for k in names]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * len(names) + 1.5), 4.0))
    ax.bar(names, vals, width=0.7)
    if m["eer"] is not None:
        ax.axhline(m["eer"] * 100, ls="--", lw=1.0, color="0.35",
                   label=f"EER датасета = {m['eer'] * 100:.2f}%")
        ax.legend()
    ax.set_ylabel("EER, %")
    ax.set_xlabel("Атака")
    ax.set_title(f"{dataset_id}: EER по атакам")
    ax.tick_params(axis="x", rotation=90 if len(names) > 12 else 0)
    return fig