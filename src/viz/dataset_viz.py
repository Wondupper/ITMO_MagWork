"""Графики состава данных: баланс классов, сплиты, атаки, длительности (§5).

Читают манифесты Стадии 0 (§6.1). Нужны в работе по двум причинам: во-первых,
описать данные честно (сколько чего, насколько несбалансировано), во-вторых —
увидеть то, что потом объясняет метрики: перекос классов, разное число записей на
атаку, разброс длительностей относительно `t_fixed`.

Про длительности отдельно. Колонки `duration` адаптеры не заполняют — она
опциональная и по факту NULL у всех четырёх текущих датасетов (§6.1, Приложение А).
Поэтому длительность берётся из числа кадров кэша признаков: `T · hop / sr`. Это
не обход контракта, а более полезная величина — модель работает именно с кадрами,
и сравнивать с `t_fixed` надо их. Читаются только заголовки `.npy` (через memmap),
данные с диска не поднимаются.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Paths
from ..data.features import FeatureExtractor

_LABEL_NAME = {0: "подлинные", 1: "спуф"}


# =============================================================================
# Чтение
# =============================================================================
def load_manifests(paths: Paths, dataset_ids: list[str]) -> pd.DataFrame:
    """Склеить манифесты нескольких датасетов (§3: только concat, без ветвления)."""
    frames = []
    for ds in dataset_ids:
        p = paths.manifest_path(ds)
        if not p.exists():
            raise FileNotFoundError(
                f"нет манифеста {p} — сначала Стадия 0: python -m src.data.manifest {ds}"
            )
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)


def composition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка «датасет × сплит»: сколько записей каждого класса и их доля.

    Готова к вставке в раздел с описанием данных. Столбец «спуф, %» — та самая
    несбалансированность, из-за которой на Стадии 2 включается `pos_weight` (§10).
    """
    g = (df.groupby(["dataset_id", "split", "label"], observed=True)
           .size().unstack("label", fill_value=0))
    g = g.rename(columns=_LABEL_NAME)
    for c in _LABEL_NAME.values():
        if c not in g.columns:
            g[c] = 0
    g["всего"] = g["подлинные"] + g["спуф"]
    g["спуф, %"] = (100 * g["спуф"] / g["всего"].where(g["всего"] > 0)).round(1)
    return g.reset_index()


# =============================================================================
# Состав
# =============================================================================
def plot_class_balance(df: pd.DataFrame, title: str = "Баланс классов") -> plt.Figure:
    """Число подлинных и спуф-записей по каждому датасету (все сплиты вместе)."""
    g = (df.groupby(["dataset_id", "label"], observed=True).size()
           .unstack("label", fill_value=0).rename(columns=_LABEL_NAME))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(g) + 2), 4.0))
    x = np.arange(len(g))
    w = 0.38
    for i, col in enumerate(_LABEL_NAME.values()):
        vals = g[col].to_numpy() if col in g else np.zeros(len(g))
        ax.bar(x + (i - 0.5) * w, vals, width=w, label=col)
    ax.set_xticks(x, g.index.astype(str), rotation=15 if len(g) > 3 else 0)
    ax.set_ylabel("Записей")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_split_composition(df: pd.DataFrame, title: str = "Состав сплитов") -> plt.Figure:
    """Разбивка «сплит × класс» по датасетам — стопкой, в логарифме по количеству.

    Лог по оси нужен, потому что размеры отличаются на порядки (train 2019 LA против
    eval-наборов 2021): в линейном масштабе мелкие сплиты сплющиваются в ноль.
    """
    g = (df.groupby(["dataset_id", "split", "label"], observed=True).size()
           .unstack("label", fill_value=0).rename(columns=_LABEL_NAME))
    labels = [f"{ds}\n{sp}" for ds, sp in g.index]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.15 * len(g) + 1.5), 4.2))
    bottom = np.zeros(len(g))
    for col in _LABEL_NAME.values():
        vals = g[col].to_numpy() if col in g else np.zeros(len(g))
        ax.bar(labels, vals, bottom=bottom, label=col)
        bottom += vals
    ax.set_yscale("log")
    ax.set_ylabel("Записей (лог. шкала)")
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    return fig


def plot_attack_counts(df: pd.DataFrame, dataset_id: str) -> plt.Figure:
    """Сколько записей на каждую атаку. Читается вместе с EER по атакам (§8).

    Атака с десятком записей даёт очень шумный EER — сравнивать её столбец с
    атакой на тысячах записей нельзя, и эта картинка показывает, почему.
    """
    sub = df[(df["dataset_id"] == dataset_id) & df["attack_type"].notna()]
    if sub.empty:
        raise ValueError(f"[{dataset_id}] attack_type не размечен (§6.1)")
    counts = sub["attack_type"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * len(counts) + 1.5), 4.0))
    ax.bar(counts.index.astype(str), counts.to_numpy(), width=0.7)
    ax.set_ylabel("Записей")
    ax.set_xlabel("Атака")
    ax.set_title(f"{dataset_id}: число записей по атакам")
    ax.tick_params(axis="x", rotation=90 if len(counts) > 12 else 0)
    return fig


# =============================================================================
# Длительности
# =============================================================================
def frame_counts(paths: Paths, extractor: FeatureExtractor, dataset_id: str, *,
                 sample: int | None = 3000, seed: int = 1337) -> np.ndarray:
    """Число кадров T у записей датасета — из заголовков `.npy` кэша признаков.

    `sample` ограничивает число просматриваемых файлов (детерминированно): для
    гистограммы формы распределения трёх тысяч записей достаточно, а полный обход
    сотен тысяч файлов занял бы заметное время без пользы. `sample=None` — все.
    """
    df = pd.read_parquet(paths.manifest_path(dataset_id), columns=["dataset_id", "utt_id"])
    df = df.sort_values(["dataset_id", "utt_id"], kind="stable").reset_index(drop=True)
    if sample is not None and sample < len(df):
        idx = np.sort(np.random.default_rng(seed).permutation(len(df))[:sample])
        df = df.iloc[idx]

    out = []
    for r in df.itertuples(index=False):
        p = extractor.output_path(paths, str(r.dataset_id), str(r.utt_id))
        if p.exists():
            out.append(np.load(p, mmap_mode="r").shape[0])  # заголовок, не данные
    if not out:
        raise FileNotFoundError(
            f"[{dataset_id}] в кэше {extractor.cache_dir(paths).name} нет ни одного "
            f"файла — сначала Стадия 1."
        )
    return np.asarray(out)


def plot_durations(counts: dict[str, np.ndarray], extractor: FeatureExtractor, *,
                   t_fixed: int | None = None, bins: int = 60) -> plt.Figure:
    """Распределение длительностей записей по датасетам, с отметкой `t_fixed`.

    Ключевая картинка для интерпретации кросс-датасетных метрик (§8). Линия
    `t_fixed` делит ось на две области с РАЗНЫМ обращением с записью (§7):
    правее — запись обрезается и часть сигнала модель не видит; левее — запись
    повторяется до нужной длины, то есть модель видит один и тот же фрагмент
    несколько раз. Если распределение датасета целиком лежит по одну сторону
    линии (как у `for-2seconds`), это систематический артефакт домена, а не
    случайная особенность отдельных файлов, и его следует оговорить.
    """
    hop, sr = getattr(extractor, "hop_length", 160), extractor.target_sr
    fig, ax = plt.subplots()
    for ds, c in counts.items():
        ax.hist(np.asarray(c) * hop / sr, bins=bins, alpha=0.55,
                label=f"{ds} (n={len(c)})", density=True)
    if t_fixed:
        edge = t_fixed * hop / sr
        ax.axvline(edge, color="red", ls="--", lw=1.2,
                   label=f"t_fixed = {t_fixed} кадров ≈ {edge:.2f} с")
        ax.text(edge, ax.get_ylim()[1] * 0.97, " кроп →", color="red", fontsize=8,
                ha="left", va="top")
        ax.text(edge, ax.get_ylim()[1] * 0.97, "← повтор ", color="red", fontsize=8,
                ha="right", va="top")
    ax.set_xlabel("Длительность записи, с")
    ax.set_ylabel("Плотность")
    ax.set_title(f"Длительности записей (по кадрам кэша {extractor.name})")
    ax.legend()
    return fig