"""Графики уровня одной записи: сигнал, спектр, карта признаков (§5).

Отвечает на вопрос «что вообще видит модель». Ноутбук 01 показывает одну и ту же
запись в трёх видах — волна, спектрограмма, кэшированные признаки `[T, C]`, — и
рядом ту же тройку для записи другого класса. Это не метрика, а иллюстрация:
она нужна, чтобы в работе можно было объяснить, из чего берётся решение, и чтобы
самому увидеть, если с признаками что-то не так.

Важное про источник данных. Аудио читается ТЕМ ЖЕ декодером, что и на Стадии 1
(`data.features.load_audio`): тот же фолбэк на ffmpeg, тот же ресемпл к
`extractor.target_sr`. Свой `librosa.load` в ноутбуке дал бы слегка другие числа
и незаметно рассогласовал картинку с кэшем. Признаки не пересчитываются вовсе —
читается готовый `.npy` (§6).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Paths
from ..data.features import FeatureExtractor, load_audio


# =============================================================================
# Выбор примера и чтение артефактов
# =============================================================================
def pick_examples(paths: Paths, dataset_id: str, *, n: int = 1,
                  label: int | None = None, split: str | None = None,
                  attack_type: str | None = None, seed: int = 1337) -> pd.DataFrame:
    """Детерминированно выбрать n строк манифеста под заданные условия.

    Тот же инвариант, что у подвыборок (§6.4): стабильная сортировка по ключу,
    затем выборка с фиксированным seed — иначе «пример из работы» не воспроизвести.
    """
    df = pd.read_parquet(paths.manifest_path(dataset_id))
    if label is not None:
        df = df[df["label"] == label]
    if split is not None:
        df = df[df["split"] == split]
    if attack_type is not None:
        df = df[df["attack_type"] == attack_type]
    if df.empty:
        raise ValueError(
            f"[{dataset_id}] нет записей под условия "
            f"label={label}, split={split}, attack_type={attack_type}"
        )
    df = df.sort_values(["dataset_id", "utt_id"], kind="stable").reset_index(drop=True)
    idx = np.random.default_rng(seed).permutation(len(df))[:n]
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def load_waveform(paths: Paths, row, extractor: FeatureExtractor) -> tuple[np.ndarray, int]:
    """Сигнал записи из `data/raw/` — декодером Стадии 1, на её же частоте."""
    return load_audio(paths.raw_dataset(str(row.dataset_id)) / str(row.path),
                      extractor.target_sr)


def load_feature(paths: Paths, row, extractor: FeatureExtractor) -> np.ndarray:
    """Кэшированные признаки `[T, C]` этой записи (§6.5). НЕ пересчитываются."""
    p = extractor.output_path(paths, str(row.dataset_id), str(row.utt_id))
    if not p.exists():
        raise FileNotFoundError(
            f"нет признаков {p} — сначала Стадия 1: python -m src.data.features "
            f"{row.dataset_id} -f {extractor.name}"
        )
    return np.load(p)


def describe(row) -> str:
    """Короткая подпись примера: класс, атака, датасет."""
    kind = "спуф" if int(row.label) == 1 else "подлинная"
    attack = "" if pd.isna(row.attack_type) else f", {row.attack_type}"
    return f"{row.dataset_id}/{row.utt_id} — {kind}{attack}"


# =============================================================================
# Отдельные панели
# =============================================================================
def plot_waveform(y: np.ndarray, sr: int, *, ax=None, title: str | None = None):
    """Осциллограмма: амплитуда во времени. Отсюда видно длительность и тишину."""
    ax = ax or plt.subplots()[1]
    t = np.arange(len(y)) / sr
    ax.plot(t, y, lw=0.5)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Амплитуда")
    if title:
        ax.set_title(title)
    return ax


def plot_spectrogram(y: np.ndarray, sr: int, *, ax=None, title: str | None = None,
                     n_fft: int = 512, hop_length: int = 160, top_db: float = 80.0):
    """Спектрограмма в дБ: как энергия распределена по частотам во времени.

    Строится линейный STFT, а не мел: для анти-спуфинга интересен в том числе
    верхний диапазон, который мел-шкала сжимает (§8 — LFCC не случайно линейный).
    Растровая картинка → сохранять в PNG с DPI ≥ 300 (§7).
    """
    import librosa  # тяжёлый импорт — только когда реально рисуем спектр

    ax = ax or plt.subplots()[1]
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    db = librosa.amplitude_to_db(S, ref=np.max, top_db=top_db)
    im = ax.imshow(db, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, len(y) / sr, 0, sr / 2000])
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, кГц")
    ax.grid(False)
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="дБ", pad=0.02)
    return ax


def plot_feature_map(arr: np.ndarray, *, ax=None, title: str | None = None,
                     hop_length: int = 160, sr: int = 16000, channel_label: str = "Канал C"):
    """Карта признаков `[T, C]` из кэша — ровно то, что попадает в модель (§6.6).

    Ось времени переводится в секунды через hop экстрактора, чтобы картинка
    сопоставлялась с осциллограммой над ней.
    """
    ax = ax or plt.subplots()[1]
    dur = arr.shape[0] * hop_length / sr
    im = ax.imshow(arr.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, dur, 0, arr.shape[1]])
    ax.set_xlabel("Время, с")
    ax.set_ylabel(channel_label)
    ax.grid(False)
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, pad=0.02)
    return ax


# =============================================================================
# Сборка
# =============================================================================
def plot_example(paths: Paths, row, extractor: FeatureExtractor,
                 t_fixed: int | None = None) -> plt.Figure:
    """Одна запись в трёх видах: волна → спектрограмма → признаки.

    Если передан `t_fixed`, на панелях отмечается граница окна, которое реально
    увидит модель (§7): всё правее отбрасывается кропом, а короткая запись
    повторяется до этой границы. Это самая наглядная иллюстрация того, что
    фиксированная длина — не бесплатная операция.
    """
    y, sr = load_waveform(paths, row, extractor)
    arr = load_feature(paths, row, extractor)
    hop = getattr(extractor, "hop_length", 160)

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.0))
    plot_waveform(y, sr, ax=axes[0], title=describe(row))
    plot_spectrogram(y, sr, ax=axes[1], hop_length=hop)
    plot_feature_map(arr, ax=axes[2], hop_length=hop, sr=extractor.target_sr,
                     title=f"{extractor.name}: [T={arr.shape[0]}, C={arr.shape[1]}]")

    if t_fixed:
        edge = t_fixed * hop / extractor.target_sr
        for ax in axes:
            if edge < ax.get_xlim()[1]:
                ax.axvline(edge, color="red", ls="--", lw=1.0)
        axes[0].text(0.99, 0.95, f"t_fixed = {t_fixed} кадров ≈ {edge:.2f} с",
                     transform=axes[0].transAxes, ha="right", va="top",
                     fontsize=8, color="red")
    fig.tight_layout()
    return fig


def plot_class_pair(paths: Paths, dataset_id: str, extractor: FeatureExtractor,
                    *, split: str = "train", seed: int = 1337) -> plt.Figure:
    """Признаки подлинной и спуф-записи рядом, в общей цветовой шкале.

    Общая шкала обязательна: без неё две карты нормируются по-своему и «разница»
    на картинке будет артефактом раскраски, а не свойством данных.
    """
    rows = [pick_examples(paths, dataset_id, label=lab, split=split, seed=seed).iloc[0]
            for lab in (0, 1)]
    arrs = [load_feature(paths, r, extractor) for r in rows]
    vmin = min(a.min() for a in arrs)
    vmax = max(a.max() for a in arrs)
    hop = getattr(extractor, "hop_length", 160)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, r, a in zip(axes, rows, arrs):
        im = ax.imshow(a.T, origin="lower", aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax,
                       extent=[0, a.shape[0] * hop / extractor.target_sr, 0, a.shape[1]])
        ax.set_title(describe(r), fontsize=9)
        ax.set_xlabel("Время, с")
        ax.grid(False)
    axes[0].set_ylabel(f"{extractor.name}, канал C")
    fig.colorbar(im, ax=axes, pad=0.02)
    return fig