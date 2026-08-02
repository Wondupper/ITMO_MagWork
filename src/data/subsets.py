"""Детерминированные стратифицированные подвыборки: манифест → манифест (§6.4).

Подвыборка — это НЕ копия аудио/признаков, а отфильтрованный манифест: он указывает
на те же файлы в `raw/` и те же записи в `features/`. Бесплатна по диску (§6.4).
Роль в проекте — быстрая итерация и smoke-прогоны Стадий 2–3 на маленьком, но
репрезентативном срезе; итоговые метрики берутся на полном тесте (§6.4, оговорка).

Единственный неочевидный инвариант, ради которого всё написано именно так:

    сначала СТАБИЛЬНАЯ сортировка по (dataset_id, utt_id), и только ПОТОМ выборка
    с зафиксированным seed.

Без сортировки тот же seed на иначе упорядоченном DataFrame даёт ДРУГИЕ строки —
и эксперименты становятся несравнимыми (§6.4). Пара (dataset_id, utt_id) —
глобальный ключ (§3), поэтому сортировка по ней однозначна для любого манифеста,
в том числе склеенного из нескольких датасетов.

Публичный API:
    make_subset(df, n, seed, stratify) -> DataFrame   — сама подвыборка.
    subset_filename(...) / save_subset(...)           — самодокументирующий артефакт
                                                         `manifests/subsets/*.parquet`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import Paths
from ..utils.io import atomic_write_parquet

#: Ключ, по которому манифест сортируется перед любой выборкой (§3, §6.4).
_KEY = ["dataset_id", "utt_id"]

#: Дефолтная стратификация (§6.4): классовый баланс внутри каждого сплита.
#: При склейке нескольких датасетов вызывающий код добавляет "dataset_id".
DEFAULT_STRATIFY: tuple[str, ...] = ("split", "label")


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Стабильная сортировка по глобальному ключу — основа детерминизма (§6.4)."""
    return df.sort_values(_KEY, kind="stable").reset_index(drop=True)


def _allocate(n: int, sizes: np.ndarray) -> np.ndarray:
    """Разложить бюджет n по стратам пропорционально их размеру.

    Метод наибольших остатков: floor от пропорции + раздача остатка стратам с
    наибольшей дробной частью — так сумма квот в точности равна n (пока n ≤ N).
    Итоговая квота ещё кэпируется размером страты вызывающим кодом.
    """
    total = int(sizes.sum())
    if total == 0:
        return np.zeros_like(sizes)
    exact = n * sizes / total
    floor = np.floor(exact).astype(int)
    remainder = n - int(floor.sum())
    if remainder > 0:
        # Раздаём +1 стратам с наибольшей дробной частью; при равенстве —
        # по возрастанию индекса (argsort стабилен) → детерминированно.
        order = np.argsort(-(exact - floor), kind="stable")
        floor[order[:remainder]] += 1
    return floor


def make_subset(
    df: pd.DataFrame,
    n: int | None,
    *,
    seed: int,
    stratify: Sequence[str] = DEFAULT_STRATIFY,
) -> pd.DataFrame:
    """Детерминированная стратифицированная подвыборка размера ≈ n (§6.4).

    Параметры
    ---------
    df : манифест (одного датасета или уже склеенный из нескольких).
    n  : целевой размер. None или n ≥ len(df) → вернуть ВЕСЬ df (стабильно
         отсортированным) — удобно для «полного» прогона тем же кодом.
    seed : фиксированный seed выборки.
    stratify : колонки стратификации. Отсутствующие в df — молча игнорируются
         (напр. одиночный датасет без нужды в "dataset_id"). Колонки берутся
         только те, что реально есть → функция не ветвится по датасету (§3).

    Возвращает отфильтрованный df тех же колонок и типов, стабильно
    отсортированный по ключу. Аудио/признаки не копируются.

    Замечание (честно): при стратификации по колонке с NULL (напр. attack_type,
    §6.1) NULL образует отдельную страту. Дефолт (split, label) от этого свободен —
    это non-null обязательные колонки. Для очень малого n редкая страта может
    оказаться недопредставленной — подвыборка предназначена для отладки, не для
    итоговых метрик (§6.4).
    """
    df = _stable_sort(df)
    if n is None or n >= len(df):
        return df

    strat_cols = [c for c in stratify if c in df.columns]
    rng = np.random.default_rng(seed)

    if not strat_cols:
        idx = rng.permutation(len(df))[:n]
        return _stable_sort(df.iloc[np.sort(idx)])

    # sort=True → страты обходятся в детерминированном порядке ключа; один rng,
    # продвигаясь по стратам в этом порядке, даёт воспроизводимый результат.
    groups = list(df.groupby(strat_cols, sort=True, dropna=False, observed=True))
    sizes = np.array([len(g) for _, g in groups], dtype=int)
    quotas = _allocate(n, sizes)

    parts: list[pd.DataFrame] = []
    for (_key, g), q in zip(groups, quotas):
        q = min(int(q), len(g))
        if q <= 0:
            continue
        # g унаследовал порядок из стабильно отсортированного df; берём
        # детерминированную перестановку и оставляем позиции в исходном порядке.
        take = np.sort(rng.permutation(len(g))[:q])
        parts.append(g.iloc[take])

    out = pd.concat(parts) if parts else df.iloc[0:0]
    return _stable_sort(out)


# =============================================================================
# Артефакт подвыборки (§6.4) — опционально: в цикле обучения подвыборка обычно
# передаётся прямо в Dataset, но сохранение полезно для воспроизводимости/отчёта.
# =============================================================================
def subset_filename(
    n: int, seed: int, *, dataset_id: str = "mixed", stratify: Sequence[str] | None = None
) -> str:
    """Самодокументирующее имя (§6.4): <dataset_id>__n<N>__seed<S>[__by-<cols>].parquet."""
    name = f"{dataset_id}__n{n}__seed{seed}"
    if stratify:
        name += "__by-" + "-".join(stratify)
    return name + ".parquet"


def save_subset(
    df: pd.DataFrame,
    paths: Paths,
    *,
    n: int,
    seed: int,
    dataset_id: str = "mixed",
    stratify: Sequence[str] | None = None,
) -> Path:
    """Сохранить подвыборку в manifests/subsets/ атомарно (§6.4, §7).

    Параметры (n, seed, stratify) кодируются в имя файла; полное описание
    прогона — в experiments/<exp>.yaml.
    """
    out = paths.subsets / subset_filename(n, seed, dataset_id=dataset_id, stratify=stratify)
    return atomic_write_parquet(df, out)