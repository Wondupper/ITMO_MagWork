"""Селекторы манифестов: список {dataset_id, split} → склеенный DataFrame (§3, §6.1).

Общий кусок Стадий 2 и 3. Обе читают одни и те же манифесты по одинаковым правилам:
взять нужные сплиты нужных датасетов, склеить (`concat`), отдать вниз канонические
колонки. Разница только в наборе колонок: обучению хватает меток, оценке нужен ещё
`attack_type` для разбивки EER по атакам (§8).

Держать это в одном месте важно не из вкуса к DRY, а по существу: `concat` и
`group-by` — единственные законные употребления `dataset_id` ниже манифеста
(правило не-ветвления, §3). Пока такая функция одна, правило видно и проверяемо;
две её копии в разных стадиях начали бы расходиться.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from ..config import Paths
from .subsets import DEFAULT_STRATIFY

#: Колонки, нужные любой стадии ниже манифеста (§7: не тянем лишнего).
BASE_COLUMNS = ["dataset_id", "utt_id", "label", "split"]


def load_selectors(
    paths: Paths,
    selectors: list[dict],
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Прочитать нужные строки манифеста(ов) по списку (dataset_id, split) и склеить.

    Параметры
    ---------
    selectors : список словарей {dataset_id, split} из experiments/<exp>.yaml.
    extra_columns : колонки сверх BASE_COLUMNS (стратификация подвыборки,
        `attack_type` для разбивки EER и т.п.). Дубли и None отбрасываются.

    Единственные операции над dataset_id — фильтр по split и concat (§3):
    никакого `if dataset_id == …`. Пустой результат селектора — ошибка, а не
    молчаливый пропуск: это почти всегда опечатка в конфиге.
    """
    need = list(dict.fromkeys([*BASE_COLUMNS, *(c for c in extra_columns if c)]))
    frames = []
    for sel in selectors:
        ds, sp = sel["dataset_id"], sel["split"]
        mpath = paths.manifest_path(ds)
        if not mpath.exists():
            raise FileNotFoundError(
                f"нет манифеста {mpath} — сначала Стадия 0: "
                f"python -m src.data.manifest {ds}"
            )
        df = pd.read_parquet(mpath, columns=need)
        df = df[df["split"] == sp]
        if df.empty:
            raise ValueError(f"[{ds}] нет строк со split='{sp}' — проверьте селектор.")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def stratify_for(yaml_stratify: Sequence[str] | None, selectors: list[dict]) -> tuple[str, ...]:
    """Колонки стратификации подвыборки: из YAML или дефолт (§6.4).

    При склейке нескольких датасетов `dataset_id` добавляется обязательно —
    иначе подвыборка может перекосить состав по датасетам (§6.4).
    """
    strat = tuple(yaml_stratify) if yaml_stratify else DEFAULT_STRATIFY
    datasets = {s["dataset_id"] for s in selectors}
    if len(datasets) > 1 and "dataset_id" not in strat:
        strat = ("dataset_id", *strat)
    return strat