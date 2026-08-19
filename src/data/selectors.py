"""Селекторы манифестов: список {dataset_id, split} → склеенный DataFrame (§3, §6.1).

Общий кусок Стадий 2 и 3. Обе читают одни и те же манифесты по одинаковым правилам:
взять нужные сплиты нужных датасетов, склеить (`concat`), отдать вниз канонические
колонки. Разница только в наборе колонок: обучению хватает меток, оценке нужен ещё
`attack_type` для разбивки EER по атакам (§8).

Держать это в одном месте важно не из вкуса к DRY, а по существу: `concat` и
`group-by` — единственные законные употребления `dataset_id` ниже манифеста
(правило не-ветвления, §3). Пока такая функция одна, правило видно и проверяемо;
две её копии в разных стадиях начали бы расходиться.

Объём прогона задаётся ТОЛЬКО составом селекторов (какие датасеты и сплиты) — это
единственная ручка (§6.4, v15). Механизма подвыборок нет: маленький прогон = маленький
датасет (`synthetic_mini`), а не срез большого.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..config import Paths

#: Колонки, нужные любой стадии ниже манифеста (§7: не тянем лишнего).
BASE_COLUMNS = ["dataset_id", "utt_id", "label", "split"]

#: Ключ, по которому склеенный манифест стабильно сортируется (§3).
_KEY = ["dataset_id", "utt_id"]


def load_selectors(
    paths: Paths,
    selectors: list[dict],
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Прочитать нужные строки манифеста(ов) по списку (dataset_id, split) и склеить.

    Параметры
    ---------
    selectors : список словарей {dataset_id, split} из experiments/<exp>.yaml.
    extra_columns : колонки сверх BASE_COLUMNS (напр. `attack_type` для разбивки
        EER по атакам). Дубли и None отбрасываются.

    Единственные операции над dataset_id — фильтр по split и concat (§3):
    никакого `if dataset_id == …`. Пустой результат селектора — ошибка, а не
    молчаливый пропуск: это почти всегда опечатка в конфиге.

    Результат стабильно отсортирован по (dataset_id, utt_id). Сортировка здесь
    не косметика: порядок строк определяет порядок примеров в Dataset, а значит и
    соответствие «скор ↔ ключ» на Стадии 3 (§7). Раньше её обеспечивал модуль
    подвыборок; после его удаления (§6.4, v15) инвариант живёт тут.
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
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(_KEY, kind="stable").reset_index(drop=True)