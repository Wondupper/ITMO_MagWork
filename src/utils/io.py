"""Утилиты ввода-вывода. Атомарная запись — общий строительный блок (§7)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Записать parquet атомарно: временный файл в той же папке, затем os.replace.

    Падение посреди записи не оставит битый .parquet, который позже сочли бы
    готовым. Тот же паттерн критичен для кэша признаков и чекпойнтов (§7);
    здесь он же обслуживает единичную запись манифеста.

    Требует движок parquet (pyarrow) — см. requirements.txt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # атомарно в пределах одной ФС
    return path

def atomic_write_json(doc: dict, path: str | Path) -> Path:
    """Записать JSON атомарно (§7); стиль как у `_spec.json` Стадии 1.

    `sort_keys=True` — чтобы диффы между прогонами показывали изменившиеся
    значения, а не переставленные ключи; `ensure_ascii=False` — чтобы русские
    строки читались глазами, а не как \\uXXXX.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp, path)  # атомарно в пределах одной ФС
    return path


def atomic_write_npy(arr: np.ndarray, path: str | Path) -> Path:
    """Записать .npy атомарно: временный файл рядом, затем os.replace (§7).

    Тот же принцип, что и atomic_write_parquet: падение посреди записи не оставит
    полу-записанный .npy, который проверка «существует → пропустить» (Стадия 1)
    ошибочно сочла бы готовым кэшем.

    np.save() дописывает '.npy' к ИМЕНИ файла, если передать путь-строку без
    этого расширения; поэтому пишем через открытый дескриптор — так суффикс
    временного файла (.npy.tmp) не искажается.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)  # атомарно в пределах одной ФС
    return path