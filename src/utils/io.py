"""Утилиты ввода-вывода. Атомарная запись — общий строительный блок (§7)."""
from __future__ import annotations

import os
from pathlib import Path

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