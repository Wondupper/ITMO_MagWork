"""Общие помощники адаптеров: метки, нормализация атаки, обход аудио, чтение
пробел-разделённых протоколов. Датасет-специфика остаётся в самих адаптерах.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ..schema import LABEL_BONAFIDE, LABEL_SPOOF

#: Расширения, считаемые аудио при обходе папок.
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}

#: Строки, которые в поле атаки означают «не атака» → должны стать NULL (§6.1).
_ATTACK_SENTINELS = {"", "-", "bonafide", "unknown", "none", "n/a"}


def label_to_int(s: str) -> int:
    """'spoof' → 1, 'bonafide' → 0 (иначе ошибка). Единая конвенция §6.1."""
    v = s.strip().lower()
    if v == "spoof":
        return LABEL_SPOOF
    if v == "bonafide":
        return LABEL_BONAFIDE
    raise ValueError(f"неизвестная метка {s!r} (ожидалось spoof/bonafide)")


def norm_attack(attack: str | None, label_int: int):
    """Нормализовать поле атаки в canonical attack_type или NULL.

    У подлинных записей атаки нет → NULL. Сентинелы (`-`, `bonafide`, …) тоже
    → NULL. Это снимает ловушку из Приложения А (LA bonafide=`bonafide`,
    DF bonafide=`-`) и проходит валидатор §6.1.
    """
    if label_int == LABEL_BONAFIDE:
        return pd.NA
    a = (attack or "").strip()
    if a.lower() in _ATTACK_SENTINELS:
        return pd.NA
    return a


def codec_from_suffix(path: Path):
    """Кодек контейнера файла из расширения (flac/wav/…), §6.1. NULL если пусто."""
    ext = path.suffix.lstrip(".").lower()
    return ext or pd.NA


def iter_audio(root: Path) -> Iterator[Path]:
    """Все аудиофайлы под root (рекурсивно), в стабильном порядке — детерминизм."""
    if not root.exists():
        raise FileNotFoundError(f"нет папки аудио: {root}")
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    return iter(sorted(files))


def read_protocol(path: Path) -> list[list[str]]:
    """Пробел-разделённый протокол → список полей по строкам (пустые строки — пропуск)."""
    if not path.exists():
        raise FileNotFoundError(f"нет файла протокола/ключей: {path}")
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                rows.append(parts)
    return rows