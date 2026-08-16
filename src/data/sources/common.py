"""Общие помощники адаптеров: метки, нормализация атаки и условия записи, обход
аудио, чтение пробел-разделённых протоколов. Датасет-специфика остаётся в самих
адаптерах.
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

#: Строки, которые в поле УСЛОВИЯ означают «поля нет» → NULL. Список короче, чем
#: у атаки, намеренно: `none` / `nocodec` — содержательные значения условия
#: («сжатие не применялось»), а не заглушки, и обязаны сохраниться.
_CONDITION_SENTINELS = {"", "-", "n/a"}


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


def norm_condition(value: str | None):
    """Нормализовать поле условия/вокодера в строку или NULL (§6.1).

    Отличие от norm_attack принципиальное: условие относится к записи ЛЮБОГО
    класса (подлинную запись тоже могли прогнать через кодек), поэтому метка
    здесь не участвует, и `none` / `nocodec` НЕ считаются заглушкой — это
    базовая точка «без сжатия», без которой разбивка EER по условиям теряет
    точку отсчёта.
    """
    v = (value or "").strip()
    return pd.NA if v.lower() in _CONDITION_SENTINELS else v


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


def protocol_table(
    path: Path,
    *,
    fields: dict[str, int],
    key: str = "utt_id",
    min_fields: int | None = None,
    dataset_id: str = "",
) -> dict[str, dict[str, str]]:
    """Пробел-разделённый протокол → {значение key: {имя поля: значение}}.

    `fields` — отображение «имя поля → индекс в строке», напр.
    ``{"utt_id": 1, "speaker_id": 0, "attack": 4, "label": 5}``. Помощник
    намеренно НЕ знает семантики полей: какое поле что значит и где лежит файл —
    датасет-специфика (§6.2), она остаётся в адаптере, здесь только разбор.
    Поэтому же сигнатура словарная, а не фиксированный набор аргументов: новое
    поле (условие, вокодер) добавляется в адаптере, помощник не трогается.

    `min_fields` — минимальная ширина строки; по умолчанию выводится из самого
    правого запрошенного индекса. Ключ обязан быть уникальным: молчаливая
    перезапись дублей даёт манифест с недостачей строк без единой ошибки.
    """
    if key not in fields:
        raise ValueError(f"protocol_table: ключ {key!r} отсутствует в fields")
    need = min_fields if min_fields is not None else max(fields.values()) + 1
    tag = f"[{dataset_id}] " if dataset_id else ""

    table: dict[str, dict[str, str]] = {}
    for lineno, parts in enumerate(read_protocol(path), 1):
        if len(parts) < need:
            raise ValueError(
                f"{tag}{path.name}:{lineno}: ожидалось >= {need} полей, "
                f"получено {len(parts)}: {parts}"
            )
        row = {name: parts[i] for name, i in fields.items()}
        utt = row.pop(key)
        if utt in table:
            raise ValueError(f"{tag}{path.name}:{lineno}: дубль ключа {utt!r}")
        table[utt] = row
    return table