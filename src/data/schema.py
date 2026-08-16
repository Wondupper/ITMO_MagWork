"""Каноническая схема манифеста и валидатор (PROJECT_ARCHITECTURE.md §6.1–6.2).

Манифест — единственный интерфейс «любой датасет → пайплайн». Любой адаптер из
`sources/` обязан выдавать ровно эту схему; всё ниже манифеста опирается только
на неё (правило не-ветвления, §3). Валидатор превращает агностичность из лозунга
в проверяемый контракт (критерий №2).

Публичный API:
    coerce_schema(df)   — привести кадр адаптера к каноническому виду (колонки/типы).
    validate_manifest(df) — проверить контракт; при нарушении бросить с полным списком.
    empty_manifest()    — пустой типизированный кадр как основа для адаптера.
"""
from __future__ import annotations

import re

import pandas as pd


def _cast(col: pd.Series, dtype: str) -> pd.Series:
    """NA-безопасное приведение к каноническому dtype.

    Для numpy-float64 маршрут через to_numeric переводит pd.NA/None в NaN
    (а мусорные строки — в ошибку). Для string / nullable Int64 astype сам
    корректно держит NA. Для int64 (label, non-null) NA намеренно приводит
    к ошибке — это ловушка пропущенной метки.
    """
    if dtype == "float64":
        return pd.to_numeric(col, errors="raise").astype("float64")
    return col.astype(dtype)

# --- Конвенция меток: единая для всех адаптеров (§6.1). Это решение, не факт. ---
LABEL_BONAFIDE = 0
LABEL_SPOOF = 1
VALID_LABELS = {LABEL_BONAFIDE, LABEL_SPOOF}
VALID_SPLITS = {"train", "dev", "test"}

# utt_id становится ИМЕНЕМ ФАЙЛА в кэше признаков (§6.5), поэтому обязан быть
# безопасным для файловой системы: без разделителей пути и без выхода наверх.
_UTT_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_RESERVED_UTT_IDS = {".", ".."}

# Обязательные колонки → dtype pandas. Все non-null.
REQUIRED_DTYPES: dict[str, str] = {
    "dataset_id": "string",
    "utt_id": "string",
    "path": "string",
    "label": "int64",
    "split": "string",
}
# Опциональные колонки → dtype. NULL (pd.NA / NaN) допустим.
#
# ВНИМАНИЕ, две разные вещи с похожими именами:
#   `codec`     — формат КОНТЕЙНЕРА файла на диске (flac/wav/…), из расширения;
#   `condition` — кодек/сжатие, применённые к САМОМУ СИГНАЛУ до сохранения
#                 (напр. `low_mp3` в DF-2021, `g722` в LA-2021). Файл при этом
#                 лежит на диске как flac.
# Значения `condition`, `transmission`, `vocoder` ЛОКАЛЬНЫ для датасета — как и
# `attack_type`. Группировать по ним допустимо только внутри одного dataset_id;
# `low_mp3` из DF и `none` из LA живут в разных пространствах имён.
OPTIONAL_DTYPES: dict[str, str] = {
    "attack_type": "string",
    "condition": "string",      # условие сжатия/кодека сигнала
    "transmission": "string",   # канал передачи (LA-2021: ita_tx/loc_tx/sin_tx/mad_tx)
    "vocoder": "string",        # вокодер спуф-системы, где размечен (DF-2021)
    "speaker_id": "string",
    "sample_rate": "Int64",     # nullable integer
    "codec": "string",          # формат контейнера файла (НЕ кодек сигнала)
    "duration": "float64",
}

REQUIRED_COLUMNS = list(REQUIRED_DTYPES)
OPTIONAL_COLUMNS = list(OPTIONAL_DTYPES)
CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
_ALL_DTYPES = {**REQUIRED_DTYPES, **OPTIONAL_DTYPES}


class ManifestSchemaError(ValueError):
    """Манифест нарушает канонический контракт (§6.1)."""


def empty_manifest() -> pd.DataFrame:
    """Пустой правильно типизированный манифест — удобная основа для адаптера."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in _ALL_DTYPES.items()})


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Привести кадр адаптера к канону: добить отсутствующие опциональные колонки
    как NULL, выбрать канонический порядок, привести dtype.

    Обязательные колонки должны присутствовать. Приведение dtype — это ещё и
    ранняя ловушка ошибок адаптера: если `label` остался строкой ("spoof") или
    в обязательной колонке затесался None, astype упадёт здесь, а не на Стадии 2.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestSchemaError(f"нет обязательных колонок: {missing}")

    out = df.copy()
    for c in OPTIONAL_COLUMNS:
        if c not in out.columns:
            # Нативный NA нужного dtype: NaN для float, <NA> для string/Int64.
            out[c] = pd.Series(index=out.index, dtype=OPTIONAL_DTYPES[c])
    out = out[CANONICAL_COLUMNS]
    for c, t in _ALL_DTYPES.items():
        try:
            out[c] = _cast(out[c], t)
        except (ValueError, TypeError) as e:
            raise ManifestSchemaError(f"колонку '{c}' не привести к {t}: {e}") from e
    return out


def validate_manifest(df: pd.DataFrame, *, dataset_id: str | None = None) -> None:
    """Проверить манифест на контракт §6.1. Собирает ВСЕ нарушения и бросает одну
    ManifestSchemaError со списком (удобнее чинить пачкой, а не по одному)."""
    problems: list[str] = []

    # 1. Обязательные колонки присутствуют.
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            problems.append(f"нет обязательной колонки '{c}'")
    if problems:  # без колонок дальше проверять нечего
        raise ManifestSchemaError(_format(problems, dataset_id))

    # 2. В обязательных колонках нет NULL.
    for c in REQUIRED_COLUMNS:
        n = int(df[c].isna().sum())
        if n:
            problems.append(f"в обязательной колонке '{c}' {n} NULL")

    # 3. label ∈ {0, 1}.
    bad_labels = {int(v) for v in pd.unique(df["label"].dropna())} - VALID_LABELS
    if bad_labels:
        problems.append(
            f"недопустимые label: {sorted(bad_labels)} (должно быть {sorted(VALID_LABELS)})"
        )

    # 4. split ∈ {train, dev, test}.
    bad_splits = {str(v) for v in pd.unique(df["split"].dropna())} - VALID_SPLITS
    if bad_splits:
        problems.append(
            f"недопустимые split: {sorted(bad_splits)} (должно быть {sorted(VALID_SPLITS)})"
        )

    # 5. Пара (dataset_id, utt_id) уникальна — глобальный ключ (§3, §6.1).
    dup = df.duplicated(subset=["dataset_id", "utt_id"], keep=False)
    if bool(dup.any()):
        example = df.loc[dup, ["dataset_id", "utt_id"]].head(3).to_dict("records")
        problems.append(
            f"дубли ключа (dataset_id, utt_id): {int(dup.sum())} строк, напр. {example}"
        )

    # 6. attack_type не должен содержать сентинелов вместо NULL — прямая ловушка из
    #    Приложения А (LA bonafide = "bonafide", DF bonafide = "-" → обязаны стать NULL).
    sentinels = {"unknown", "-", "bonafide", "none", "n/a", ""}
    hits = {str(v).lower() for v in pd.unique(df["attack_type"].dropna())} & sentinels
    if hits:
        problems.append(
            f"attack_type содержит сентинелы вместо NULL: {sorted(hits)} "
            "(подлинные/неизвестные атаки должны быть NULL, §6.1)"
        )

    # 7. utt_id безопасен как имя файла кэша (§6.5). Неявный контракт Стадии 1:
    #    путь кэша — features/<hash>/<dataset_id>/<utt_id>.npy. Слэш или '..'
    #    в utt_id разложил бы кэш не туда (в худшем случае — за пределы папки),
    #    причём молча. Дешёвая проверка здесь ловит это до долгого прогона.
    bad_utt = [
        v for v in df["utt_id"].dropna()
        if not _UTT_ID_RE.match(str(v)) or str(v) in _RESERVED_UTT_IDS
    ]
    if bad_utt:
        problems.append(
            f"небезопасные utt_id: {len(bad_utt)} шт., напр. {bad_utt[:3]} "
            "(допустимы только [A-Za-z0-9._-], utt_id становится именем файла кэша)"
        )

    # 8. path — относительный путь внутри raw/<dataset_id>/, без выхода наверх.
    bad_paths = [
        v for v in df["path"].dropna()
        if str(v).startswith("/") or "\\" in str(v) or ".." in str(v).split("/")
    ]
    if bad_paths:
        problems.append(
            f"недопустимые path: {len(bad_paths)} шт., напр. {bad_paths[:3]} "
            "(ожидается относительный posix-путь внутри raw/<dataset_id>/)"
        )

    if problems:
        raise ManifestSchemaError(_format(problems, dataset_id))


def _format(problems: list[str], dataset_id: str | None) -> str:
    head = "манифест не прошёл валидацию" + (f" [{dataset_id}]" if dataset_id else "")
    return head + ":\n" + "\n".join(f"  - {p}" for p in problems)