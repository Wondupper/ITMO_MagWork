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

# Обязательные колонки → dtype pandas. Все non-null.
REQUIRED_DTYPES: dict[str, str] = {
    "dataset_id": "string",
    "utt_id": "string",
    "path": "string",
    "label": "int64",
    "split": "string",
}
# Опциональные колонки → dtype. NULL (pd.NA / NaN) допустим.
OPTIONAL_DTYPES: dict[str, str] = {
    "attack_type": "string",
    "speaker_id": "string",
    "sample_rate": "Int64",   # nullable integer
    "codec": "string",
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

    if problems:
        raise ManifestSchemaError(_format(problems, dataset_id))


def _format(problems: list[str], dataset_id: str | None) -> str:
    head = "манифест не прошёл валидацию" + (f" [{dataset_id}]" if dataset_id else "")
    return head + ":\n" + "\n".join(f"  - {p}" for p in problems)