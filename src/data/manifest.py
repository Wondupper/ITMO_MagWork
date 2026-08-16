"""Стадия 0 — индексация: один проход по датасету → канонический манифест (§4).

Аудио НЕ загружается: адаптер собирает только пути/метаданные. Оркестрация тонкая —
вся датасет-специфика в адаптере (§6.2), вся проверка в валидаторе (§6.1).
Запускается независимо, по одному датасету.

Рядом с манифестом пишется **паспорт** `<dataset_id>.manifest.json` (§6.5): сводка
состава + провенанс. Мотив: числа состава данных нужны в тексте работы, а раньше
они печатались в stdout и умирали вместе с терминалом; провенанс отвечает на
вопрос «каким кодом собран лежащий на диске .parquet», на который сам .parquet
ответить не может.

Замечание по контракту вызова: §4 задаёт общий шаблон стадий как run(config), но
Стадия 0 гоняется по одному датасету, поэтому здесь параметр dataset_id явный.
Общий run(config), управляемый experiments/<exp>.yaml, актуальнее для Стадий 1–3.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Config, Paths, default_config
from ..utils.io import atomic_write_json, atomic_write_parquet
from .schema import CANONICAL_COLUMNS, LABEL_BONAFIDE, LABEL_SPOOF, OPTIONAL_COLUMNS
from .sources import get_adapter

#: Порог мощности, до которого в паспорт кладётся полный словарь значений колонки.
#: Выше — только количества: словарь на 5 000 дикторов раздул бы паспорт и ничего
#: бы не объяснил.
_VALUE_COUNT_LIMIT = 64


def build_manifest(dataset_id: str, config: Config | None = None) -> Path:
    """Собрать, провалидировать и записать полный манифест одного датасета.

    Возвращает путь записанного файла: data/manifests/<dataset_id>.parquet.
    Рядом кладёт паспорт <dataset_id>.manifest.json (§6.5).
    Бросает, если адаптер не зарегистрирован или манифест не прошёл валидацию.
    """
    config = config or default_config()
    adapter = get_adapter(dataset_id, config)
    df = adapter.build_validated()
    out = config.paths.manifest_path(dataset_id)
    atomic_write_parquet(df, out)
    passport = build_passport(df, adapter=adapter, manifest_path=out, config=config)
    atomic_write_json(passport, config.paths.manifest_passport_path(dataset_id))
    return out


# --- Паспорт манифеста (§6.5) ------------------------------------------------

def build_passport(
    df: pd.DataFrame, *, adapter, manifest_path: Path, config: Config
) -> dict:
    """Сводка состава + провенанс манифеста.

    Паспорт ПЕРЕЗАПИСЫВАЕТСЯ при каждой пересборке, в отличие от `_spec.json`
    Стадии 1, который пишется один раз. Разница по существу: `_spec.json`
    описывает папку, накапливаемую многими прогонами (и обязан помнить, каким
    прогоном что посчитано), а манифест каждый раз строится целиком заново —
    описывать он должен ровно текущий файл.
    """
    return {
        "dataset_id": adapter.dataset_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "adapter": {
            "class": type(adapter).__name__,
            "module": type(adapter).__module__,
        },
        "manifest": {
            "file": manifest_path.name,
            "rows": int(len(df)),
            "bytes": manifest_path.stat().st_size if manifest_path.exists() else None,
            "fingerprint": _fingerprint(df),
        },
        "class_balance": _class_balance(df),
        "columns": _column_summary(df),
        "provenance": _provenance(config.paths.root),
    }


def _fingerprint(df: pd.DataFrame) -> str:
    """sha1 содержимого манифеста — «тот же самый индекс или другой».

    Кадр стабильно сортируется по (dataset_id, utt_id) и сериализуется в CSV в
    каноническом порядке колонок, поэтому отпечаток не зависит от порядка строк,
    в котором адаптер обошёл файлы.

    Ограничение, о котором стоит помнить: отпечаток сравним **в пределах одной
    среды**. Смена версии pandas может изменить представление NULL или чисел в
    CSV и дать другой отпечаток при том же содержимом. Для задачи «манифест
    изменился с прошлого прогона» этого достаточно; как канонический хэш
    датасета отпечаток использовать нельзя.
    """
    ordered = df[CANONICAL_COLUMNS].sort_values(["dataset_id", "utt_id"], kind="stable")
    blob = ordered.to_csv(index=False).encode("utf-8")
    return "sha1:" + hashlib.sha1(blob).hexdigest()


def _class_balance(df: pd.DataFrame) -> dict:
    """{split: {bonafide, spoof, total}} — то, что идёт в таблицу состава данных."""
    names = {LABEL_BONAFIDE: "bonafide", LABEL_SPOOF: "spoof"}
    out: dict[str, dict[str, int]] = {}
    for (split, label), n in df.groupby(["split", "label"], observed=True).size().items():
        row = out.setdefault(str(split), {"bonafide": 0, "spoof": 0, "total": 0})
        row[names[int(label)]] = int(n)
        row["total"] += int(n)
    return out


def _column_summary(df: pd.DataFrame) -> dict:
    """По каждой опциональной колонке: заполненность и словарь значений.

    Заполненность — это и есть «покрытие метаданных»: сразу видно, что у датасета
    размечено, а что NULL, не открывая parquet. Словарь значений кладётся только
    для колонок малой мощности (_VALUE_COUNT_LIMIT).
    """
    out: dict[str, dict] = {}
    for col in OPTIONAL_COLUMNS:
        series = df[col].dropna()
        info: dict = {"non_null": int(len(series)), "n_unique": int(series.nunique())}
        if 0 < info["n_unique"] <= _VALUE_COUNT_LIMIT:
            counts = series.value_counts()
            info["values"] = {str(k): int(v) for k, v in counts.items()}
        out[col] = info
    return out


def _provenance(root: Path) -> dict:
    """Чем и когда собрано — ответ на вопрос «каким кодом получен этот файл».

    Коммит берётся best-effort: вне git-репозитория или без установленного git
    ключи просто становятся None. Падать из-за отсутствия провенанса было бы
    хуже, чем остаться без него.
    """
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        **_git_state(root),
    }


def _git_state(root: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return res.stdout.strip() if res.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "git_commit": commit,
        # None (git недоступен) и False (дерево чистое) — разные вещи, не схлопывать.
        "git_dirty": None if status is None else bool(status),
    }


# --- Тонкий раннер стадии (запуск: python -m src.data.manifest ...) -----------
# Это НЕ глобальная точка входа/диспетчер (§4 её отвергает), а независимый
# запуск одной стадии по одному-нескольким датасетам.

def _summarize(passport: dict) -> str:
    """Компактная сводка для санити-проверки сразу после индексации.

    Считается из паспорта, а не из кадра заново: иначе те же числа считались бы
    двумя путями и могли бы разойтись — ровно та ошибка, которую §6 запрещает
    ноутбукам («не пересчитывать, а читать артефакт»).
    """
    lines = [f"    строк: {passport['manifest']['rows']}"]
    for split, row in sorted(passport["class_balance"].items()):
        lines.append(
            f"      split={split:<5} bonafide: {row['bonafide']:<8} spoof: {row['spoof']}"
        )
    filled = [
        f"{c}: {info['non_null']}"
        for c, info in passport["columns"].items()
        if info["non_null"]
    ]
    lines.append("    заполнено — " + ("; ".join(filled) if filled else "ничего"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from .sources import available

    parser = argparse.ArgumentParser(
        prog="python -m src.data.manifest",
        description="Стадия 0 (индексация): построить канонический манифест(ы).",
    )
    parser.add_argument("datasets", nargs="*", help="dataset_id (один или несколько)")
    parser.add_argument("--all", action="store_true", help="обработать все зарегистрированные датасеты")
    parser.add_argument("--list", action="store_true", help="показать доступные dataset_id и выйти")
    parser.add_argument("--root", type=Path, default=None,
                        help="корень проекта (по умолчанию — определяется автоматически)")
    args = parser.parse_args(argv)

    if args.list:
        print("Доступные датасеты:")
        for d in available():
            print(f"  {d}")
        return 0

    targets = available() if args.all else args.datasets
    if not targets:
        parser.error("укажите dataset_id, либо --all, либо --list")

    config = Config(paths=Paths(root=args.root)) if args.root else default_config()

    failed: list[str] = []
    for ds in targets:
        print(f"[{ds}] индексация…", flush=True)
        try:
            out = build_manifest(ds, config)
        except Exception as e:  # noqa: BLE001 — раннеру важно не падать на первом же датасете
            print(f"[{ds}] ОШИБКА: {type(e).__name__}: {e}")
            failed.append(ds)
            continue
        passport_path = config.paths.manifest_passport_path(ds)
        print(f"[{ds}] записан: {out}")
        print(f"[{ds}] паспорт: {passport_path}")
        print(_summarize(json.loads(passport_path.read_text(encoding="utf-8"))))

    if failed:
        print(f"\nне удалось: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())