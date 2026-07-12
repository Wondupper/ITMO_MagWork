"""Стадия 0 — индексация: один проход по датасету → канонический манифест (§4).

Аудио НЕ загружается: адаптер собирает только пути/метаданные. Оркестрация тонкая —
вся датасет-специфика в адаптере (§6.2), вся проверка в валидаторе (§6.1).
Запускается независимо, по одному датасету.

Замечание по контракту вызова: §4 задаёт общий шаблон стадий как run(config), но
Стадия 0 гоняется по одному датасету, поэтому здесь параметр dataset_id явный.
Общий run(config), управляемый experiments/<exp>.yaml, актуальнее для Стадий 1–3.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config, Paths, default_config
from ..utils.io import atomic_write_parquet
from .sources import get_adapter


def build_manifest(dataset_id: str, config: Config | None = None) -> Path:
    """Собрать, провалидировать и записать полный манифест одного датасета.

    Возвращает путь записанного файла: data/manifests/<dataset_id>.parquet.
    Бросает, если адаптер не зарегистрирован или манифест не прошёл валидацию.
    """
    config = config or default_config()
    adapter = get_adapter(dataset_id, config)
    df = adapter.build_validated()
    out = config.paths.manifest_path(dataset_id)
    return atomic_write_parquet(df, out)


# --- Тонкий раннер стадии (запуск: python -m src.data.manifest ...) -----------
# Это НЕ глобальная точка входа/диспетчер (§4 её отвергает), а независимый
# запуск одной стадии по одному-нескольким датасетам.

def _summarize(df) -> str:
    """Компактная сводка манифеста для санити-проверки сразу после индексации."""
    lines = [f"    строк: {len(df)}"]
    for (split, label), n in df.groupby(["split", "label"]).size().items():
        lines.append(f"      split={split:<5} label={label}: {n}")
    lines.append(
        f"    с attack_type: {int(df['attack_type'].notna().sum())}; "
        f"speaker_id: {int(df['speaker_id'].notna().sum())}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    import pandas as pd

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
        print(f"[{ds}] записан: {out}")
        print(_summarize(pd.read_parquet(out)))

    if failed:
        print(f"\nне удалось: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())