"""Проверка окружения и данных: `python -m src.doctor` (§2).

Не стадия пайплайна, а сервисная команда: отвечает на вопрос «почему у меня не
запускается» до того, как человек полезет в код. Проверяет три вещи, в которых
ломается развёртывание на чужой машине:

1. **Пакеты.** Все ли прямые зависимости на месте и какие версии. Отдельно —
   `soundfile`: §2 требует libsndfile **1.2.0**; 1.2.2 молча не декодирует часть
   валидных FLAC ASVspoof 2021, и это видно только по `_failures.csv` через час
   прогона.
2. **torch и устройство.** `torch` живёт в экстрах `cpu`/`gpu` (см. pyproject.toml),
   поэтому «поставил не тот экстра» — самая вероятная ошибка. Здесь она видна сразу.
3. **Данные.** Какие датасеты реально лежат в `data/raw/`, для каких построен
   манифест, какие папки кэша признаков существуют. Каталоги НЕ обходятся целиком:
   у 2021 сотни тысяч файлов, полный glob здесь был бы дороже самой проверки.

Печатает отчёт и возвращает код `1`, если чего-то обязательного не хватает, —
чтобы годилось и как ручная проверка, и как первый шаг любого скрипта.

Маркеры статуса — ASCII (`[ok]`/`[!!]`/`[--]`), а не юникод-символы: консоль
Windows в cp866/cp1251 на псевдографике падает с UnicodeEncodeError, и проверка
окружения превратилась бы в собственную проблему окружения.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import Config, Paths, default_config

OK, BAD, NONE = "[ok]", "[!!]", "[--]"

#: Прямые зависимости из pyproject.toml: имя дистрибутива → зачем нужен.
REQUIRED = {
    "numpy": "массивы",
    "pandas": "манифесты",
    "pyarrow": "движок .parquet",
    "librosa": "признаки",
    "soundfile": "декодирование аудио",
    "scipy": "DCT, DET-шкала",
    "scikit-learn": "roc_curve → EER",
    "matplotlib": "графики",
    "pyyaml": "конфиги Стадий 2-3",
    "tqdm": "прогресс",
    "psutil": "лог RSS",
}

#: §2: версия libsndfile, на которой FLAC ASVspoof 2021 декодируется целиком.
GOOD_LIBSNDFILE = "1.2.0"


def _packages() -> bool:
    """Версии прямых зависимостей. True, если всё на месте."""
    print("Пакеты")
    ok = True
    for dist, why in REQUIRED.items():
        try:
            print(f"  {OK} {dist:<14} {version(dist):<12} {why}")
        except PackageNotFoundError:
            print(f"  {NONE} {dist:<14} {'НЕТ':<12} {why}")
            ok = False
    return ok


def _soundfile() -> bool:
    """Отдельная проверка libsndfile — тихий баг из §2."""
    try:
        import soundfile as sf
    except ImportError:
        return False  # уже отмечено в _packages()

    libsnd = getattr(sf, "__libsndfile_version__", "?")
    if libsnd.startswith(GOOD_LIBSNDFILE):
        print(f"  {OK} libsndfile     {libsnd}")
        return True
    print(
        f"  {BAD} libsndfile     {libsnd} — ожидается {GOOD_LIBSNDFILE} (§2).\n"
        f"       Эта версия молча не декодирует часть FLAC ASVspoof 2021.\n"
        f"       Причина обычно одна: soundfile обновили выше 0.12.1."
    )
    return False


def _torch() -> bool:
    """torch, устройство и подсказка про экстры."""
    print("\nВычисления")
    try:
        import torch
    except ImportError:
        print(
            f"  {NONE} torch          НЕТ\n"
            f"       torch ставится экстрой:  uv sync --extra cpu   (нет NVIDIA GPU)\n"
            f"                                uv sync --extra gpu   (есть NVIDIA GPU)"
        )
        return False

    cuda = torch.cuda.is_available()
    print(f"  {OK} torch          {torch.__version__}")
    if cuda:
        names = ", ".join(torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count()))
        print(f"  {OK} CUDA           доступна: {names}")
    else:
        # Не ошибка: разработка идёт на CPU (§2). Но если сборка CUDA-шная, а карты
        # нет — стоит знать, что 3 ГБ колёс лежат зря.
        build = "+cu" in torch.__version__
        mark = BAD if build else OK
        note = " (сборка CUDA-шная, но GPU не виден — вероятно нужен --extra cpu)" if build else ""
        print(f"  {mark} CUDA           недоступна, считаем на CPU{note}")

    ffmpeg = shutil.which("ffmpeg")
    print(f"  {OK if ffmpeg else NONE} ffmpeg         "
          f"{ffmpeg or 'нет — фолбэк декодирования недоступен (§2), обычно не нужен'}")
    return True


def _data(config: Config) -> None:
    """Что реально лежит на диске: сырьё, манифесты, кэш признаков."""
    from .data.sources import available

    paths: Paths = config.paths
    print(f"\nДанные (корень: {paths.root})")

    for ds in available():
        raw = paths.raw_dataset(ds)
        has_raw = raw.is_dir() and next(raw.iterdir(), None) is not None
        manifest = paths.manifest_path(ds)
        rows = _rows(manifest) if manifest.exists() else None

        mark = OK if has_raw else NONE
        raw_txt = "аудио есть" if has_raw else "нет в data/raw/"
        man_txt = f"манифест: {rows} строк" if rows is not None else "манифеста нет"
        print(f"  {mark} {ds:<20} {raw_txt:<20} {man_txt}")

    caches = sorted(p.name for p in paths.features.iterdir() if p.is_dir()) \
        if paths.features.is_dir() else []
    print(f"\nКэш признаков ({paths.features})")
    print(f"  {OK} конфигураций: {len(caches)} — {', '.join(caches)}" if caches
          else f"  {NONE} пуст — Стадия 1 ещё не запускалась")


def _rows(manifest: Path) -> int | None:
    """Число строк манифеста из метаданных parquet — без чтения самих данных."""
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(manifest).metadata.num_rows
    except Exception:  # noqa: BLE001 — диагностика не должна падать сама
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.doctor",
        description="Проверить окружение и наличие данных.",
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="корень проекта вручную (по умолчанию — от src/config.py)")
    args = parser.parse_args(argv)

    config = default_config()
    if args.root is not None:
        config = Config(paths=Paths(root=args.root.resolve()), runtime=config.runtime)

    print(f"Python {platform.python_version()} — {platform.platform()}")
    print(f"Интерпретатор: {sys.executable}\n")

    ok = _packages()
    ok = _soundfile() and ok
    ok = _torch() and ok
    _data(config)

    print("\nОкружение в порядке." if ok else
          "\nЕсть проблемы — см. пометки [!!] и [--] выше.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
