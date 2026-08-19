"""Сборка `mini` — крошечного датасета для smoke-прогонов (§6.4, v15).

Зачем. Механизма подвыборок в проекте нет (§6.4): объём прогона задаётся составом
селекторов. Но проверять «работает ли цепочка Стадия 1 → 2 → 3 вообще» на
десятках тысяч файлов дорого. Поэтому есть отдельный маленький ДАТАСЕТ: несколько
десятков реальных записей, взятых по чуть-чуть из каждого источника.

Что делает. Читает уже построенные манифесты (Стадия 0) и по каждому источнику
берёт **5 подлинных + 5 спуф** записей, копируя аудио в `data/raw/mini/<источник>/`
и дописывая строку в `data/raw/mini/_index.csv` — «родной протокол» получившегося
датасета. Дальше `mini` индексируется обычной Стадией 0 (`sources/mini.py`), и все
стадии ниже работают с ним как с любым другим датасетом — ни одна из них про его
происхождение не знает (§3).

Два решения, которые стоит проговорить.

**Почему 5+5, а не «первые 10».** Оба класса обязаны присутствовать в каждом
сплите: `compute_eer` на одноклассовой выборке возвращает NaN (§8), а `pos_weight`
отключается. Прогон при этом формально «не падает», но именно числовой путь —
скор → метка → EER — остаётся непроверенным, а это самое ценное в smoke-тесте.
Разделимость классов при этом НЕ подразумевается: записи берутся как есть, EER на
`mini` — случайное число около 0.5, и интерпретировать его нельзя (см. ниже).

**Почему отдельная команда, а не адаптер.** Адаптер по контракту (§6.2) только
читает `data/raw/<ds>/` и не грузит аудио. Здесь же надо копировать файлы и читать
ЧУЖИЕ манифесты. Разделение сохраняет контракт: генератор материализует папку,
адаптер её индексирует, как любую другую.

⚠️ **Никаких результатов с `mini` в работу.** Записи берутся из всех сплитов
источников, включая тестовые, и переразмечаются в train/dev/test заново — то есть
там заведомо есть утечка. Это инструмент проверки кода, а не измерения.

Запуск (модуль пакета, из корня проекта):
    python -m src.data.mini                       # из всех датасетов с манифестом
    python -m src.data.mini asvspoof2019_la       # только из указанных
    python -m src.data.mini --force               # пересобрать с нуля
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import Config, Paths, default_config
from .schema import LABEL_BONAFIDE, LABEL_SPOOF

#: dataset_id получившегося датасета; совпадает с именем папки в data/raw/.
MINI_DATASET_ID = "mini"

#: Имя «родного протокола» mini — его читает адаптер sources/mini.py.
INDEX_NAME = "_index.csv"

#: Сколько записей КАЖДОГО класса берём из одного источника.
PER_CLASS = 5

#: Как эти записи (каждого класса) раскладываются по сплитам. Сумма = PER_CLASS,
#: на два класса выходит 4 train / 4 dev / 2 test с одного источника.
SPLIT_QUOTA = {"train": 2, "dev": 2, "test": 1}

#: Колонки протокола. `source_*` не нужны пайплайну, но отвечают на вопрос
#: «откуда взялся этот файл» без раскопок — провенанс в духе паспорта (§6.5).
INDEX_COLUMNS = [
    "utt_id", "path", "label", "split",
    "source_dataset_id", "source_utt_id", "attack_type",
]


@dataclass
class MiniBuildStats:
    """Итог сборки — для сводки и кода возврата раннера."""
    taken: dict[str, int]        # источник → сколько записей взято
    skipped: dict[str, str]      # источник → причина пропуска
    index_path: Path | None
    raw_dir: Path

    @property
    def total(self) -> int:
        return sum(self.taken.values())

    def summary(self) -> str:
        lines = [f"    взято: {self.total} записей из {len(self.taken)} источник(ов)"]
        for ds, n in sorted(self.taken.items()):
            lines.append(f"      {ds}: {n}")
        for ds, why in sorted(self.skipped.items()):
            lines.append(f"      ПРОПУЩЕН {ds}: {why}")
        return "\n".join(lines)


def _available_sources(paths: Paths) -> list[str]:
    """Датасеты с построенным манифестом, кроме самого mini."""
    if not paths.manifests.exists():
        return []
    return sorted(
        p.stem for p in paths.manifests.glob("*.parquet") if p.stem != MINI_DATASET_ID
    )


def _pick(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Первые PER_CLASS записей класса после стабильной сортировки по ключу.

    Выбор детерминирован без всякого seed: тот же манифест → те же файлы. Для
    инструмента проверки кода этого достаточно, а отсутствие seed — одна ручка,
    которую не надо документировать и согласовывать между прогонами.
    """
    g = df[df["label"] == label].sort_values("utt_id", kind="stable")
    return g.head(PER_CLASS)


def _assign_splits(n: int) -> list[str]:
    """Разложить n записей одного класса по сплитам согласно SPLIT_QUOTA.

    Остаток (если источник дал меньше PER_CLASS) уходит в train: лучше недобрать
    в dev/test, чем оставить сплит пустым и получить NaN-EER на пустом наборе.
    """
    out: list[str] = []
    for split, q in SPLIT_QUOTA.items():
        out.extend([split] * q)
    return (out + ["train"] * n)[:n]


def build_mini(
    config: Config | None = None,
    *,
    sources: list[str] | None = None,
    force: bool = False,
) -> MiniBuildStats:
    """Собрать data/raw/mini/ из манифестов источников. Вернуть статистику.

    Источник пропускается (не роняя сборку), если у него нет манифеста, нет
    исходного аудио или в нём меньше PER_CLASS записей хотя бы одного класса —
    ровно случай «датасет слишком мал, ничего не происходит».
    """
    config = config or default_config()
    paths = config.paths
    raw_dir = paths.raw_dataset(MINI_DATASET_ID)

    if force and raw_dir.exists():
        shutil.rmtree(raw_dir)

    targets = sources if sources else _available_sources(paths)
    taken: dict[str, int] = {}
    skipped: dict[str, str] = {}
    rows: list[dict] = []

    for ds in targets:
        mpath = paths.manifest_path(ds)
        if not mpath.exists():
            skipped[ds] = f"нет манифеста {mpath.name} — сначала Стадия 0"
            continue
        df = pd.read_parquet(mpath)

        picked = [_pick(df, LABEL_BONAFIDE), _pick(df, LABEL_SPOOF)]
        if any(len(p) < PER_CLASS for p in picked):
            n_bona, n_spoof = len(picked[0]), len(picked[1])
            skipped[ds] = (
                f"мало записей: bonafide {n_bona}, spoof {n_spoof} "
                f"(нужно по {PER_CLASS} каждого)"
            )
            continue

        src_root = paths.raw_dataset(ds)
        ds_rows: list[dict] = []
        missing: Path | None = None
        for part in picked:
            splits = _assign_splits(len(part))
            for split, row in zip(splits, part.itertuples(index=False)):
                src = src_root / str(row.path)
                if not src.exists():
                    missing = src
                    break
                ds_rows.append({
                    "utt_id": f"{ds}__{row.utt_id}",
                    "path": f"{ds}/{row.utt_id}{src.suffix}",
                    "label": int(row.label),
                    "split": split,
                    "source_dataset_id": ds,
                    "source_utt_id": str(row.utt_id),
                    "attack_type": getattr(row, "attack_type", pd.NA),
                    "_src": src,
                })
            if missing is not None:
                break
        if missing is not None:
            skipped[ds] = f"нет исходного аудио {missing}"
            continue

        for r in ds_rows:
            dst = raw_dir / r["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.pop("_src"), dst)
        rows.extend(ds_rows)
        taken[ds] = len(ds_rows)

    if not rows:
        return MiniBuildStats(taken, skipped, None, raw_dir)

    index = pd.DataFrame(rows)[INDEX_COLUMNS].sort_values("utt_id", kind="stable")
    index_path = raw_dir / INDEX_NAME
    index.to_csv(index_path, index=False)
    return MiniBuildStats(taken, skipped, index_path, raw_dir)


# =============================================================================
# Тонкий раннер (python -m src.data.mini ...) — как раннеры Стадий 0/1
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.data.mini",
        description="Собрать крошечный датасет 'mini' для smoke-прогонов.",
    )
    parser.add_argument("sources", nargs="*",
                        help="dataset_id источников (по умолчанию — все с манифестом)")
    parser.add_argument("--force", action="store_true",
                        help="удалить существующий data/raw/mini/ и собрать заново")
    parser.add_argument("--root", type=Path, default=None,
                        help="корень проекта (по умолчанию — автоопределение)")
    args = parser.parse_args(argv)

    config = Config(paths=Paths(root=args.root)) if args.root else default_config()

    stats = build_mini(config, sources=args.sources or None, force=args.force)
    print(f"[{MINI_DATASET_ID}] сборка из манифестов…")
    print(stats.summary())
    if stats.index_path is None:
        print("не набрано ни одной записи — mini не собран.")
        return 1
    print(f"[{MINI_DATASET_ID}] аудио:    {stats.raw_dir}")
    print(f"[{MINI_DATASET_ID}] протокол: {stats.index_path}")
    print(f"\nдальше: python -m src.data.manifest {MINI_DATASET_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())