"""Единый конфиг проекта: пути, seed, параметры среды выполнения.

Принцип (PROJECT_ARCHITECTURE.md §2, §3): device / batch_size / num_workers —
это *параметры*, а не константы в коде, чтобы проект оставался переносимым на GPU
без переписывания. На Стадии 0 (индексация) используется только `paths`; секцию
`runtime` подключают Стадии 1–3. Секции признаков и обучения появятся здесь,
когда дойдём до соответствующих стадий, — заранее их не заводим.

Отношение к `experiments/<exp>.yaml`: YAML-слой оверрайдов относится к Стадиям 1–3
(гиперпараметры признаков/обучения). Для Стадии 0 достаточно путей и дефолтов —
загрузчик YAML добавим тогда, когда появится что оверрайдить.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Корень проекта = папка над src/ (src/config.py -> project/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Все пути выводятся из одного корня — переносимо между машинами."""

    root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def manifests(self) -> Path:
        return self.root / "data" / "manifests"

    @property
    def subsets(self) -> Path:
        return self.manifests / "subsets"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features"

    @property
    def experiments(self) -> Path:
        return self.root / "experiments"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    def raw_dataset(self, dataset_id: str) -> Path:
        """Корень сырого аудио датасета: data/raw/<dataset_id>/ (родная структура)."""
        return self.raw / dataset_id

    def manifest_path(self, dataset_id: str) -> Path:
        """Полный манифест датасета: data/manifests/<dataset_id>.parquet."""
        return self.manifests / f"{dataset_id}.parquet"


@dataclass(frozen=True)
class Runtime:
    """Единственное место, знающее про железо (§3, п.6). На Стадии 0 не используется."""

    device: str = "cpu"      # "cpu" | "cuda"
    batch_size: int = 32
    num_workers: int = 0     # на CPU загрузка конкурирует с обучением за ядра (§7)
    seed: int = 1337


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    runtime: Runtime = field(default_factory=Runtime)


def default_config() -> Config:
    """Конфиг по умолчанию (пути от PROJECT_ROOT, CPU)."""
    return Config()