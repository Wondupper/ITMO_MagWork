"""Контракт адаптера датасета и реестр (PROJECT_ARCHITECTURE.md §6.2, §3).

Адаптер инкапсулирует ВСЁ датасет-специфичное и наружу выдаёт только канонический
манифест (§6.1). Ниже манифеста никакого ветвления по dataset_id — это плата за
контракт. Реестр (`@register`) позволяет выбирать адаптер по dataset_id без
if-цепочек: новый датасет = новый модуль + строка импорта в sources/__init__.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ...config import Config
from ..schema import coerce_schema, validate_manifest


class DatasetAdapter(ABC):
    """База адаптера. Подкласс обязан задать `dataset_id` и реализовать `build()`."""

    #: Совпадает с именем папки в data/raw/ и со значением колонки dataset_id.
    dataset_id: str = ""

    def __init__(self, config: Config):
        if not self.dataset_id:
            raise ValueError(f"{type(self).__name__}: не задан атрибут dataset_id")
        self.config = config

    @property
    def raw_dir(self) -> Path:
        """data/raw/<dataset_id>/ — родная структура датасета, только чтение."""
        return self.config.paths.raw_dataset(self.dataset_id)

    @abstractmethod
    def build(self) -> pd.DataFrame:
        """Собрать манифест этого датасета. Обязанности адаптера (§6.2):
          (1) найти аудио в self.raw_dir;
          (2) разрешить метку на файл (протокол / имя папки / шаблон имени),
              закодировать как LABEL_SPOOF=1 / LABEL_BONAFIDE=0;
          (3) разрешить родной сплит и нормализовать в {train, dev, test},
              либо назначить сплит по явной политике (§6.3);
          (4) заполнить опциональные поля где есть, поставить NULL где нет.

        Достаточно вернуть обязательные колонки — coerce_schema() добьёт
        опциональные как NULL. Аудио НЕ загружать (§4): только пути/метаданные.
        """

    def build_validated(self) -> pd.DataFrame:
        """build() → канонизация типов → валидация. Наружу гарантированно
        контрактный манифест."""
        df = coerce_schema(self.build())
        if not bool((df["dataset_id"] == self.dataset_id).all()):
            raise ValueError(
                f"адаптер '{self.dataset_id}': в колонке dataset_id чужие значения"
            )
        validate_manifest(df, dataset_id=self.dataset_id)
        return df


# --- Реестр -----------------------------------------------------------------
_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Декоратор: зарегистрировать класс адаптера по его dataset_id."""
    if not cls.dataset_id:
        raise ValueError(f"{cls.__name__}: без dataset_id нельзя зарегистрировать")
    if cls.dataset_id in _REGISTRY:
        raise ValueError(f"адаптер для '{cls.dataset_id}' уже зарегистрирован")
    _REGISTRY[cls.dataset_id] = cls
    return cls


def get_adapter(dataset_id: str, config: Config) -> DatasetAdapter:
    if dataset_id not in _REGISTRY:
        raise KeyError(
            f"нет адаптера для '{dataset_id}'. Зарегистрированы: {available()}"
        )
    return _REGISTRY[dataset_id](config)


def available() -> list[str]:
    """Список зарегистрированных dataset_id."""
    return sorted(_REGISTRY)