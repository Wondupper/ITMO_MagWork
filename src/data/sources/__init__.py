"""Реестр адаптеров датасетов. Импорт этого пакета регистрирует все адаптеры.

Каждый новый датасет = один модуль здесь + строка импорта ниже (§5, §6.2).
Имя модуля — валидный идентификатор (для 'for-2seconds' файл будет for_2seconds.py),
а dataset_id внутри адаптера сохраняет дефисы — он же имя папки в data/raw/.
"""
from .base import DatasetAdapter, available, get_adapter, register

# --- Конкретные адаптеры (регистрируются самим фактом импорта) ---------------
from . import asvspoof2019_la  # noqa: F401,E402
from . import asvspoof2021_la  # noqa: F401,E402
from . import asvspoof2021_df  # noqa: F401,E402
from . import for_2seconds     # noqa: F401,E402
from . import mini  # noqa: F401

__all__ = ["DatasetAdapter", "register", "get_adapter", "available"]