"""Реестр архитектур и контракт модели (§6.6) — зеркало реестра экстракторов.

Смысл Стадии 2: цикл обучения (`train.py`) модель-агностичен. Он не знает про
конкретную сеть — он берёт её из реестра по имени и работает с ней через узкий
контракт. Смена архитектуры = новый класс + `@register_model`, ничего в цикле не
меняя. Это ровно та же схема, что у `features.py` для признаков.

Контракт модели (§6.6):
  • объявляет ожидаемое число каналов входа `in_channels` (= C кэша признаков);
    train.py сверяет его с признаком ДО обучения — аналог валидатора схемы (§6.1);
  • forward(x, lengths) → logits, где
        x        : [B, T, C]  float32  (батч с общей длиной T; §6.6, коллатор)
        lengths  : [B]        long     (число валидных кадров каждого примера)
        logits   : [B]        float32  (ОДИН логит на пример; спуф=1, §6.1)
  • пулинг/свёртка по оси времени T до фиксированной длины — ОБЯЗАННОСТЬ модели,
    а не кэша (§6.6): кэш хранит сырое [T, C], контракт стабилен для любых T.

`lengths` в сигнатуре присутствует всегда, даже если Стадия 2 подаёт примеры
единой длины (маска тогда тривиальна): это сохраняет интерфейс на будущее, когда
захочется честная переменная длина с маскированием паддинга — без правки моделей.
Перенос батча на устройство делает цикл обучения (единственная точка `.to(device)`,
§6.6, п. «Устройство»), поэтому модели про device ничего не знают.
"""
from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class AntispoofModel(nn.Module):
    """База архитектуры анти-спуфинга. Подкласс реализует `forward` по контракту.

    Подкласс ОБЯЗАН в `__init__` вызвать `super().__init__(in_channels)` — так C
    фиксируется единообразно и доступен train.py для сверки с признаком (§6.6).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels должно быть > 0, получено {in_channels}")
        self.in_channels = int(in_channels)

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """[B, T, C] + [B] → логиты [B] (один на пример). См. контракт в модуле."""

    def num_parameters(self) -> int:
        """Число обучаемых параметров — для лога «лёгкости» метода (§1)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Реестр (зеркало register_extractor / get_extractor из features.py)
# =============================================================================
_MODELS: dict[str, type[AntispoofModel]] = {}


def register_model(cls: type[AntispoofModel]) -> type[AntispoofModel]:
    """Декоратор: зарегистрировать класс модели по атрибуту `name`."""
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(f"{cls.__name__}: без атрибута name нельзя зарегистрировать")
    if name in _MODELS:
        raise ValueError(f"модель '{name}' уже зарегистрирована")
    _MODELS[name] = cls
    return cls


def get_model(name: str, in_channels: int, **overrides) -> AntispoofModel:
    """Собрать модель по имени под данное число каналов признаков.

    `in_channels` приходит из конфига признаков (§6.6): train.py резолвит признак
    → его C → передаёт сюда, так что первый слой модели строится под реальный C.
    `overrides` — гиперпараметры из experiments/<exp>.yaml (точка входа YAML-слоя,
    как `get_extractor(name, **overrides)`).
    """
    if name not in _MODELS:
        raise KeyError(f"нет модели '{name}'. Зарегистрированы: {available_models()}")
    return _MODELS[name](in_channels=in_channels, **overrides)


def available_models() -> list[str]:
    return sorted(_MODELS)