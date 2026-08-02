"""Baseline-модель Стадии 2: statistics-pooling классификатор.

Это НАПОЛНЕНИЕ, а не выбор архитектуры (§1, §9): нужна одна разумная лёгкая сеть,
чтобы проект прошёл end-to-end и цикл обучения был проверен. Любая следующая
архитектура добавляется рядом новым классом + `@register_model`, ничего в цикле
не меняя (§6.6). Почему именно эта в качестве первой:

  • лёгкая и дешёвая на CPU — согласуется с ракурсом новизны «лёгкий метод» (§1),
    и служит честной точкой отсчёта, от которой измеряется улучшение (§8);
  • C-агностична: первый слой строится под фактический `in_channels` признака, так
    что LFCC-60, logmel-80 и любой будущий признак работают без правок (§6.6);
  • time-агностична: statistics pooling (mean+std по времени) сворачивает любую
    длину T в вектор фиксированного размера — пулинг по T на стороне модели, как
    требует контракт (§6.6), кэш остаётся сырым [T, C].

Конвейер: [B,T,C] → покадровый Linear+ReLU (проекция C→H) → масочный
mean/std-pooling по T → [B,2H] → MLP-голова → один логит [B]. Пулинг уважает
`lengths` (усредняет только по валидным кадрам) — при единой длине Стадии 2 маска
тривиальна, но код готов к будущей переменной длине (§6.6).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import AntispoofModel, register_model


def _masked_stats(x: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5):
    """Mean и std по времени только для валидных кадров.

    x: [B, T, H], lengths: [B]. Возвращает (mean, std) по [B, H]. Маскирование
    делает пулинг корректным при паддинге; при lengths == T маска — все единицы.
    """
    b, t, _ = x.shape
    # [B, T]: True там, где кадр валиден (индекс < длины примера).
    idx = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
    mask = (idx < lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)  # [B, T, 1]
    n = mask.sum(dim=1).clamp_min(1.0)                             # [B, 1]
    mean = (x * mask).sum(dim=1) / n
    var = ((x - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / n
    std = torch.sqrt(var.clamp_min(eps))
    return mean, std


@register_model
class StatPoolMLP(AntispoofModel):
    """Лёгкий stat-pooling классификатор (см. модуль). Гиперпараметры — из YAML."""

    #: Ключ реестра; совпадает с полем model.name в experiments/<exp>.yaml.
    name = "statpool_mlp"

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__(in_channels)
        self.hidden = int(hidden)

        # Покадровая проекция C → H (общая для всех кадров).
        self.frame = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Голова над [mean; std] (размер 2H) → один логит.
        self.head = nn.Sequential(
            nn.Linear(2 * self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] → покадровая проекция → [B, T, H]
        h = self.frame(x)
        mean, std = _masked_stats(h, lengths)          # оба [B, H]
        pooled = torch.cat([mean, std], dim=1)         # [B, 2H]
        return self.head(pooled).squeeze(-1)           # [B] — один логит на пример