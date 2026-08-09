"""Инференс-проход: загрузчик → скоры. Общий для Стадий 2 и 3 (§8).

Зачем отдельным модулем. Val-EER Стадии 2 и итоговый EER Стадии 3 должны быть
сопоставимы — это прямо критерий №2. Общая метрика (`utils/metrics.compute_eer`)
половина дела; вторая половина — чтобы скоры, которые в неё подаются, получались
ОДНИМ кодом. Иначе разойтись могут не формулы, а мелочи прохода (что считается
скором, в каком режиме модель, что происходит с последним неполным батчем).

Скор = сырой логит модели; больше ⇒ увереннее «спуф» (§6.1, конвенция
`metrics.compute_eer`). Сигмоида не применяется: EER инвариантен к монотонному
преобразованию, а логит удобнее хранить (нет насыщения в 0/1).

Порядок. При `shuffle=False` и `drop_last=False` порядок выдачи загрузчика
совпадает с порядком примеров в Dataset — на этом инварианте Стадия 3 связывает
скор с ключом (dataset_id, utt_id), не заводя id в сам батч.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch


@torch.no_grad()
def iter_batch_scores(model, loader, device) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Пройти загрузчик, отдавая (скоры, метки) по одному батчу.

    Батчевая форма нужна Стадии 3: она пишет промежуточные шарды скоров, чтобы
    прерванный прогон продолжался с невыполненного (§7). Стадии 2 достаточно
    обёртки `predict_scores`.
    """
    model.eval()
    for x, y, lengths in loader:
        logits = model(x.to(device), lengths.to(device))
        yield (
            logits.detach().float().cpu().numpy().reshape(-1),
            y.detach().cpu().numpy().reshape(-1),
        )


def predict_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Скоры и метки по всему загрузчику: (scores [N], labels [N])."""
    scores, labels = [], []
    for s, y in iter_batch_scores(model, loader, device):
        scores.append(s)
        labels.append(y)
    if not scores:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    return np.concatenate(scores), np.concatenate(labels)