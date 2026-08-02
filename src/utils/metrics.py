"""Метрики детекции. EER — сквозная метрика проекта (§8).

Одна функция EER, общая для Стадии 2 (val-EER по эпохам) и Стадии 3 (итоговый EER
на тесте). Держать её в одном месте важно принципиально: если val и финал считать
разным кодом, числа окажутся несопоставимы — а сопоставимость и есть критерий №2.

Конвенция скора: БОЛЬШЕ скор ⇒ увереннее «это spoof» (положительный класс = 1,
§6.1). Модель Стадии 2 отдаёт один логит на пример; сырой логит годится как скор
напрямую (монотонная сигмоида порядок не меняет).

Зависимости — только numpy и scikit-learn (обе обязательны, §2); scipy не тянем.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.metrics import roc_curve


class EER(NamedTuple):
    """Результат: сама метрика и порог, на котором FPR = FNR."""
    eer: float
    threshold: float


def compute_eer(scores, labels) -> EER:
    """Equal Error Rate: точка, где доли ложного пропуска и ложной тревоги равны.

    Определения при положительном классе spoof=1:
      • FPR (ложная тревога) — доля bonafide (0), которым поставили скор ≥ порога;
      • FNR (ложный пропуск) — доля spoof (1) со скором < порога.
    EER — значение этих долей в пороге, где кривые FPR(θ) и FNR(θ) пересекаются.
    Порог между узлами ROC находится линейной интерполяцией по точке пересечения
    (FNR − FPR меняет знак) — оценка точнее, чем ближайший узел.

    Вырожденный случай (в батче/подвыборке представлен один класс) EER не
    определён: возвращаем NaN, не роняя прогон. На Стадии 2 это нормально для
    крохотного smoke-val — эпоха логируется с val_eer=NaN, обучение продолжается;
    итоговый EER всё равно берётся на полном тесте (§6.4).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)

    if len(np.unique(labels)) < 2:
        return EER(float("nan"), float("nan"))

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fnr - fpr  # монотонно убывает от + к − при движении по узлам ROC

    # Узел, ближайший к пересечению; при точном совпадении узла хватает и его.
    i = int(np.nanargmin(np.abs(diff)))
    if diff[i] == 0 or i + 1 >= len(diff):
        return EER(float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i]))

    # Выбираем соседний узел так, чтобы пересечение лежало между i и j.
    j = i + 1 if diff[i] * diff[i + 1] < 0 else i - 1
    if j < 0 or diff[i] * diff[j] > 0:
        return EER(float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i]))

    # Линейная интерполяция точки diff = 0 на отрезке [i, j].
    t = diff[i] / (diff[i] - diff[j])
    eer = float(fpr[i] + t * (fpr[j] - fpr[i]))
    threshold = float(thresholds[i] + t * (thresholds[j] - thresholds[i]))
    return EER(eer, threshold)